"""Functional local SQL/edge sync backend with ACID SQLite commits.
This is the portable backend; it is not Delta Lake or Spark.
"""
import sqlite3,json,hashlib,datetime,gzip
from pathlib import Path
import pandas as pd

SCHEMA='''CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, source TEXT NOT NULL, timestamp TEXT NOT NULL, region TEXT NOT NULL, category TEXT NOT NULL, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS commits(version INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT UNIQUE NOT NULL, committed_at TEXT NOT NULL, record_count INTEGER NOT NULL, sha256 TEXT NOT NULL);'''
class SQLLakehouse:
    def __init__(self,path):
        self.path=str(path);Path(path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as con:con.executescript(SCHEMA)
    def connect(self):return sqlite3.connect(self.path)
    def ingest(self,records,batch_id):
        clean=[]
        for rec in records:
            try:
                r={k:rec[k] for k in ('event_id','source','timestamp','region','category','value')}
                r['timestamp']=pd.Timestamp(r['timestamp'],tz='UTC' if not pd.Timestamp(r['timestamp']).tzinfo else None).isoformat()
                r['value']=float(r['value']);r={k:str(v) if k!='value' else v for k,v in r.items()}
                if not r['event_id'] or not r['source']:continue
                clean.append(r)
            except (KeyError,TypeError,ValueError):continue
        data=json.dumps(clean,sort_keys=True).encode();digest=hashlib.sha256(data).hexdigest()
        with self.connect() as con:
            prior=con.execute('SELECT sha256 FROM commits WHERE batch_id=?',(batch_id,)).fetchone()
            if prior:
                if prior[0]!=digest:raise ValueError('Batch ID reused with changed content')
                return 0
            before=con.total_changes
            con.executemany('INSERT OR IGNORE INTO events VALUES(:event_id,:source,:timestamp,:region,:category,:value)',clean)
            inserted=con.total_changes-before
            con.execute('INSERT INTO commits(batch_id,committed_at,record_count,sha256) VALUES(?,?,?,?)',
                        (batch_id,datetime.datetime.now(datetime.timezone.utc).isoformat(),inserted,digest))
            return inserted
    def query(self,sql,params=()):
        if not sql.lstrip().lower().startswith('select'):raise ValueError('Read-only SELECT statements only')
        with self.connect() as con:return pd.read_sql_query(sql,con,params=params)
    def snapshot(self):
        with self.connect() as con:return con.execute('SELECT COALESCE(MAX(version),0) FROM commits').fetchone()[0]

class EdgeAgent:
    def __init__(self,node_id,root,cloud):
        self.node_id=node_id;self.local=SQLLakehouse(Path(root)/f'edge_{node_id}.sqlite');self.cloud=cloud
        self.pending=[];self.sent_batches=set()
    def receive(self,records,batch_id):
        accepted=self.local.ingest(records,batch_id)
        if accepted:self.pending.append((batch_id,list(records)))
        return accepted
    def sync(self,network_up=True):
        if not network_up:return 0
        total=0
        while self.pending:
            batch,records=self.pending[0]
            total+=self.cloud.ingest(records,f'edge-{self.node_id}-{batch}')
            self.pending.pop(0)
        return total

def demo_data(n=100,seed=42):
    import numpy as np
    rng=np.random.default_rng(seed)
    start=pd.Timestamp('2025-01-01',tz='UTC')
    return [{'event_id':f'evt-{i}','source':('taxi','debs','gdelt')[i%3],
             'timestamp':(start+pd.Timedelta(seconds=i)).isoformat(),
             'region':f'r{i%4}','category':f'c{i%5}','value':float(rng.uniform(0,100))} for i in range(n)]

def validate_hybrid_execution(root):
    cloud=SQLLakehouse(Path(root)/'cloud.sqlite')
    edges=[EdgeAgent(i,root,cloud) for i in range(4)]
    records=demo_data()
    for i,e in enumerate(edges):e.receive(records[i::4],f'batch-{i}')
    local_total=sum(int(e.local.query('SELECT COUNT(*) AS n FROM events').iloc[0]['n']) for e in edges)
    assert local_total==len(records)
    transferred=sum(e.sync() for e in edges)
    assert transferred==len(records)
    assert sum(e.sync() for e in edges)==0
    cloud_total=int(cloud.query('SELECT COUNT(*) AS n FROM events').iloc[0]['n'])
    assert cloud_total==len(records)
    result=cloud.query('SELECT source, COUNT(*) AS n, SUM(value) AS total FROM events GROUP BY source ORDER BY source')
    return {'local_rows':local_total,'cloud_rows':cloud_total,'synchronized':transferred,'snapshot_version':cloud.snapshot(),'query_results':result.to_dict('records')}
