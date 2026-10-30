"""Adapters for user-supplied authentic NYC Taxi, DEBS and GDELT exports.
Never silently invents rows to fill missing public datasets.
"""
from pathlib import Path
import pandas as pd

def read_dataset(path):
    path=Path(path)
    if not path.exists():raise FileNotFoundError(f'Dataset not found: {path}')
    if path.suffix.lower()=='.parquet':return pd.read_parquet(path)
    if path.suffix.lower() in ('.csv','.gz','.tsv'):
        sep='\t' if path.suffix.lower()=='.tsv' else ','
        return pd.read_csv(path,sep=sep,low_memory=False)
    raise ValueError(f'Expected CSV/TSV/Parquet: {path}')

def normalize(path,dataset,out):
    df=read_dataset(path).drop_duplicates()
    dataset=dataset.lower()
    if dataset=='taxi':
        aliases={'timestamp':['tpep_pickup_datetime','lpep_pickup_datetime','pickup_datetime'],
                 'region':['PULocationID','pickup_longitude'], 'value':['trip_distance','total_amount']}
    elif dataset=='gdelt':
        aliases={'timestamp':['SQLDATE','Day','date'], 'region':['ActionGeo_CountryCode','Actor1CountryCode','country'],
                 'value':['GoldsteinScale','NumMentions','value']}
    elif dataset=='debs':
        aliases={'timestamp':['timestamp','time','ts'], 'region':['sensor','sensor_id','house_id'],
                 'value':['value','measurement','reading']}
    else:raise ValueError('dataset must be taxi, gdelt or debs')
    mapping={}
    for key,variants in aliases.items():
        found=next((x for x in variants if x in df.columns),None)
        if not found:raise ValueError(f'Missing {key} column; detected {list(df.columns)[:20]}')
        mapping[key]=found
    outdf=pd.DataFrame()
    raw_time=df[mapping['timestamp']].astype(str)
    if dataset=='gdelt' and raw_time.str.fullmatch(r'\d{8}').all():
        outdf['timestamp']=pd.to_datetime(raw_time,format='%Y%m%d',utc=True,errors='coerce')
    else:outdf['timestamp']=pd.to_datetime(raw_time,utc=True,errors='coerce')
    outdf['region']=df[mapping['region']].astype(str)
    outdf['value']=pd.to_numeric(df[mapping['value']],errors='coerce')
    outdf['source']=dataset
    outdf['category']=df['EventCode'].astype(str) if dataset=='gdelt' and 'EventCode' in df else dataset
    outdf=outdf.dropna(subset=['timestamp','region','value']).sort_values('timestamp').reset_index(drop=True)
    outdf.insert(0,'event_id',[f'{dataset}-{i}' for i in range(len(outdf))])
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.suffix=='.parquet':outdf.to_parquet(out,index=False)
    else:outdf.to_csv(out,index=False)
    return {'input_rows':int(len(df)),'retained_rows':int(len(outdf)),'out':str(out),'column_mapping':mapping}
