"""Reproducible logical hybrid cloud/edge workload environment.
Times and costs are modelled, not physical cluster measurements.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import numpy as np

ACTIONS=('edge','cloud','split','cache','replicate')
@dataclass
class Query:
    qid:int
    dataset:str
    bytes_in:int
    joins:int
    aggregation:float
    partitions:tuple
    sla_ms:float
    arrival_s:float
    selectivity:float
    intent:str
    exact:bool=True

@dataclass
class Node:
    cpu_capacity:float
    memory_gb:float
    queue_end:float=0.

@dataclass
class EnvironmentConfig:
    bandwidth_mbps:float=100.
    rtt_ms:float=20.
    edge_cpu:float=4.
    cloud_cpu:float=16.
    edge_mem_gb:float=8.
    cloud_mem_gb:float=64.
    cache_gb:float=1.
    seed:int=42
    cost_edge_cpu_ms:float=.000020
    cost_cloud_cpu_ms:float=.000035
    cost_network_mb:float=.003
    cost_storage_mb:float=.0004
    cost_rtt:float=.00002
    max_wait_ms:float=60000.

@dataclass
class Outcome:
    latency_ms:float
    cost:float
    transfer_bytes:int
    sla_met:bool
    feasible:bool
    action:str
    cache_hit:bool=False
    rejected:bool=False
    finished_at:float=0.
    components:dict=field(default_factory=dict)

class HybridEnvironment:
    def __init__(self,cfg:EnvironmentConfig):
        self.cfg=cfg
        self.rng=np.random.default_rng(cfg.seed)
        self.reset()
    def reset(self):
        self.edge=[Node(self.cfg.edge_cpu,self.cfg.edge_mem_gb) for _ in range(4)]
        self.cloud=Node(self.cfg.cloud_cpu,self.cfg.cloud_mem_gb)
        self.cache=OrderedDict()
        self.cache_bytes=0
        self.replicas=set()
        self.partition_reads=defaultdict(int)
        self.history=[]
        self.last_time=0.
        return self
    def locality(self,q):
        return sum(p in self.replicas or p%4==q.qid%4 for p in q.partitions)/max(len(q.partitions),1)
    def cache_key(self,q):
        return (q.dataset,tuple(q.partitions),q.joins,round(q.aggregation,3),round(q.selectivity,3))
    def mask(self,q):
        loc=self.locality(q)
        size_gb=q.bytes_in/1e9
        edge_ok=size_gb <= self.cfg.edge_mem_gb*.8 and q.joins<=2 and loc>=.80
        cloud_ok=size_gb<=self.cfg.cloud_mem_gb*.8
        split_ok=(size_gb*q.selectivity<=self.cfg.edge_mem_gb*.8 and cloud_ok)
        cache_ok=split_ok and self.cfg.cache_gb>0
        repl_ok=split_ok and size_gb*.25<self.cfg.cache_gb
        return np.array([edge_ok,cloud_ok,split_ok,cache_ok,repl_ok],dtype=bool)
    def state(self,q):
        now=q.arrival_s
        edge_q=np.mean([max(0,n.queue_end-now) for n in self.edge])
        cloud_q=max(0,self.cloud.queue_end-now)
        v=np.array([np.log1p(q.bytes_in)/25,q.joins/5,q.aggregation,len(q.partitions)/15,
            np.log1p(q.sla_ms)/12,q.selectivity,self.locality(q),
            np.clip(edge_q/5,0,1),np.clip(cloud_q/5,0,1),
            self.cfg.bandwidth_mbps/100,self.cfg.rtt_ms/100,
            float(q.intent=='batch'),float(q.intent=='streaming'),float(q.intent=='event'),
            float(self.cache_key(q) in self.cache)],dtype=np.float32)
        return v
    def step(self,q,action:int):
        action=int(action)
        if not 0<=action<len(ACTIONS): raise ValueError('Unknown action')
        valid=self.mask(q)
        if not valid[action]:
            out=Outcome(self.cfg.max_wait_ms,0,0,False,False,ACTIONS[action],rejected=True)
            self.history.append((q,out));return out
        now=q.arrival_s
        idx=q.qid%4
        node=self.edge[idx]
        data_mb=q.bytes_in/1e6
        loc=self.locality(q)
        hit=action==3 and self.cache_key(q) in self.cache
        # deterministic operations, with stochastic fluctuations controlled by seed
        complexity=(1+.18*q.joins+.4*q.aggregation)
        edge_ms=data_mb*3.2/self.cfg.edge_cpu*complexity
        cloud_ms=data_mb*1.6/self.cfg.cloud_cpu*complexity
        selected=ACTIONS[action]
        if action==0:
            e_ms=edge_ms; c_ms=0; bytes_tx=0
        elif action==1:
            e_ms=0;c_ms=cloud_ms;bytes_tx=q.bytes_in
        else:
            e_ms=edge_ms*.6;c_ms=cloud_ms*(.35+.5*q.selectivity)
            bytes_tx=int(q.bytes_in*q.selectivity)
            if hit: e_ms*=.15;c_ms*=.3;bytes_tx=int(bytes_tx*.1)
            if action==4: bytes_tx+=int(q.bytes_in*.25)
        transfer_ms=(bytes_tx*8/self.cfg.bandwidth_mbps/1e6)*1000 if bytes_tx else 0
        if bytes_tx:transfer_ms+=self.cfg.rtt_ms
        # shared resource queues, measured against absolute arrival time
        edge_wait=max(0,node.queue_end-now)*1000 if e_ms else 0
        cloud_wait=max(0,self.cloud.queue_end-now)*1000 if c_ms else 0
        e_start=max(now,node.queue_end) if e_ms else now
        e_done=e_start+e_ms/1000
        c_start=max(self.cloud.queue_end,e_done+transfer_ms/1000) if c_ms else e_done
        c_done=c_start+c_ms/1000
        if e_ms: node.queue_end=e_done
        if c_ms:self.cloud.queue_end=c_done
        noise=float(self.rng.lognormal(0,.045))
        latency_ms=max(0,(c_done-now)*1000 if c_ms else (e_done-now)*1000)*noise+1.5
        cost=(e_ms*self.cfg.cost_edge_cpu_ms+c_ms*self.cfg.cost_cloud_cpu_ms+
              bytes_tx/1e6*self.cfg.cost_network_mb+
              (data_mb*.0002)*self.cfg.cost_storage_mb+
              (self.cfg.rtt_ms*self.cfg.cost_rtt if bytes_tx else 0))
        if action==3 and not hit:
            key=self.cache_key(q);capacity=int(self.cfg.cache_gb*1e9)
            nbytes=min(int(q.bytes_in*q.selectivity*.3),capacity)
            while self.cache and self.cache_bytes+nbytes>capacity:
                _,v=self.cache.popitem(last=False);self.cache_bytes-=v
            self.cache[key]=nbytes;self.cache_bytes+=nbytes
        if action==4:
            self.replicas.update(q.partitions)
        for p in q.partitions:self.partition_reads[p]+=1
        out=Outcome(float(latency_ms),float(cost),int(bytes_tx),latency_ms<=q.sla_ms,True,selected,
                    hit,False,float(max(c_done,e_done)),{'edge_ms':e_ms,'cloud_ms':c_ms,
                    'transfer_ms':transfer_ms,'edge_wait_ms':edge_wait,'cloud_wait_ms':cloud_wait})
        self.history.append((q,out));self.last_time=max(self.last_time,out.finished_at)
        return out

def reward(out,sla_ms):
    if not out.feasible:return -10.
    return float(-out.cost-2*max(0,(out.latency_ms-sla_ms)/max(sla_ms,1)))

def make_queries(n=400,seed=42,arrival_qps=50,dataset='mixed'):
    rng=np.random.default_rng(seed)
    names=['taxi','debs','gdelt'] if dataset=='mixed' else [dataset]
    queries=[]
    for i in range(n):
        name=names[i%len(names)]
        if name=='taxi':
            mb=float(rng.choice([100,250,500,1000]));joins=int(rng.integers(0,4));intent='batch'
            sla=1500.
        elif name=='debs':
            mb=float(rng.choice([1,2.5,5,7.5]));joins=0;intent='streaming';sla=250.
        else:
            mb=float(rng.choice([5,10,25,50]));joins=int(rng.integers(0,3));intent='event';sla=500.
        # recurring partitions enable cache and locality effects
        partbase=int(rng.integers(0,15))
        query=Query(i,name,int(mb*1e6),joins,float(rng.uniform(.1,.9)),
                    tuple(range(partbase,partbase+int(rng.integers(1,4)))),sla,
                    i/arrival_qps,float(rng.uniform(.05,.6)),intent)
        queries.append(query)
    return queries
