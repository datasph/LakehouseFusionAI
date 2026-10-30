"""Matched comparisons and reports. Results are simulation estimates."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from .core import HybridEnvironment,EnvironmentConfig,make_queries,ACTIONS

def plan(method,env,q,agent=None,predictors=None):
    mask=env.mask(q)
    if not mask.any():return None
    if method=='proposed' and agent is not None:return agent.choose(env.state(q),mask,0.)
    if method=='no_dqn' and predictors is not None:
        feasible=np.flatnonzero(mask)
        return int(min(feasible,key=lambda a:predictors.predict(env.state(q),int(a))[1]+2*max(0,predictors.predict(env.state(q),int(a))[0]/q.sla_ms-1)))
    preferences={'cloud_only':[1],'edge_only':[0],'static_hybrid':[2,1,0],
                 'proposed':[3,4,2,0,1],'no_cache':[4,2,0,1],
                 'no_replication':[3,2,0,1],'no_edge_preprocess':[1,0]}
    return next((a for a in preferences.get(method,[1]) if mask[a]),None)

def one_run(method,seed,n,qps,network='normal',agent=None,predictors=None,dataset='mixed'):
    cfg=EnvironmentConfig(seed=seed,bandwidth_mbps=100 if network=='normal' else 20,
                          rtt_ms=20 if network=='normal' else 100)
    env=HybridEnvironment(cfg)
    qs=make_queries(n,seed,arrival_qps=qps,dataset=dataset)
    rows=[]
    for q in qs:
        a=plan(method,env,q,agent,predictors)
        if a is None:
            rows.append({'qid':q.qid,'status':'rejected','latency_ms':np.nan,'cost':0.,'transfer_bytes':0,'sla_met':False,'arrival_s':q.arrival_s,'finish_s':np.nan})
            continue
        out=env.step(q,a)
        rows.append({'qid':q.qid,'status':'completed' if out.feasible else 'rejected','action':out.action,
                     'latency_ms':out.latency_ms,'cost':out.cost,'transfer_bytes':out.transfer_bytes,
                     'sla_met':out.sla_met,'arrival_s':q.arrival_s,'finish_s':out.finished_at})
    df=pd.DataFrame(rows)
    good=df[df.status=='completed']
    # Entirely infeasible methods remain explicit failures rather than disappearing.
    elapsed=max(float(good.finish_s.max()) if not good.empty else 0.,qs[-1].arrival_s+1/qps)
    metrics={'method':method,'seed':seed,'network':network,'qps_target':qps,'dataset':dataset,
             'mean_latency_ms':float(good.latency_ms.mean()) if not good.empty else float('nan'),'p95_latency_ms':float(good.latency_ms.quantile(.95)) if not good.empty else float('nan'),
             'normalized_cost':float(df.cost.sum()),'sla_pct':float(df.sla_met.mean()*100),
             'throughput_qps':float(len(good)/elapsed),'data_transfer_mb':float(df.transfer_bytes.sum()/1e6),
             'completed':int(len(good)),'rejected':int(len(df)-len(good)),'elapsed_s':elapsed}
    return metrics,df

def evaluate(methods=('cloud_only','edge_only','static_hybrid','proposed'),seeds=range(42,52),
             n=200,qps=50,network='normal',agent=None,predictors=None,dataset='mixed'):
    output=[];traces=[]
    for seed in seeds:
        for method in methods:
            metric,raw=one_run(method,seed,n,qps,network,agent,predictors,dataset)
            output.append(metric)
            raw.insert(0,'seed',seed);raw.insert(0,'method',method);traces.append(raw)
    return pd.DataFrame(output),pd.concat(traces,ignore_index=True)

def summary(df):
    cols=['mean_latency_ms','p95_latency_ms','normalized_cost','sla_pct','throughput_qps','data_transfer_mb']
    return df.groupby(['method','network','qps_target'])[cols].agg(['mean','std']).reset_index()

def statistical_tests(df,reference='proposed'):
    cols=['mean_latency_ms','p95_latency_ms','normalized_cost','sla_pct','throughput_qps','data_transfer_mb']
    rows=[]
    for (net,qps),subset in df.groupby(['network','qps_target']):
        ref=subset[subset.method==reference].set_index('seed')
        for metric in cols:
            family=[]
            for name in sorted(set(subset.method)-{reference}):
                alt=subset[subset.method==name].set_index('seed')
                common=ref.index.intersection(alt.index)
                if len(common)<2:continue
                dif=(ref.loc[common,metric]-alt.loc[common,metric]).to_numpy()
                try:p=float(wilcoxon(dif).pvalue) if np.any(dif!=0) else 1.
                except ValueError:p=1.
                family.append({'network':net,'qps':qps,'metric':metric,'comparator':name,'n':len(common),'mean_paired_difference':float(dif.mean()),'p_raw':p})
            if family:
                corrected=multipletests([r['p_raw'] for r in family],method='holm')[1]
                for row,p in zip(family,corrected):row['p_holm']=float(p)
                rows.extend(family)
    return pd.DataFrame(rows)
