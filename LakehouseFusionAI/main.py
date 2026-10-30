import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from lakehousefusion import HybridEnvironment,EnvironmentConfig,make_queries,DQN,Predictors,evaluate,summary,statistical_tests
from lakehousefusion.storage import validate_hybrid_execution
from lakehousefusion.datasets import normalize

def main():
    p=argparse.ArgumentParser(description='LakehouseFusionAI reproducible research toolkit')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('sql-demo').add_argument('--out',default='results/sql_demo')
    d=sub.add_parser('prepare');d.add_argument('--input',required=True);d.add_argument('--dataset',choices=['taxi','debs','gdelt'],required=True);d.add_argument('--out',required=True)
    t=sub.add_parser('train');t.add_argument('--out',default='checkpoints');t.add_argument('--n',type=int,default=350);t.add_argument('--epochs',type=int,default=3);t.add_argument('--seed',type=int,default=42)
    e=sub.add_parser('evaluate');e.add_argument('--out',default='results');e.add_argument('--n',type=int,default=200);e.add_argument('--runs',type=int,default=10);e.add_argument('--qps',type=int,default=50);e.add_argument('--network',choices=['normal','constrained'],default='normal');e.add_argument('--dataset',choices=['mixed','taxi','debs','gdelt'],default='mixed');e.add_argument('--models',default='checkpoints');e.add_argument('--ablation',action='store_true')
    vis=sub.add_parser('plot');vis.add_argument('--metrics',default='results/run_metrics.csv');vis.add_argument('--out',default='results/figures')
    args=p.parse_args();torch.set_num_threads(1)
    if args.cmd=='plot':
        from lakehousefusion.plots import make_plots
        make_plots(args.metrics,args.out);print('Saved 600-dpi figures to',args.out);return
    if args.cmd=='sql-demo':
        print(json.dumps(validate_hybrid_execution(args.out),indent=2));return
    if args.cmd=='prepare':
        print(json.dumps(normalize(args.input,args.dataset,args.out),indent=2));return
    if args.cmd=='train':
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        env=HybridEnvironment(EnvironmentConfig(seed=args.seed))
        qs=make_queries(args.n,seed=args.seed,arrival_qps=30)
        # Explore training period, record chronological observed outcomes only
        records=[];rng=np.random.default_rng(args.seed)
        for q in qs:
            mask=env.mask(q);ids=np.flatnonzero(mask)
            if not len(ids):continue
            a=int(rng.choice(ids));s=env.state(q);o=env.step(q,a)
            records.append((s,a,o.latency_ms,o.cost))
        pred=Predictors();info=pred.fit(records);pred.save(out/'predictors.pt')
        agent=DQN(seed=args.seed)
        stats=agent.train_environment(HybridEnvironment(EnvironmentConfig(seed=args.seed+1)),qs,epochs=args.epochs)
        agent.save(out/'dqn.pt')
        (out/'train_metrics.json').write_text(json.dumps({'predictors':info,'dqn':stats,'training_records':len(records)},indent=2))
        print(json.dumps({'predictors':info,'dqn':stats,'training_records':len(records)},indent=2));return
    if args.cmd=='evaluate':
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        agent=None;pred=None
        if (Path(args.models)/'dqn.pt').exists():
            agent=DQN();agent.load(Path(args.models)/'dqn.pt')
        if (Path(args.models)/'predictors.pt').exists():
            pred=Predictors();pred.load(Path(args.models)/'predictors.pt')
        methods=['cloud_only','edge_only','static_hybrid','proposed']
        if args.ablation:methods+=['no_cache','no_replication','no_edge_preprocess']
        seeds=range(10001,10001+args.runs)
        raw,traces=evaluate(methods,seeds,args.n,args.qps,args.network,agent,pred,args.dataset)
        raw.to_csv(out/'run_metrics.csv',index=False);traces.to_csv(out/'query_traces.csv',index=False)
        summary(raw).to_csv(out/'summary.csv',index=False)
        statistical_tests(raw).to_csv(out/'statistics.csv',index=False)
        (out/'metadata.json').write_text(json.dumps({'evaluation_type':'stateful logical emulation (not physical distributed measurements)','trained_policy':agent is not None,'runs':args.runs,'qps':args.qps,'network':args.network,'dataset':args.dataset},indent=2))
        print(summary(raw).to_string(index=False));print('Saved reports to',out)
if __name__=='__main__':main()
