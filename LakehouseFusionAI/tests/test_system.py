import numpy as np
import pytest
from lakehousefusion import HybridEnvironment,EnvironmentConfig,make_queries,DQN,Predictors
from lakehousefusion.storage import SQLLakehouse,EdgeAgent,demo_data,validate_hybrid_execution
from lakehousefusion.evaluation import evaluate,statistical_tests

def test_actions_and_state():
    e=HybridEnvironment(EnvironmentConfig());q=make_queries(12)[0]
    assert e.state(q).shape==(15,)
    assert e.mask(q).shape==(5,)
    action=int(np.flatnonzero(e.mask(q))[0]);out=e.step(q,action)
    assert out.feasible and out.latency_ms>0 and out.cost>=0

def test_dqn_mask():
    a=DQN();s=np.zeros(15,dtype=np.float32)
    assert a.choose(s,np.array([False,True,False,False,False]),epsilon=1)==1
    assert a.choose(s,np.zeros(5,dtype=bool)) is None

def test_sql_transaction_and_sync(tmp_path):
    result=validate_hybrid_execution(tmp_path)
    assert result['local_rows']==result['cloud_rows']==100
    assert result['snapshot_version']==4

def test_idempotence(tmp_path):
    db=SQLLakehouse(tmp_path/'a.sqlite');rows=demo_data(4)
    assert db.ingest(rows,'batch1')==4
    assert db.ingest(rows,'batch1')==0
    with pytest.raises(ValueError):db.ingest(rows[:2],'batch1')

def test_eval():
    m,tr=evaluate(seeds=[5,6],n=18,qps=30)
    assert len(m)==8 and len(tr)==18*8
    assert 'p_holm' in statistical_tests(m).columns
