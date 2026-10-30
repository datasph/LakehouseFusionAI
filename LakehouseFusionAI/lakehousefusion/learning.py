"""MLP outcome regressors and feasibility-aware DQN."""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn
from .core import reward

class MLP(nn.Module):
    def __init__(self,n_in,n_out=1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,n_out))
    def forward(self,x):return self.net(x)

class Predictors:
    def __init__(self,state_dim=15,lr=.001):
        self.models=[MLP(state_dim+5) for _ in range(2)]
        self.opts=[torch.optim.Adam(m.parameters(),lr=lr) for m in self.models]
        self.y_scale=np.ones(2)
    @staticmethod
    def features(state,action):return np.r_[state,np.eye(5,dtype=np.float32)[action]].astype(np.float32)
    def fit(self,records,epochs=60,patience=10,batch=64):
        if len(records)<20:raise ValueError('At least 20 training records required')
        x=np.array([self.features(s,a) for s,a,_,_ in records],dtype=np.float32)
        y=np.array([[np.log1p(max(0,l)),np.log1p(max(0,c))] for _,_,l,c in records],dtype=np.float32)
        cut=max(1,int(.85*len(x)))
        xtr=torch.tensor(x[:cut]);xval=torch.tensor(x[cut:]);ytr=torch.tensor(y[:cut]);yval=torch.tensor(y[cut:])
        outputs=[]
        for j,(model,opt) in enumerate(zip(self.models,self.opts)):
            best=float('inf');wait=0;bestweights=None
            for epoch in range(epochs):
                model.train()
                for indices in torch.randperm(len(xtr)).split(batch):
                    loss=((model(xtr[indices]).flatten()-ytr[indices,j])**2).mean()
                    opt.zero_grad();loss.backward();opt.step()
                model.eval()
                with torch.no_grad():v=float(((model(xval).flatten()-yval[:,j])**2).mean())
                if v<best-1e-6:best=v;bestweights=copy.deepcopy(model.state_dict());wait=0
                else:
                    wait+=1
                    if wait>=patience:break
            model.load_state_dict(bestweights);outputs.append({'val_mse_log':best,'epochs':epoch+1})
        return outputs
    def predict(self,state,action):
        x=torch.tensor(self.features(state,action)[None,:])
        with torch.no_grad():return tuple(max(0,float(torch.expm1(m(x).flatten()[0]))) for m in self.models)
    def save(self,path):torch.save({'models':[m.state_dict() for m in self.models]},path)
    def load(self,path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        for m,v in zip(self.models,d['models']):m.load_state_dict(v)

class DQN:
    def __init__(self,state_dim=15,seed=42,lr=1e-4,gamma=.95,capacity=50000,batch=64):
        torch.manual_seed(seed);self.rng=np.random.default_rng(seed)
        self.online=MLP(state_dim,5);self.target=copy.deepcopy(self.online)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr)
        self.memory=deque(maxlen=capacity);self.batch=batch;self.gamma=gamma;self.steps=0
    def choose(self,state,mask,epsilon=.0):
        ids=np.flatnonzero(mask)
        if len(ids)==0:return None
        if self.rng.random()<epsilon:return int(self.rng.choice(ids))
        with torch.no_grad():v=self.online(torch.tensor(np.asarray(state,dtype=np.float32)[None,:]))[0].numpy()
        v[~mask]=-np.inf
        return int(v.argmax())
    def remember(self,s,a,r,ns,done,next_mask):
        self.memory.append((np.asarray(s,dtype=np.float32),int(a),float(r),np.asarray(ns,dtype=np.float32),bool(done),np.asarray(next_mask,dtype=bool)))
    def optimize(self):
        if len(self.memory)<self.batch:return None
        chosen=self.rng.choice(len(self.memory),self.batch,replace=False)
        batch=[self.memory[int(i)] for i in chosen]
        s=torch.tensor(np.stack([b[0] for b in batch]));a=torch.tensor([b[1] for b in batch],dtype=torch.long)
        r=torch.tensor([b[2] for b in batch],dtype=torch.float32)
        ns=torch.tensor(np.stack([b[3] for b in batch]));done=torch.tensor([b[4] for b in batch],dtype=torch.float32)
        masks=torch.tensor(np.stack([b[5] for b in batch]))
        q=self.online(s).gather(1,a[:,None]).flatten()
        with torch.no_grad():
            next_q=self.target(ns).masked_fill(~masks,-1e9).max(1).values
            next_q=torch.where(masks.any(1),next_q,torch.zeros_like(next_q))
            target=r+self.gamma*(1-done)*next_q
        loss=nn.functional.smooth_l1_loss(q,target)
        self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.online.parameters(),1.);self.optimizer.step()
        self.steps+=1
        if self.steps%200==0:self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())
    def train_environment(self,env,queries,epochs=3):
        losses=[]
        for ep in range(epochs):
            env.reset()
            eps=max(.05,1-ep/max(epochs-1,1)*.95)
            for i,q in enumerate(queries):
                state=env.state(q);mask=env.mask(q);action=self.choose(state,mask,eps)
                if action is None:continue
                out=env.step(q,action)
                if i+1<len(queries):ns=env.state(queries[i+1]);nm=env.mask(queries[i+1]);done=False
                else:ns=np.zeros_like(state);nm=np.zeros(5,dtype=bool);done=True
                self.remember(state,action,reward(out,q.sla_ms),ns,done,nm)
                loss=self.optimize()
                if loss is not None:losses.append(loss)
        return {'transitions':len(self.memory),'updates':self.steps,'last_loss':losses[-1] if losses else None}
    def save(self,path):torch.save({'online':self.online.state_dict(),'target':self.target.state_dict()},path)
    def load(self,path):
        d=torch.load(path,map_location='cpu',weights_only=True);self.online.load_state_dict(d['online']);self.target.load_state_dict(d['target'])
