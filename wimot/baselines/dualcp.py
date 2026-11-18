"""DualCP (AAAI 2025), paper-based independent Mantis-output adaptation.

Primary: https://arxiv.org/html/2503.18042v1, Eqs. 5--13, Appendix A.
Use the authors' BASE-DOMAIN MEAN guidance alternative (Table 5), not CLIP.
Domain identity is inferred from feature/domain-mean cosine, never supplied.
Singleton hierarchy levels have no decision to learn; reduce to VanillaCP as
discussed in the paper. Compact 512-hidden-unit two-layer calibrators and normalized outputs are explicit
implementation choices; this is not an exact reproduction of official ViT runs.
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def components(adjacency):
    remaining=set(range(len(adjacency))); groups=[]
    while remaining:
        stack=[min(remaining)]; group=[]
        while stack:
            i=stack.pop()
            if i not in remaining: continue
            remaining.remove(i);group.append(i)
            stack.extend(int(j) for j in np.flatnonzero(adjacency[i]) if j in remaining)
        groups.append(sorted(group))
    return groups

def etf(guidance):
    """Rows are guidance vectors; return d x K unit ETF columns (K>1)."""
    guidance=np.asarray(guidance,dtype=np.float64); k=len(guidance)
    if k < 2: raise ValueError('Singleton level has no ETF decision')
    q,_=np.linalg.qr(guidance.T,mode='reduced')
    return np.sqrt(k/(k-1))*q@(np.eye(k)-np.ones((k,k))/k)

class Calibrator(nn.Module):
    def __init__(self,d,groups,hidden_dim=512):
        super().__init__();self.groups=groups;self.d=d
        self.coarse=nn.Sequential(nn.Linear(d,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,d)) if len(groups)>1 else None
        self.fine=nn.ModuleDict({str(i):nn.Sequential(nn.Linear(2*d,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,d))
                                 for i,g in enumerate(groups) if len(g)>1})
    def representations(self,x):
        coarse=F.normalize(self.coarse(x),dim=1) if self.coarse is not None else torch.zeros_like(x)
        fine={int(i):F.normalize(m(torch.cat((x,coarse),1)),dim=1) for i,m in self.fine.items()}
        return coarse,fine

class DualCP:
    def __init__(self,seed=42,device='cuda',threshold=.85,epochs=20,input_dim=9216,alpha=.5,classes=3,hidden_dim=512,first_epochs=100,lr=.1,batch_size=128):
        self.seed,self.device=seed,torch.device(device);self.threshold=float(threshold)
        self.classes=int(classes);self.hidden_dim=int(hidden_dim);self.first_epochs=int(first_epochs);self.lr=float(lr);self.batch_size=int(batch_size)
        self.epochs,self.d,self.alpha=epochs,input_dim,alpha
        self.models=[];self.domain_means=[];self.metadata=[];self.groups=None
    def _x(self,X):
        x=np.asarray(X,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=self.d or not np.isfinite(x).all(): raise ValueError('Invalid features')
        return x
    def initialize(self,x,y):
        means=np.array([x[y==c].mean(0) for c in range(self.classes)],dtype=np.float64)
        means/=np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-12)
        self.base_means=means.copy();self.groups=components(means@means.T>self.threshold)
        self.coarse_prototypes=(torch.tensor(etf(np.array([means[g].mean(0) for g in self.groups])),
                              device=self.device,dtype=torch.float32) if len(self.groups)>1 else None)
        self.fine_prototypes={i:torch.tensor(etf(means[g]),device=self.device,dtype=torch.float32)
                              for i,g in enumerate(self.groups) if len(g)>1}
    def update(self,X,y):
        x=self._x(X);y=np.asarray(y,dtype=int)
        if not len(x) or y.shape!=(len(x),) or not np.isin(y,np.arange(self.classes)).all(): raise ValueError('Invalid labels')
        if not np.array_equal(np.unique(y),np.arange(self.classes)): raise ValueError('Missing current classes')
        if self.groups is None: self.initialize(x,y)
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            # Equal initialization per independent domain calibrator.
            torch.manual_seed(self.seed)
            model=Calibrator(self.d,self.groups,self.hidden_dim).to(self.device)
            optimizer=torch.optim.SGD(model.parameters(),lr=self.lr,momentum=.9,weight_decay=2e-4)
            epochs=self.first_epochs if not self.models else self.epochs
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
            xt=torch.tensor(x,device=self.device);yt=torch.tensor(y,device=self.device)
            group_for=np.empty(self.classes,dtype=int);within=np.empty(self.classes,dtype=int)
            for i,g in enumerate(self.groups):
                for j,c in enumerate(g): group_for[c]=i;within[c]=j
            targets=torch.tensor(group_for[y],device=self.device);local=torch.tensor(within[y],device=self.device)
            rng=np.random.default_rng(self.seed+len(self.models)*100);history=[]
            for epoch in range(epochs):
                model.train();losses=[]
                order=rng.permutation(len(x))
                for start in range(0,len(x),self.batch_size):
                    ids=order[start:start+self.batch_size];coarse,fine=model.representations(xt[ids])
                    loss=xt.new_zeros(())
                    if model.coarse is not None:
                        target=self.coarse_prototypes[:,targets[ids]].T
                        loss=self.alpha*((coarse*target).sum(1)-1).square().mean()
                    for i,z in fine.items():
                        mask=targets[ids]==i
                        if mask.any():
                            proto=self.fine_prototypes[i][:,local[ids][mask]].T
                            loss=loss+(1-self.alpha)*((z[mask]*proto).sum(1)-1).square().sum()/len(ids)
                    if not torch.isfinite(loss): raise FloatingPointError('Invalid DDR loss')
                    optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
                scheduler.step();history.append(float(np.mean(losses)))
        model.eval().requires_grad_(False);self.models.append(model)
        mean=x.mean(0);self.domain_means.append(mean/np.maximum(np.linalg.norm(mean),1e-12))
        self.metadata.append({'groups':self.groups,'history':history,'train_rows':len(x),'epochs':epochs})
        return self
    def infer_domains(self,X):
        x=self._x(X)
        if not self.models: raise RuntimeError('Update first')
        return (x@np.stack(self.domain_means).T).argmax(1)
    def predict(self,X,task_id=None):
        if task_id is not None: raise ValueError('True task ID is prohibited')
        x=self._x(X);routes=self.infer_domains(x);result=np.empty(len(x),dtype=int)
        with torch.inference_mode():
            for t,model in enumerate(self.models):
                ids=np.flatnonzero(routes==t)
                for start in range(0,len(ids),128):
                    chosen=ids[start:start+128];xx=torch.tensor(x[chosen],device=self.device)
                    coarse,fine=model.representations(xx)
                    group=(coarse@self.coarse_prototypes).argmax(1).cpu().numpy() if len(self.groups)>1 else np.zeros(len(xx),dtype=int)
                    local={i:(z@self.fine_prototypes[i]).argmax(1).cpu().numpy() for i,z in fine.items()}
                    result[chosen]=[self.groups[g][local[g][j] if g in local else 0] for j,g in enumerate(group)]
        return result
    @property
    def memory_bytes(self):
        n=sum(p.numel()*p.element_size() for m in self.models for p in m.state_dict().values())
        n+=sum(v.nbytes for v in self.domain_means)+self.base_means.nbytes
        n+=sum(p.numel()*p.element_size() for p in self.fine_prototypes.values())
        if self.coarse_prototypes is not None:n+=self.coarse_prototypes.numel()*4
        return int(n)
    def diagnostics(self):
        return {'paper':'AAAI2025 DualCP','implementation':'paper-based CSI frozen-feature adaptation',
                'guidance':'first domain classifier-train class mean alternative, frozen',
                'groups':self.groups,'threshold':self.threshold,'alpha':self.alpha,'calibrators':len(self.models),
                'routing':'argmax feature-domain-mean cosine, no true task ID',
                'singleton_reduction':'VanillaCP if all groups singleton; skip singleton decisions',
                'epochs':self.epochs,'memory_bytes':self.memory_bytes,'replay_rows':0,'metadata':self.metadata}


class NativeRoutingDualCP(DualCP):
    """DualCP (threshold < 1) or VanillaCP (threshold 1) with cosine domain routing on the uncentred,
    variance-scaled input (``routing_offset`` = first-domain mean / sd); only the calibrator input is centred."""
    routing_offset = 0.

    def update(self, X, y):
        super().update(X, y)
        mean = np.asarray(X, np.float32).mean(0).astype(float)+self.routing_offset
        self.domain_means[-1] = mean/max(np.linalg.norm(mean), 1e-12)
        return self

    def infer_domains(self, X):
        x = self._x(X).astype(float)+self.routing_offset
        return (x@np.stack(self.domain_means).T).argmax(1)

    def at_stage(self, stage):
        """Calibrators are frozen once trained, so the stage-k predictor is the final model restricted to k calibrators."""
        m = copy.copy(self)
        m.models, m.domain_means, m.metadata = self.models[:stage], self.domain_means[:stage], self.metadata[:stage]
        return m
