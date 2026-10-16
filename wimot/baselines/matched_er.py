"""Explicit ER extensions; no proposed representation or checkpoints.

CN-DPM replaces cross-domain novelty retention by the canonical recording
reservoir. Temporary novelty STM contains CURRENT rows only and is discarded
at each boundary. Historical rows never enter resampled sleep; each is visited
once per wake epoch. This is a history-aligned protocol adaptation, not native
one-pass CN-DPM. Its sole persistent individual cache is owned by the runner.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..common import digest
from .cn_dpm import CNDPM
from .dualcp import Calibrator, NativeRoutingDualCP
from .duct import WarmStartedDUCT


def epoch_orders(seed, stage, count, current_rows, epochs):
    rng=np.random.default_rng(seed+stage*100)
    for epoch in range(epochs):
        order=rng.permutation(count)
        yield epoch,order,dict(epoch=epoch,current_visits=int((order<current_rows).sum()),
                               old_visits=int((order>=current_rows).sum()),order_sha256=digest(order))


class ReplayDUCT(WarmStartedDUCT):
    def update_aligned(self,x,y,old):
        self.er_old=old;self.er_epochs=[]
        try:super().update(x,y)
        finally:del self.er_old
        return self

    def _fit(self,x,y,retrain=False):
        old=self.er_old
        if retrain or not len(old[1]):return super()._fit(x,y,retrain)
        cfg=self.config;n=len(y)
        xx=torch.cat((x,torch.from_numpy(old[0]).to(self.device)))
        # Current rows use the native current-slot CE. ER rows address their
        # acquired domain/class slot using training provenance, never test IDs.
        yy=torch.cat((y,torch.from_numpy(old[2]*self.classes+old[1]).to(self.device)))
        optimizer=torch.optim.SGD([{'params':self.backbone.parameters(),'lr':cfg.learning_rate*cfg.backbone_lr_scale},
                                   {'params':self.head.parameters(),'lr':cfg.learning_rate}],
                                  momentum=cfg.momentum,weight_decay=cfg.weight_decay)
        epochs=cfg.epochs
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(epochs*.6),int(epochs*.8)],gamma=.1)
        losses=[]
        for epoch,order,visits in epoch_orders(self.seed,self.num_tasks,len(yy),n,epochs):
            total=0.
            for start in range(0,len(order),cfg.batch_size):
                ids=torch.as_tensor(order[start:start+cfg.batch_size],device=self.device)
                logits=self.head(self.backbone(xx[ids]));current=ids<n;value=logits.sum()*0
                if current.any():value=value+F.cross_entropy(logits[current,self.classes*self.num_tasks:],yy[ids][current],reduction='sum')
                if (~current).any():value=value+F.cross_entropy(logits[~current],yy[ids][~current],reduction='sum')
                loss=value/len(ids)
                if not torch.isfinite(loss):raise FloatingPointError('DUCT ER loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
            scheduler.step();losses.append(total/len(yy));visits['loss']=losses[-1];self.er_epochs.append(visits)
        return losses


class ReplayDualCP(NativeRoutingDualCP):
    def update_aligned(self,X,y,old):
        x=self._x(X);y=np.asarray(y,dtype=np.int64);n=len(y)
        if self.groups is None:self.initialize(x,y)
        pool=np.concatenate((x,old[0]));labels=np.concatenate((y,old[1]))
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed);model=Calibrator(self.d,self.groups,self.hidden_dim).to(self.device)
        optimizer=torch.optim.SGD(model.parameters(),lr=self.lr,momentum=.9,weight_decay=2e-4)
        epochs=self.first_epochs if not self.models else self.epochs
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        xt=torch.from_numpy(pool).to(self.device)
        group_for=np.empty(self.classes,dtype=np.int64);within=np.empty(self.classes,dtype=np.int64)
        for i,g in enumerate(self.groups):
            for j,c in enumerate(g):group_for[c]=i;within[c]=j
        targets=torch.from_numpy(group_for[labels]).to(self.device);local=torch.from_numpy(within[labels]).to(self.device)
        self.er_epochs=[];history=[]
        for epoch,order,visits in epoch_orders(self.seed,len(self.models),len(labels),n,epochs):
            total=0.;model.train()
            for start in range(0,len(order),self.batch_size):
                ids=torch.as_tensor(order[start:start+self.batch_size],device=self.device)
                coarse,fine=model.representations(xt[ids]);loss=xt.new_zeros(())
                if model.coarse is not None:
                    proto=self.coarse_prototypes[:,targets[ids]].T
                    loss=self.alpha*((coarse*proto).sum(1)-1).square().mean()
                for i,z in fine.items():
                    mask=targets[ids]==i
                    if mask.any():
                        proto=self.fine_prototypes[i][:,local[ids][mask]].T
                        loss=loss+(1-self.alpha)*((z[mask]*proto).sum(1)-1).square().sum()/len(ids)
                if not torch.isfinite(loss):raise FloatingPointError('CP ER loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
            scheduler.step();history.append(total/len(labels));visits['loss']=history[-1];self.er_epochs.append(visits)
        model.eval().requires_grad_(False);self.models.append(model)
        # Preserve current-domain-only routing anchors; ER must not shift them.
        mean=x.mean(0,dtype=np.float64)+self.routing_offset
        self.domain_means.append(mean/max(np.linalg.norm(mean),1e-12))
        self.metadata.append(dict(groups=self.groups,history=history,train_rows=n,old_rows=len(old[1]),epochs=epochs))
        return self


class HistoryAlignedCNDPM(CNDPM):
    def _wake_aligned(self,x,y,is_old):
        density,pred=self._collect(x,training=True);nll=density+self._class_nll(pred,y)
        logprior=torch.cat((torch.tensor([self.config.log_alpha],device=self.device),self.log_counts))
        logjoint=logprior[None,:]-nll
        # Historical replay updates existing experts, instead of being silently
        # skipped as placeholder novelty. It cannot enter native sleep STM.
        if self.experts:logjoint=logjoint.clone();logjoint[is_old,0]=-torch.inf
        with torch.no_grad():
            novel=logjoint.argmax(1)==0
            if self.experts:
                raw=torch.exp(logjoint-logjoint.max(1,keepdim=True).values);raw[:,0]=0
                responsibility=raw/(raw.sum(1,keepdim=True)+1e-7);usage=responsibility[:,1:].sum(0)
                self.log_counts=torch.logaddexp(self.log_counts,usage.log())
            else:responsibility=torch.zeros_like(logjoint);usage=torch.empty(0,device=self.device)
        if self.experts:
            losses=(nll[:,1:]*responsibility[:,1:]*(~novel)[:,None]).sum(0)
            for go,do in self.optimizers:go.zero_grad(set_to_none=True);do.zero_grad(set_to_none=True)
            if (~novel).any():
                losses.sum().backward()
                for expert,(go,do),used in zip(self.experts,self.optimizers,usage):
                    if used.item()>self.config.update_min_usage:
                        nn.utils.clip_grad_value_(expert.parameters(),self.config.gradient_clip_value);go.step();do.step()
        assert not (novel&is_old).any()
        for row,label in zip(x[novel].detach().cpu(),y[novel].detach().cpu()):
            self.stm_x.append(row.clone());self.stm_y.append(int(label));self.stm_tasks.append(self.update_count)
            self.stm_peak=max(self.stm_peak,self.stm_size)
            if self.stm_size==self.config.stm_capacity:self._sleep()

    def update_aligned(self,X,y,old):
        started=time.monotonic();x=self._validate_x(X,nonempty=True);labels=np.asarray(y,dtype=np.int64);n=len(labels)
        assert not self.stm_x,'No residual novelty records cross domain boundaries'
        if self.normalizer_mean is None:
            self.normalizer_mean=x.mean(0,dtype=np.float64).astype(np.float32)
            std=x.std(0,ddof=0,dtype=np.float64);self.normalizer_std=np.where(std>1e-6,std,1).astype(np.float32)
        pool=np.concatenate((x,old[0]));targets=np.concatenate((labels,old[1]))
        xt=torch.from_numpy(self._normalize(pool)).to(self.device);yt=torch.from_numpy(targets).to(self.device)
        # First domain preserves native one-pass wake/sleep. Later domains use
        # fifty wake passes to match the proposed fifty historical visits.
        epochs=1 if self.update_count==0 else 50;self.er_epochs=[]
        for epoch,order,visits in epoch_orders(self.seed,self.update_count,len(targets),n,epochs):
            for start in range(0,len(order),self.config.batch_size):
                ids=torch.as_tensor(order[start:start+self.config.batch_size],device=self.device)
                self._wake_aligned(xt[ids],yt[ids],ids>=n)
            self.er_epochs.append(visits)
            if epoch%10==0:print('CN_ALIGNED_EPOCH',self.update_count+1,epoch,self.expert_count,flush=True)
        # Discard only CURRENT novelty workspace. Sole persistent individual
        # cache is the canonical reservoir, populated by the runner, not a
        # second cache added alongside native cross-domain novelty memory.
        residual=self.stm_size;self.stm_x.clear();self.stm_y.clear();self.stm_tasks.clear()
        self.seen_samples+=n;self.update_count+=1;self.update_duration=time.monotonic()-started
        assert all(s['old_stm_rows']==0 and s['old_sample_visits_actual']==0 for s in self.sleep_access)
        self.stage_history.append(dict(update=self.update_count,experts=self.expert_count,sleep_count=self.sleep_count,
                                       discarded_current_novelty=residual,old_rows=len(old[1]),wake_epochs=epochs,
                                       memory_bytes=self.memory_bytes,update_duration=self.update_duration))
        return self

    def diagnostics(self):
        v=super().diagnostics();v.update(history_replay=True,extra_historical_cache=False,
              history_protocol='Canonical class-priority reservoir replaces cross-domain novelty retention; 50 bounded wake passes; sleep accesses current rows only',
              native_one_pass=False,novelty_cross_domain_rows=0)
        return v
