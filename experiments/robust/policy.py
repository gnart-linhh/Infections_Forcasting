"""Synthetic-trained rank weighting of summaries of one unchanged sMOA match."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier

FEATURES = ['relative_growth','coefficient_variation','last_relative_maximum',
            'analogue_dispersion','horizon','history_length']


def candidate_features(engine, history):
    match=engine.match(history)
    z=np.maximum(np.asarray(history,float),1.)
    anchor=z[-1];scale=max(float(z.mean()),1.)
    cumulative=anchor+np.cumsum(match.future,axis=1)
    # Quantile of reconstructed cumulative trajectories, NOT cumulative quantile
    # of individual increments. Candidate identity 1 is exact original sMOA.
    low,high=np.quantile(cumulative,[.25,.75],axis=0)
    candidates=np.column_stack([np.maximum(0,low),match.baseline,np.maximum(0,high)])
    ordering=np.argsort(candidates,axis=1,kind='stable')
    ranked=np.take_along_axis(candidates,ordering,axis=1)
    growth=(z[-2:].mean()-z[-4:-2].mean())/max(float(z[-4:-2].mean()),1.)
    common=[growth,float(z.std()/scale),float(z[-1]/max(float(z.max()),1.))]
    x=np.array([common+[float(match.dispersion[h]),h+1,len(z)] for h in range(len(match.baseline))])
    if not np.isfinite(x).all() or not np.isfinite(ranked).all():raise ValueError('Nonfinite candidates or features')
    return x,candidates,ranked,ordering


def rank_labels(ranked, ordering, actual):
    loss=np.abs(np.asarray(ranked)-np.asarray(actual)[:,None])
    labels=np.argmin(loss,axis=1)
    original_rank=np.argmax(np.asarray(ordering)==1,axis=1)
    rows=np.arange(len(loss))
    # Prefer exact original on absolute-error ties, otherwise stable lowest rank.
    prefer=loss[rows,original_rank]==loss.min(axis=1)
    labels[prefer]=original_rank[prefer]
    return labels


def convex_forecast(original,ranked,probabilities,rho):
    p=np.asarray(probabilities,float)
    ranked=np.asarray(ranked,float);original=np.asarray(original,float)
    if not 0<=rho<=1 or p.shape!=ranked.shape or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Need valid rank probabilities and rho in [0,1]')
    if rho==0:return original.copy()
    p=p/p.sum(axis=1,keepdims=True)
    blended=(1-rho)*original+rho*np.sum(p*ranked,axis=1)
    return np.where(np.all(ranked==ranked[:,:1],axis=1),ranked[:,0],blended)


class RankPolicy:
    def __init__(self,depth=3,seed=42):
        self.depth,self.seed=depth,seed
        self.model=RandomForestClassifier(n_estimators=100,max_depth=depth,min_samples_leaf=8,
                    max_features=None,random_state=seed,n_jobs=1)

    def fit(self, frame):
        if set(frame.dataset.unique()) != {'synthetic_training'}:
            raise ValueError('Policy fitting accepts synthetic training only')
        ranked=frame[['rank_0','rank_1','rank_2']].to_numpy()
        order=frame[['identity_0','identity_1','identity_2']].to_numpy(int)
        target=rank_labels(ranked,order,frame.actual.to_numpy())
        # Equal total weight per family and prespecified count-scale bin; no
        # raw-error weights, so enormous epidemics cannot dominate training.
        strata=frame.family.astype(str)+'|'+np.digitize(frame.scale,[1000,100000]).astype(str)
        weights=1/strata.map(strata.value_counts()).to_numpy(float)
        self.model.fit(frame[FEATURES].to_numpy(float),target,sample_weight=weights/weights.mean())
        return self

    def probabilities(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=len(FEATURES) or not np.isfinite(x).all():raise ValueError('Invalid policy features')
        p=np.zeros((len(x),3))
        p[:,self.model.classes_.astype(int)]=self.model.predict_proba(x)
        return p/p.sum(axis=1,keepdims=True)

    def forecast(self,engine,history,rho=1.):
        x,candidates,ranked,ordering=candidate_features(engine,history)
        weights=self.probabilities(x)
        return convex_forecast(candidates[:,1],ranked,weights,rho),dict(features=x,candidates=candidates,
                  ranked=ranked,ordering=ordering,probabilities=weights,rho=rho)
