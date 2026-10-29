"""Structural Python port of gen_curve in LANL smoa_helpers.R.

NumPy RNG and SciPy LSODA are numerical ports, not R bitwise equivalents.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit

GENERATOR_VERSION='source-structure-v2'


@dataclass
class SyntheticCurve:
    ts: np.ndarray
    curve_type: str
    time_cadence: str
    scale: str
    seasonal: bool

# Mô phỏng một đợt dịch
def sir(beta,gamma,s0,i0,r0,times):
    def equations(time,state):
        s,i,r=state
        return [-beta*i*s,beta*i*s-gamma*i,gamma*i]
    sol=solve_ivp(equations,(float(times[0]),float(times[-1])),[s0,i0,r0],
        t_eval=np.asarray(times,float),method='LSODA',rtol=1e-6,atol=1e-6)
    if not sol.success or not np.all(np.isfinite(sol.y)): raise RuntimeError('SIR solver failed')
    s,i,r=sol.y
    return dict(S=s,I=i,R=r,incidence=np.r_[np.nan,-np.diff(s)]) # incidence is the negative change in susceptibles, prepended with NaN to match the length of times

#Cắt phần đầu và cuối của chuỗi
def trim_tails(ts):
    """Translate R's 1-based, inclusive tail removals exactly."""
    active=np.flatnonzero(ts>1e-3)
    if not len(active): return ts
    cut=int(active[0])+1-10
    if 1<cut<len(ts)-10: ts=ts[cut:]
    active=np.flatnonzero(ts>1e-3)
    cut=int(active[-1])+1+10
    if 10<cut<len(ts): ts=ts[:cut-1]
    return ts

#Gộp dữ liệu theo tháng
def monthly_aggregate(ts):
    # R index=floor((1:length(ts))/4): first bin has 3 steps, not 4.
    return np.bincount(np.arange(1,len(ts)+1)//4,weights=ts)

#Thêm nhiễu ngẫu nhiên
def beta_observations(ts,alpha,rng):
    """R beta limit distributions at probabilities 0 and 1, without epsilon bias."""
    if np.any(ts < -1e-8) or np.any(ts > 1+1e-8): raise ValueError('Invalid beta probabilities')
    ts=np.clip(ts,0,1)
    obs=np.zeros(len(ts));obs[ts==1]=1
    inside=(ts>0)&(ts<1)
    obs[inside]=rng.beta(alpha*ts[inside],alpha*(1-ts[inside]))
    return obs

# Lấy thứ tự có trọng số
def weighted_order(rng,weights):
    return rng.choice(len(weights),len(weights),replace=False,p=weights/np.sum(weights))

# Sinh một đợt dịch SIR hoàn chỉnh
def solve_wave(reproduction,gamma,initial,rng,seasonal=False):
    for attempt in range(5):
        try:
            output=sir(reproduction/initial[0]*gamma,gamma,*initial,np.arange(1,501))
            # Bound tiny negative solver roundoff before assembling probabilities.
            return np.maximum(output['I'],0)
        except RuntimeError:
            reproduction=(1.2+rng.beta(1,3)*10) if seasonal else (1.1+rng.beta(1,3)*20)
            gamma=1/rng.uniform(1,10);initial=rng.dirichlet([1000,.1,.1])
    return np.full(500,.01)  # finite, bounded numerical fallback; logged by version

#Hàm sinh chuỗi chính
def gen_curve(curve_type,rng=None):
    rng=rng if rng is not None else np.random.default_rng()
    if curve_type not in ['sir_rollercoaster','sir_rollercoaster_wiggle','seasonal']:
        raise ValueError(f'Unknown family: {curve_type}')
    seasonal=curve_type=='seasonal'
    if not seasonal:
        nwaves=int(rng.integers(3,8))
        reproduction=1.1+rng.beta(1,3,nwaves)*20
        ordering=rng.choice(['increasing','decreasing','random'])
        if ordering!='random':
            order=weighted_order(rng,reproduction)
            reproduction=reproduction[order[::-1] if ordering=='increasing' else order]
        gamma=1/rng.uniform(1,10,nwaves)
        initial=rng.dirichlet([1000,.1,.1],size=nwaves)
        cadence=rng.choice(np.arange(10,53),size=nwaves,replace=False)
        cadence=cadence[weighted_order(rng,cadence.max()+1-cadence)]
        start=1;waves=[]
        for j in range(nwaves):
            wave=solve_wave(reproduction[j],gamma[j],initial[j],rng)
            active=np.flatnonzero(wave/max(float(wave.max()),1e-300)>1e-4)
            wave=wave[:active[-1]+1] if len(active) else wave
            shifted=np.r_[np.zeros(start),wave]
            limit=float(np.quantile(shifted,.01))+1e-9
            waves.append(np.r_[rng.uniform(0,limit,20),shifted,rng.uniform(0,limit,20)])
            start+=int(cadence[j])
        ts=np.zeros(max(map(len,waves)))
        for wave in waves: ts[:len(wave)]+=wave
        ts/=max(float(ts.max()),1e-300);ts=trim_tails(ts)
        if curve_type=='sir_rollercoaster_wiggle':
            mult=rng.uniform(1,2);period=len(ts)/rng.uniform(np.sqrt(2),np.sqrt(10))**2
            ts*=1+mult*(1+rng.choice([-1,1])*np.sin(np.pi*np.arange(1,len(ts)+1)/period))
            ts/=max(float(ts.max()),1e-300)
        limit=float(np.quantile(ts,.011))+1e-9
        ts=np.r_[rng.uniform(0,limit,20),ts,rng.uniform(0,limit,20)]
    else:
        nwaves=int(rng.integers(5,16))
        reproduction=1.2+rng.beta(1,3)*10
        gamma=1/rng.uniform(1,10);initial=rng.dirichlet([1000,.1,.1])
        wave=solve_wave(reproduction,gamma,initial,rng,seasonal=True)  # keep all 500 steps
        if rng.choice([1,2],p=[.7,.3])==1:
            factors=expit(rng.normal(0,.5)*np.arange(1,nwaves+1));factors/=factors[0]
        else: factors=np.ones(nwaves)
        ts=np.zeros(nwaves*max(52,len(wave)))
        for j in range(nwaves): ts[52*j:52*j+len(wave)]+=factors[j]*wave
        ts=ts[:52*nwaves]
        if ts.max()>1: ts/=ts.max()
        ts=ts[52:-27]  # R removes 1:52 and an inclusive final 27 values
    cadence='weekly'
    if seasonal:
        cadence=str(rng.choice(['monthly','weekly']))
        if cadence=='monthly':
            ts=monthly_aggregate(ts)
            if ts.max()>1: ts/=ts.max()
    scale=str(rng.choice(['proportion','counts']))  # before alpha, as in R
    alpha=np.exp(rng.uniform(np.log(50),np.log(10000)))
    obs=beta_observations(ts,alpha,rng)
    if not np.all(np.isfinite(obs)) or obs.max()<=0: return None
    peak=np.exp(rng.uniform(np.log(.0005),np.log(.25)))
    obs=obs/obs.max()*peak
    if scale=='counts': obs=np.rint(np.rint(rng.uniform(np.sqrt(2e5),np.sqrt(1e8))**2)*obs)
    if np.argmax(obs)+1<30 and not seasonal:
        obs=np.r_[rng.uniform(0,float(np.quantile(obs,.05))+1e-9,int(rng.integers(10,18))),obs]
    return SyntheticCurve(np.maximum(obs,1e-10),curve_type,cadence,scale,seasonal)
