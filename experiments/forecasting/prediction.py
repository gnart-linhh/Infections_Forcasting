"""Prediction stage. No future truth or scoring values are accepted by this API."""
import numpy as np
from model import forecast
from uncertainty import QUANTILES,LONG_QUANTILES,interval_quantiles


def predict_origin(history,x,y,origin,plan,k_closest=4422,smoothing='gam',seed=13,
                   draws=5000,full_coverage=True,clip_smoothed=True,neighbor_index=None):
    point,indices,distances=forecast(history,x,y,k_closest,smoothing,clip_smoothed,neighbor_index)
    result=[]
    probabilities=LONG_QUANTILES if full_coverage else QUANTILES
    for horizon,value in enumerate(point,1):
        prediction=float(value)
        row=dict(horizon=horizon,field='forecast',point=prediction)
        if plan is not None:
            fit=plan[('forecast',horizon)]
            # Keep the original sMOA RNG key when removing other model outputs.
            rng=np.random.default_rng(np.random.SeedSequence([seed,origin.toordinal(),horizon,0]))
            row.update(fit)
            row['quantiles']=interval_quantiles(prediction,fit['size'],rng,draws,fit['method']=='initial',probabilities)
        result.append(row)
    return result,indices,distances,probabilities
