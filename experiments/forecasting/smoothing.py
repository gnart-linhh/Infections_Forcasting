"""Gaussian thin plate regression spline with GCV, as used by sMOA's GAM.

Mathematical implementation of Wood (2003), not an R runtime or a call to
mgcv. Retain the k largest-magnitude eigenpairs of the cubic radial kernel,
impose the affine constraints in that truncated space, then penalize integrated
squared curvature. Orthogonal reparameterization does not change fitted values.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh, null_space, qr, solve_triangular
from scipy.optimize import minimize_scalar


@lru_cache(maxsize=512)
def _basis(n, k):
    t = np.linspace(-.5, .5, n)
    kernel = np.abs(t[:, None] - t[None, :]) ** 3
    polynomial = np.column_stack([np.ones(n), t])
    eigenvalues, eigenvectors = eigh(kernel)
    keep = np.argsort(np.abs(eigenvalues))[::-1][:k]
    u, d = eigenvectors[:, keep], eigenvalues[keep]
    z = null_space(polynomial.T @ u)
    radial = u @ z
    design = np.column_stack([kernel @ radial, polynomial])
    penalty = np.zeros((k, k))
    penalty[:k-2, :k-2] = z.T @ (d[:, None] * z)
    q, r = qr(design, mode='economic')
    inverse_r = solve_triangular(r, np.eye(k))
    eigenvalues, rotation = eigh(inverse_r.T @ penalty @ inverse_r)
    # The two affine directions are exactly unpenalized.
    eigenvalues[:2] = 0
    if np.any(eigenvalues[2:] <= 0):
        raise ArithmeticError('Thin plate penalty lost positive definiteness')
    eigenvalues /= np.median(eigenvalues[2:])
    return q @ rotation, eigenvalues


def thin_plate_gcv(values, basis_size=None, return_info=False):
    values = np.asarray(values, float)
    n = len(values)
    k = max(3, int(round(n / 2))) if basis_size is None else int(basis_size)
    if values.ndim != 1 or n < 3 or not np.isfinite(values).all() or not 3 <= k <= n:
        raise ValueError('Need finite 1D history, n>=3 and 3<=k<=n')
    basis, penalty = _basis(n, k)
    # Normalize responses to stabilize GCV without changing the optimum.
    offset = values.mean()
    scale = max(float(np.std(values)), 1.)
    response = (values - offset) / scale
    coefficients = basis.T @ response
    residual = response - basis @ coefficients
    residual_ss = float(residual @ residual)

    def fit(log_lambda):
        product = np.exp(log_lambda) * penalty
        shrink = 1 / (1 + product)
        loss = product / (1 + product)
        rss = (0. if n==k else residual_ss) + np.sum((loss * coefficients) ** 2)
        residual_df = n-k + float(loss.sum())
        return n * rss / residual_df ** 2, shrink, n-residual_df

    grid = np.linspace(-30, 30, 121)
    objectives = [fit(x)[0] for x in grid]
    candidates = [(-30., objectives[0]), (30., objectives[-1])]
    for i in range(1, len(grid)-1):
        if objectives[i] <= objectives[i-1] and objectives[i] <= objectives[i+1]:
            optimum = minimize_scalar(lambda x: fit(x)[0], bounds=(grid[i-1], grid[i+1]), method='bounded')
            candidates.append((float(optimum.x), float(optimum.fun)))
    minimum = min(value for _, value in candidates)
    # A flat GCV curve (notably n=k=3) cannot identify lambda. Choose the
    # smoother tied solution rather than a floating-point interpolation artifact.
    optimum = max(x for x,value in candidates if value <= minimum + 1e-10*max(minimum,1e-20))
    gcv, shrink, edf = fit(optimum)
    fitted = offset + scale * (basis @ (shrink * coefficients))
    info = dict(basis_size=k, edf=edf, gcv=gcv * scale**2,
                log_lambda=optimum, method='thin_plate_gcv')
    return (fitted, info) if return_info else fitted
