"""
Unified First-Principles Gradient Descent Engine (`src/gd_engine.py`).

Every model in this repository is a *Generalized Linear Model* (GLM): it computes a
linear score matrix

    Z = X W + 1_m b          X: (m, n)   W: (n, K)   b: (1, K)   Z: (m, K)

and then applies a model-specific link + loss. Because of this shared structure, the
chain rule always factorizes into the SAME template:

    dJ/dW = X^T dZ  (+ penalty gradient)        dJ/db = 1_m^T dZ

where `dZ = dJ/dZ` (already averaged over the batch) is the only piece that differs
between Linear Regression (dZ = (Z - Y)/m), Logistic Regression (dZ = (sigma(Z) - Y)/m)
and Softmax Regression (dZ = (softmax(Z) - Y)/m). One trainer therefore drives all of
them with Batch, Mini-Batch, or Stochastic Gradient Descent, learning-rate schedules,
and L2 / L1 / ElasticNet regularization (L1 via subgradient or proximal soft-thresholding).

Contents
--------
1. Numerically stable primitives  : sigmoid_stable, softmax_naive/stable, log_softmax_stable, ...
2. Preprocessing                  : StandardScalerScratch, MinMaxScalerScratch, PolynomialFeaturesScratch
3. Learning-rate schedules        : learning_rate_at
4. Pure GD on any function        : minimize_gd, GDResult
5. Spectral diagnostics           : quadratic_hessian_spectrum
6. GLM models + unified trainer   : LinearRegressionGD, LogisticRegressionGD, SoftmaxRegressionGD
7. Closed-form references         : closed_form_linear (Normal Equation, SVD pseudo-inverse, Ridge)
8. Gradient verification          : numerical_gradient, relative_error, check_model_gradients

Pure NumPy only — no scikit-learn, no scipy.optimize.
Run `python -m src.gd_engine` from the repository root to execute the verification suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import warnings

import numpy as np

__all__ = [
    "sigmoid_stable",
    "softmax_naive",
    "softmax_stable",
    "log_softmax_stable",
    "one_hot_encode",
    "soft_threshold",
    "StandardScalerScratch",
    "MinMaxScalerScratch",
    "PolynomialFeaturesScratch",
    "learning_rate_at",
    "GDResult",
    "minimize_gd",
    "quadratic_hessian_spectrum",
    "LinearRegressionGD",
    "LogisticRegressionGD",
    "SoftmaxRegressionGD",
    "closed_form_linear",
    "numerical_gradient",
    "relative_error",
    "check_model_gradients",
    "run_verification_suite",
]

GDMethod = Literal["batch_gd", "minibatch_gd", "sgd"]
Penalty = Literal["none", "l2", "l1", "elasticnet"]
L1Solver = Literal["subgradient", "proximal"]
Schedule = Literal["constant", "inverse_time", "step", "exponential"]


# =============================================================================
# 1. NUMERICALLY STABLE PRIMITIVES
# =============================================================================

def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """
    Elementwise logistic sigmoid that never overflows.

        sigma(z) = 1 / (1 + exp(-z))        for z >= 0   (exp(-z) <= 1, safe)
        sigma(z) = exp(z) / (1 + exp(z))    for z <  0   (exp(z)  <  1, safe)

    The naive single formula overflows `exp(-z)` for z < -709.78 in float64.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr)
    pos = z_arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    exp_z = np.exp(z_arr[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def softmax_naive(Z: np.ndarray) -> np.ndarray:
    """Textbook softmax exp(Z) / sum(exp(Z)). Deliberately unshifted: returns NaN once any Z >= ~710."""
    Z_arr = np.asarray(Z, dtype=np.float64)
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        exp_Z = np.exp(Z_arr)
        return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)


def softmax_stable(Z: np.ndarray) -> np.ndarray:
    """
    Log-Sum-Exp shifted softmax: exp(Z - max Z) / sum(exp(Z - max Z)).

    Subtracting the row maximum M leaves the result unchanged (the factor exp(-M)
    cancels between numerator and denominator) but guarantees every exponent is <= 0,
    so exp(.) lies in (0, 1] and can never overflow; the largest term is exactly 1,
    so the denominator is >= 1 and can never underflow to zero.
    """
    Z_arr = np.asarray(Z, dtype=np.float64)
    Z_shift = Z_arr - np.max(Z_arr, axis=-1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)


def log_softmax_stable(Z: np.ndarray) -> np.ndarray:
    """
    Coupled log-softmax computed entirely in log-space:

        log softmax(Z)_k = (Z_k - M) - log(sum_j exp(Z_j - M))

    Avoids log(0) = -inf when a probability underflows to 0.0.
    """
    Z_arr = np.asarray(Z, dtype=np.float64)
    Z_shift = Z_arr - np.max(Z_arr, axis=-1, keepdims=True)
    return Z_shift - np.log(np.sum(np.exp(Z_shift), axis=-1, keepdims=True))


def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Map integer labels y in {0, ..., K-1}^m to a one-hot matrix Y in {0, 1}^(m x K)."""
    y_int = np.asarray(y).astype(np.int64).ravel()
    K = int(np.max(y_int) + 1) if num_classes is None else int(num_classes)
    assert np.all((y_int >= 0) & (y_int < K)), "Labels must lie in {0, ..., K-1}."
    Y = np.zeros((y_int.shape[0], K), dtype=np.float64)
    Y[np.arange(y_int.shape[0]), y_int] = 1.0
    return Y


def soft_threshold(x: Union[float, np.ndarray], tau: float) -> Union[float, np.ndarray]:
    """
    Proximal operator of tau * |.| (the L1 "shrink-or-kill" operator):

        S_tau(x) = sign(x) * max(|x| - tau, 0)

    Values with |x| <= tau are set to EXACTLY zero — the source of Lasso sparsity.
    """
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


# =============================================================================
# 2. PREPROCESSING: SCALERS & POLYNOMIAL BASIS EXPANSION
# =============================================================================

class StandardScalerScratch:
    """Z-score standardization: x_tilde_j = (x_j - mu_j) / sigma_j  (zero mean, unit variance per feature)."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2, f"Expected X of shape (m, n), got {X_arr.shape}"
        self.mean_ = X_arr.mean(axis=0, keepdims=True)
        std = X_arr.std(axis=0, keepdims=True)
        self.std_ = np.where(std < self.eps, 1.0, std)  # guard constant columns against division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "Call fit() before transform()."
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2 and X_arr.shape[1] == self.mean_.shape[1], "Feature dimension mismatch."
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "Call fit() before inverse_transform()."
        return np.asarray(X_scaled, dtype=np.float64) * self.std_ + self.mean_


class MinMaxScalerScratch:
    """Min-max normalization: maps each feature linearly onto [low, high] (default [0, 1])."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), eps: float = 1e-12) -> None:
        self.feature_range = feature_range
        self.eps = eps
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScalerScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2, f"Expected X of shape (m, n), got {X_arr.shape}"
        self.min_ = X_arr.min(axis=0, keepdims=True)
        self.max_ = X_arr.max(axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.min_ is not None and self.max_ is not None, "Call fit() before transform()."
        X_arr = np.asarray(X, dtype=np.float64)
        span = np.where((self.max_ - self.min_) < self.eps, 1.0, self.max_ - self.min_)
        low, high = self.feature_range
        return (X_arr - self.min_) / span * (high - low) + low

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PolynomialFeaturesScratch:
    """
    Polynomial basis expansion phi(x): all monomials x_1^p1 ... x_n^pn with 1 <= sum(p) <= degree.

    For a single feature this is the Vandermonde map x -> [x, x^2, ..., x^d]. The model
    stays LINEAR IN ITS PARAMETERS, so every gradient formula in this file still applies.
    """

    def __init__(self, degree: int = 2, include_bias: bool = False) -> None:
        assert degree >= 1, f"degree must be >= 1, got {degree}"
        self.degree = degree
        self.include_bias = include_bias
        self.combinations_: List[Tuple[int, ...]] = []
        self.n_input_features_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "PolynomialFeaturesScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2, f"Expected X of shape (m, n), got {X_arr.shape}"
        self.n_input_features_ = X_arr.shape[1]
        start = 0 if self.include_bias else 1
        self.combinations_ = [
            combo
            for deg in range(start, self.degree + 1)
            for combo in combinations_with_replacement(range(self.n_input_features_), deg)
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.n_input_features_ is not None, "Call fit() before transform()."
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2 and X_arr.shape[1] == self.n_input_features_, "Feature dimension mismatch."
        Phi = np.empty((X_arr.shape[0], len(self.combinations_)), dtype=np.float64)
        for col, combo in enumerate(self.combinations_):
            Phi[:, col] = 1.0 if len(combo) == 0 else np.prod(X_arr[:, list(combo)], axis=1)
        return Phi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# =============================================================================
# 3. LEARNING-RATE SCHEDULES
# =============================================================================

def learning_rate_at(
    t: int,
    base_lr: float,
    schedule: Schedule = "constant",
    decay_rate: float = 0.0,
    step_size: int = 50,
) -> float:
    """
    Learning rate alpha_t at epoch t.

    constant     : alpha_t = alpha_0
    inverse_time : alpha_t = alpha_0 / (1 + k t)             (Robbins-Monro compatible: sum = inf, sum^2 < inf)
    step         : alpha_t = alpha_0 * gamma^floor(t / s)     (decay_rate = gamma in (0, 1])
    exponential  : alpha_t = alpha_0 * exp(-k t)
    """
    if schedule == "constant":
        return float(base_lr)
    if schedule == "inverse_time":
        return float(base_lr / (1.0 + decay_rate * t))
    if schedule == "step":
        gamma = decay_rate if decay_rate > 0.0 else 1.0
        return float(base_lr * gamma ** (t // max(step_size, 1)))
    if schedule == "exponential":
        return float(base_lr * np.exp(-decay_rate * t))
    raise ValueError(f"Unknown schedule: {schedule}")


# =============================================================================
# 4. PURE GRADIENT DESCENT ON ANY DIFFERENTIABLE FUNCTION
# =============================================================================

@dataclass
class GDResult:
    """Container for a generic gradient descent run."""

    theta: np.ndarray
    trajectory: np.ndarray                       # (T + 1, P): theta^(0), ..., theta^(T)
    loss_history: np.ndarray                     # (T + 1,)
    grad_norm_history: np.ndarray                # (T + 1,)
    lr_history: np.ndarray                       # (T,)
    converged: bool = False
    diverged: bool = False
    n_steps: int = 0
    meta: Dict[str, float] = field(default_factory=dict)


def minimize_gd(
    fun_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    theta0: np.ndarray,
    learning_rate: float = 0.1,
    n_steps: int = 100,
    schedule: Schedule = "constant",
    decay_rate: float = 0.0,
    step_size: int = 50,
    tol: float = 0.0,
    divergence_threshold: float = 1e12,
) -> GDResult:
    """
    Vanilla gradient descent  theta^(t+1) = theta^(t) - alpha_t * grad J(theta^(t))  on an arbitrary
    function supplied as `fun_and_grad(theta) -> (J, grad)`.

    Stops early when ||grad||_2 < tol (converged) or when J exceeds `divergence_threshold`
    / becomes non-finite (diverged). All iterates are recorded for visualization.
    """
    theta = np.asarray(theta0, dtype=np.float64).ravel().copy()
    J, g = fun_and_grad(theta)
    g = np.asarray(g, dtype=np.float64).ravel()
    assert g.shape == theta.shape, f"Gradient shape {g.shape} != parameter shape {theta.shape}"

    traj, losses, gnorms, lrs = [theta.copy()], [float(J)], [float(np.linalg.norm(g))], []
    converged = diverged = False
    t = 0
    for t in range(n_steps):
        if tol > 0.0 and gnorms[-1] < tol:
            converged = True
            break
        alpha_t = learning_rate_at(t, learning_rate, schedule, decay_rate, step_size)
        theta = theta - alpha_t * g
        with np.errstate(over="ignore", invalid="ignore"):
            J, g = fun_and_grad(theta)
        g = np.asarray(g, dtype=np.float64).ravel()
        traj.append(theta.copy())
        losses.append(float(J))
        gnorms.append(float(np.linalg.norm(g)))
        lrs.append(alpha_t)
        if not np.isfinite(J) or abs(J) > divergence_threshold:
            diverged = True
            break
    else:
        t = n_steps
        converged = tol > 0.0 and gnorms[-1] < tol

    return GDResult(
        theta=theta,
        trajectory=np.array(traj),
        loss_history=np.array(losses),
        grad_norm_history=np.array(gnorms),
        lr_history=np.array(lrs),
        converged=converged,
        diverged=diverged,
        n_steps=len(lrs),
    )


# =============================================================================
# 5. SPECTRAL DIAGNOSTICS FOR QUADRATIC (MSE) OBJECTIVES
# =============================================================================

def quadratic_hessian_spectrum(X: np.ndarray, fit_intercept: bool = True) -> Dict[str, Union[float, np.ndarray]]:
    """
    Eigen-analysis of the MSE Hessian H = (1/m) X_aug^T X_aug  (X_aug = [1, X] if fit_intercept).

    Returns
    -------
    dict with keys
        eigenvalues      : ascending eigenvalues of H
        lambda_min/max   : extreme eigenvalues (curvatures along the flattest/steepest directions)
        condition_number : kappa = lambda_max / lambda_min  (canyon elongation)
        alpha_crit       : 2 / lambda_max                     (GD diverges for alpha > alpha_crit)
        alpha_opt        : 2 / (lambda_min + lambda_max)      (fastest fixed step on a quadratic)
        rate_opt         : (kappa - 1) / (kappa + 1)          (per-step error contraction at alpha_opt)
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    assert X_arr.ndim == 2
    m = X_arr.shape[0]
    X_aug = np.hstack([np.ones((m, 1)), X_arr]) if fit_intercept else X_arr
    eig = np.linalg.eigvalsh(X_aug.T @ X_aug / float(m))
    lam_min = max(float(eig[0]), 1e-15)
    lam_max = float(eig[-1])
    kappa = lam_max / lam_min
    return {
        "eigenvalues": eig,
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "condition_number": kappa,
        "alpha_crit": 2.0 / lam_max,
        "alpha_opt": 2.0 / (lam_min + lam_max),
        "rate_opt": (kappa - 1.0) / (kappa + 1.0),
    }


# =============================================================================
# 6. GENERALIZED LINEAR MODELS + UNIFIED GRADIENT DESCENT TRAINER
# =============================================================================

class _GLMGradientDescentBase:
    """
    Shared trainer for every linear-score model Z = X W + b.

    Objective (bias b is never penalized):

        J(W, b) = DataLoss(Z, Y) + R(W)

        R(W) = 0                                                       penalty="none"
             = (lambda / 2m) ||W||_F^2                                 penalty="l2"
             = (lambda / m)  ||W||_1                                   penalty="l1"
             = (lambda / m) [rho ||W||_1 + (1 - rho)/2 ||W||_F^2]       penalty="elasticnet"

    The `m` in the penalty is ALWAYS the full training-set size, so that a mini-batch
    gradient is an unbiased estimate of the full-objective gradient.

    Parameters
    ----------
    method        : "batch_gd" (B = m), "minibatch_gd" (B = batch_size), "sgd" (B = 1)
    learning_rate : base step size alpha_0
    epochs        : passes over the training set
    batch_size    : B for "minibatch_gd"
    penalty       : "none" | "l2" | "l1" | "elasticnet"
    reg_lambda    : regularization strength lambda >= 0
    l1_ratio      : rho in [0, 1] for "elasticnet"
    l1_solver     : "proximal" (ISTA soft-threshold step -> exact zeros) or "subgradient"
    lr_schedule   : "constant" | "inverse_time" | "step" | "exponential"
    lr_decay      : decay constant k (or gamma for "step")
    lr_step_size  : epochs between drops for "step"
    tol           : stop when the full-batch gradient norm falls below tol (0 disables)
    track_steps   : additionally record parameters + full loss after EVERY update (for SGD path plots)
    random_state  : seed for shuffling
    """

    def __init__(
        self,
        method: GDMethod = "batch_gd",
        learning_rate: float = 0.1,
        epochs: int = 500,
        batch_size: int = 32,
        penalty: Penalty = "none",
        reg_lambda: float = 0.0,
        l1_ratio: float = 0.5,
        l1_solver: L1Solver = "proximal",
        lr_schedule: Schedule = "constant",
        lr_decay: float = 0.0,
        lr_step_size: int = 50,
        tol: float = 0.0,
        track_steps: bool = False,
        random_state: int = 42,
    ) -> None:
        assert method in ("batch_gd", "minibatch_gd", "sgd"), f"Unknown method {method}"
        assert penalty in ("none", "l2", "l1", "elasticnet"), f"Unknown penalty {penalty}"
        assert reg_lambda >= 0.0 and 0.0 <= l1_ratio <= 1.0
        self.method = method
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.penalty = penalty
        self.reg_lambda = float(reg_lambda)
        self.l1_ratio = float(l1_ratio)
        self.l1_solver = l1_solver
        self.lr_schedule = lr_schedule
        self.lr_decay = float(lr_decay)
        self.lr_step_size = int(lr_step_size)
        self.tol = float(tol)
        self.track_steps = track_steps
        self.random_state = random_state

        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.weight_trajectory: List[np.ndarray] = []
        self.lr_history: List[float] = []
        self.step_trajectory: List[np.ndarray] = []
        self.step_loss_history: List[float] = []
        self.n_updates_: int = 0
        self.converged_: bool = False
        self.diverged_: bool = False

    # ------------------------------------------------------------------ subclass API
    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _data_loss_and_dZ(self, Z: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:  # pragma: no cover
        """Return (mean data loss, dJ_data/dZ already divided by the batch size)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        assert X_arr.ndim == 2, f"Expected design matrix X of shape (m, n), got {X_arr.shape}"
        return X_arr

    def _l1_l2_strengths(self) -> Tuple[float, float]:
        """Return (lambda_1, lambda_2) such that R(W) = (lambda_1/m)||W||_1 + (lambda_2/2m)||W||^2."""
        if self.penalty == "l2":
            return 0.0, self.reg_lambda
        if self.penalty == "l1":
            return self.reg_lambda, 0.0
        if self.penalty == "elasticnet":
            return self.reg_lambda * self.l1_ratio, self.reg_lambda * (1.0 - self.l1_ratio)
        return 0.0, 0.0

    def penalty_value(self, W: np.ndarray, m_reg: int) -> float:
        lam1, lam2 = self._l1_l2_strengths()
        return float(lam1 / m_reg * np.sum(np.abs(W)) + lam2 / (2.0 * m_reg) * np.sum(W ** 2))

    def loss_and_gradients(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        m_reg: Optional[int] = None,
        include_l1_subgradient: bool = True,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Full objective J(W, b) and its (sub)gradients.

        Shapes: X (m, n), Y (m, K), W (n, K), b (1, K)  ->  J scalar, grad_W (n, K), grad_b (1, K).
        """
        assert X.ndim == 2 and Y.ndim == 2 and X.shape[0] == Y.shape[0]
        assert W.shape == (X.shape[1], Y.shape[1]) and b.shape == (1, Y.shape[1])
        m_reg = X.shape[0] if m_reg is None else m_reg

        Z = X @ W + b                                  # (m, K) linear scores
        data_loss, dZ = self._data_loss_and_dZ(Z, Y)   # dZ: (m, K)
        grad_W = X.T @ dZ                              # (n, m) @ (m, K) -> (n, K)
        grad_b = np.sum(dZ, axis=0, keepdims=True)     # (1, K)

        lam1, lam2 = self._l1_l2_strengths()
        if lam2 > 0.0:
            grad_W = grad_W + (lam2 / m_reg) * W
        if lam1 > 0.0 and include_l1_subgradient:
            grad_W = grad_W + (lam1 / m_reg) * np.sign(W)
        return data_loss + self.penalty_value(W, m_reg), grad_W, grad_b

    def compute_cost_and_gradients(
        self, X: np.ndarray, y: np.ndarray, W: np.ndarray, b: Union[float, np.ndarray]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Public convenience wrapper accepting raw targets `y`; used by gradient checks and notebooks."""
        X_arr = self._as_2d(X)
        Y = self._prepare_targets(y)
        W_arr = np.asarray(W, dtype=np.float64).reshape(X_arr.shape[1], Y.shape[1])
        b_arr = np.asarray(b, dtype=np.float64).reshape(1, Y.shape[1])
        return self.loss_and_gradients(X_arr, Y, W_arr, b_arr)

    def _record_epoch(self, X: np.ndarray, Y: np.ndarray) -> float:
        assert self.W is not None and self.b is not None
        with np.errstate(over="ignore", invalid="ignore"):
            J, gW, gb = self.loss_and_gradients(X, Y, self.W, self.b)
        self.loss_history.append(float(J))
        self.grad_norm_history.append(float(np.sqrt(np.sum(gW ** 2) + np.sum(gb ** 2))))
        self.weight_trajectory.append(self.theta.copy())
        return float(J)

    def _gd_update(self, Xb: np.ndarray, Yb: np.ndarray, alpha: float, m_full: int) -> None:
        """One parameter update on the batch (Xb, Yb)."""
        assert self.W is not None and self.b is not None
        lam1, _ = self._l1_l2_strengths()
        proximal = lam1 > 0.0 and self.l1_solver == "proximal"
        with np.errstate(over="ignore", invalid="ignore"):
            _, gW, gb = self.loss_and_gradients(
                Xb, Yb, self.W, self.b, m_reg=m_full, include_l1_subgradient=not proximal
            )
        self.W = self.W - alpha * gW
        self.b = self.b - alpha * gb
        if proximal:
            # ISTA: gradient step on the smooth part, then the exact proximal step of (lam1/m)||W||_1
            self.W = soft_threshold(self.W, alpha * lam1 / m_full)
        self.n_updates_ += 1

    # ------------------------------------------------------------------ training
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        W_init: Optional[np.ndarray] = None,
        b_init: Optional[Union[float, np.ndarray]] = None,
    ):
        """Train with the configured gradient descent variant. Returns self."""
        X_arr = self._as_2d(X)
        Y = self._prepare_targets(y)
        m, n = X_arr.shape
        K = Y.shape[1]
        assert Y.shape[0] == m, f"Sample mismatch: X has {m} rows, y has {Y.shape[0]}"

        self.W = (
            np.zeros((n, K)) if W_init is None else np.asarray(W_init, dtype=np.float64).reshape(n, K).copy()
        )
        self.b = (
            np.zeros((1, K)) if b_init is None else np.asarray(b_init, dtype=np.float64).reshape(1, K).copy()
        )
        for hist in (
            self.loss_history, self.grad_norm_history, self.weight_trajectory,
            self.lr_history, self.step_trajectory, self.step_loss_history,
        ):
            hist.clear()
        self.n_updates_ = 0
        self.converged_ = self.diverged_ = False

        J0 = self._record_epoch(X_arr, Y)
        if self.track_steps:
            self.step_trajectory.append(self.theta.copy())
            self.step_loss_history.append(J0)

        B = m if self.method == "batch_gd" else (1 if self.method == "sgd" else min(self.batch_size, m))
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            alpha = learning_rate_at(epoch, self.learning_rate, self.lr_schedule, self.lr_decay, self.lr_step_size)
            self.lr_history.append(alpha)
            order = np.arange(m) if B == m else rng.permutation(m)
            for start in range(0, m, B):
                idx = order[start : start + B]
                self._gd_update(X_arr[idx], Y[idx], alpha, m)
                if self.track_steps:
                    with np.errstate(over="ignore", invalid="ignore"):
                        J_step, _, _ = self.loss_and_gradients(X_arr, Y, self.W, self.b)
                    self.step_trajectory.append(self.theta.copy())
                    self.step_loss_history.append(float(J_step))

            J = self._record_epoch(X_arr, Y)
            if not np.isfinite(J) or not np.all(np.isfinite(self.W)):
                self.diverged_ = True
                break
            if self.tol > 0.0 and self.grad_norm_history[-1] < self.tol:
                self.converged_ = True
                break
        return self

    # ------------------------------------------------------------------ accessors
    @property
    def theta(self) -> np.ndarray:
        """Flattened parameter vector [W.ravel() (row-major), b.ravel()]. For 1 feature & 1 output: [w, b]."""
        assert self.W is not None and self.b is not None
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def trajectory_array(self, per_step: bool = False) -> np.ndarray:
        """Stacked parameter history of shape (T, n*K + K); `per_step=True` uses every update (needs track_steps)."""
        source = self.step_trajectory if per_step else self.weight_trajectory
        assert len(source) > 0, "No trajectory recorded (fit first; per_step requires track_steps=True)."
        return np.vstack(source)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.b is not None, "Model must be fitted first."
        X_arr = self._as_2d(X)
        assert X_arr.shape[1] == self.W.shape[0], "Feature dimension mismatch."
        return X_arr @ self.W + self.b


class LinearRegressionGD(_GLMGradientDescentBase):
    """
    Linear regression  y_hat = X w + b  trained by gradient descent.

    loss="mse"   : J = (1/2m) sum r_i^2                         dJ/dz_i = r_i / m
    loss="huber" : quadratic for |r| <= delta, linear beyond     dJ/dz_i = clip(r_i, -delta, delta) / m
    loss="mae"   : J = (1/m) sum |r_i|                           dJ/dz_i = sign(r_i) / m
    with residual r = y_hat - y.
    """

    def __init__(self, loss: Literal["mse", "huber", "mae"] = "mse", huber_delta: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        assert loss in ("mse", "huber", "mae")
        self.loss = loss
        self.huber_delta = float(huber_delta)

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64).reshape(-1, 1)

    def _data_loss_and_dZ(self, Z: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
        m = Z.shape[0]
        r = Z - Y
        if self.loss == "mse":
            return float(np.sum(r ** 2) / (2.0 * m)), r / m
        if self.loss == "huber":
            d = self.huber_delta
            quad = np.abs(r) <= d
            per = np.where(quad, 0.5 * r ** 2, d * (np.abs(r) - 0.5 * d))
            return float(np.mean(per)), np.clip(r, -d, d) / m
        return float(np.mean(np.abs(r))), np.sign(r) / m

    @property
    def w(self) -> np.ndarray:
        """Weight column vector of shape (n, 1)."""
        assert self.W is not None
        return self.W

    @property
    def bias(self) -> float:
        assert self.b is not None
        return float(self.b[0, 0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X).ravel()

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_true = np.asarray(y, dtype=np.float64).ravel()
        ss_res = float(np.sum((y_true - self.predict(X)) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-15)


class LogisticRegressionGD(_GLMGradientDescentBase):
    """
    Binary logistic regression  p = sigma(X w + b)  trained by gradient descent.

    loss="bce" : J = (1/m) sum [log(1 + e^z) - y z]   (exact, overflow-free form of cross-entropy)
                 dJ/dz_i = (p_i - y_i) / m            (sigma' cancels -> convex, never saturates)
    loss="mse" : J = (1/2m) sum (p - y)^2             (NON-convex; kept for pedagogical contrast)
                 dJ/dz_i = (p_i - y_i) p_i (1 - p_i) / m
    """

    def __init__(self, loss: Literal["bce", "mse"] = "bce", **kwargs) -> None:
        super().__init__(**kwargs)
        assert loss in ("bce", "mse")
        self.loss = loss

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        Y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        assert np.all((Y == 0.0) | (Y == 1.0)), "Binary targets must be in {0, 1}."
        return Y

    def _data_loss_and_dZ(self, Z: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
        m = Z.shape[0]
        P = sigmoid_stable(Z)
        if self.loss == "bce":
            # -y log(sigma(z)) - (1-y) log(1-sigma(z)) == log(1 + e^z) - y z ; logaddexp avoids log(0)
            return float(np.mean(np.logaddexp(0.0, Z) - Y * Z)), (P - Y) / m
        err = P - Y
        return float(0.5 * np.mean(err ** 2)), err * P * (1.0 - P) / m

    @property
    def w(self) -> np.ndarray:
        """Weight column vector of shape (n, 1)."""
        assert self.W is not None
        return self.W

    @property
    def bias(self) -> float:
        assert self.b is not None
        return float(self.b[0, 0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid_stable(self.decision_function(X)).ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(np.int64).ravel()))


class SoftmaxRegressionGD(_GLMGradientDescentBase):
    """
    Multiclass softmax (multinomial logistic) regression  P = softmax(X W + b).

    J = -(1/m) sum_i sum_k Y_ik log P_ik         dJ/dZ = (P - Y) / m

    mode="log_sum_exp" : shifted softmax + coupled log-softmax (production-grade, overflow-free)
    mode="naive"       : unshifted exp(Z) (overflows to NaN for logits >= ~710; for demonstration)
    """

    def __init__(
        self,
        mode: Literal["log_sum_exp", "naive"] = "log_sum_exp",
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert mode in ("log_sum_exp", "naive")
        self.mode = mode
        self.num_classes = num_classes

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y)
        if y_arr.ndim == 2 and y_arr.shape[1] > 1:
            return y_arr.astype(np.float64)  # already one-hot
        return one_hot_encode(y_arr, self.num_classes)

    def _data_loss_and_dZ(self, Z: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
        m = Z.shape[0]
        if self.mode == "log_sum_exp":
            P = softmax_stable(Z)
            loss = float(-np.sum(Y * log_softmax_stable(Z)) / m)
        else:
            P = softmax_naive(Z)
            with np.errstate(divide="ignore", invalid="ignore"):
                loss = float(-np.sum(Y * np.log(P)) / m)
        return loss, (P - Y) / m

    @property
    def w(self) -> np.ndarray:
        """Weight column vector of shape (n, 1)."""
        assert self.W is not None
        return self.W

    @property
    def bias(self) -> float:
        assert self.b is not None
        return float(self.b[0, 0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.decision_function(X)
        return softmax_stable(Z) if self.mode == "log_sum_exp" else softmax_naive(Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(np.int64).ravel()))


# =============================================================================
# 7. CLOSED-FORM REFERENCE SOLUTIONS (to verify where GD should land)
# =============================================================================

def closed_form_linear(
    X: np.ndarray,
    y: np.ndarray,
    method: Literal["svd", "normal_eq"] = "svd",
    l2_lambda: float = 0.0,
) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
    """
    Exact minimizer of (1/2m)||X w + b - y||^2 + (l2_lambda/2m)||w||^2 (bias unpenalized).

    l2_lambda == 0, "normal_eq" : theta* = (X_aug^T X_aug)^{-1} X_aug^T y
    l2_lambda == 0, "svd"       : theta* = V Sigma^+ U^T y  (Moore-Penrose; minimum-norm if rank-deficient)
    l2_lambda  > 0              : w* = (X_c^T X_c + lambda I)^{-1} X_c^T y_c,  b* = y_bar - x_bar^T w*

    Returns (w of shape (n, 1), b, info) where info may contain "singular_values".
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    m, n = X_arr.shape
    info: Dict[str, np.ndarray] = {}

    if l2_lambda > 0.0:
        x_bar = X_arr.mean(axis=0, keepdims=True)
        y_bar = float(y_arr.mean())
        X_c, y_c = X_arr - x_bar, y_arr - y_bar
        w = np.linalg.solve(X_c.T @ X_c + l2_lambda * np.eye(n), X_c.T @ y_c)
        return w, float(y_bar - (x_bar @ w)[0, 0]), info

    X_aug = np.hstack([np.ones((m, 1)), X_arr])
    if method == "normal_eq":
        theta = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y_arr)
    else:
        U, s, Vt = np.linalg.svd(X_aug, full_matrices=False)
        tol = np.finfo(np.float64).eps * max(X_aug.shape) * s[0]
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        theta = (Vt.T * s_inv) @ (U.T @ y_arr)
        info["singular_values"] = s
    return theta[1:], float(theta[0, 0]), info


# =============================================================================
# 8. FINITE-DIFFERENCE GRADIENT VERIFICATION
# =============================================================================

def numerical_gradient(f: Callable[[np.ndarray], float], theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centered finite differences: dJ/dtheta_j ~= [J(theta + eps e_j) - J(theta - eps e_j)] / (2 eps).  Error O(eps^2)."""
    theta = np.asarray(theta, dtype=np.float64).ravel()
    grad = np.zeros_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = eps
        grad[j] = (f(theta + e) - f(theta - e)) / (2.0 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / (||a|| + ||b||): scale-free agreement measure (< 1e-7 means the gradient is correct)."""
    a, b = np.ravel(a), np.ravel(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-15))


def check_model_gradients(
    model: _GLMGradientDescentBase,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
    seed: int = 0,
) -> float:
    """Compare the analytical gradient of `model` at a random (W, b) against centered finite differences."""
    X_arr = model._as_2d(X)
    Y = model._prepare_targets(y)
    n, K = X_arr.shape[1], Y.shape[1]
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(n, K))
    b0 = rng.normal(size=(1, K))

    def unpack(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return theta[: n * K].reshape(n, K), theta[n * K :].reshape(1, K)

    def J(theta: np.ndarray) -> float:
        W, b = unpack(theta)
        return model.loss_and_gradients(X_arr, Y, W, b)[0]

    _, gW, gb = model.loss_and_gradients(X_arr, Y, W0, b0)
    analytic = np.concatenate([gW.ravel(), gb.ravel()])
    numeric = numerical_gradient(J, np.concatenate([W0.ravel(), b0.ravel()]), eps)
    return relative_error(analytic, numeric)


def run_verification_suite(verbose: bool = True) -> Dict[str, float]:
    """Gradient checks for every model/loss/penalty combination plus solver parity & stability tests."""
    rng = np.random.default_rng(2026)
    m, n, K = 48, 5, 4
    X = rng.normal(size=(m, n))
    y_reg = X @ rng.uniform(-2, 2, size=(n, 1)) - 0.7 + rng.normal(0, 0.3, size=(m, 1))
    y_bin = (rng.uniform(size=m) < 0.5).astype(np.float64)
    y_cls = rng.integers(0, K, size=m)

    cases = {
        "linear_mse": LinearRegressionGD(loss="mse"),
        "linear_huber": LinearRegressionGD(loss="huber", huber_delta=1.1),
        "linear_mse_l2": LinearRegressionGD(loss="mse", penalty="l2", reg_lambda=2.5),
        "linear_mse_l1": LinearRegressionGD(loss="mse", penalty="l1", reg_lambda=1.5),
        "linear_mse_elasticnet": LinearRegressionGD(loss="mse", penalty="elasticnet", reg_lambda=2.0, l1_ratio=0.3),
        "logistic_bce": LogisticRegressionGD(loss="bce"),
        "logistic_mse": LogisticRegressionGD(loss="mse"),
        "logistic_bce_l2": LogisticRegressionGD(loss="bce", penalty="l2", reg_lambda=0.8),
        "logistic_bce_l1": LogisticRegressionGD(loss="bce", penalty="l1", reg_lambda=0.6),
        "softmax_lse": SoftmaxRegressionGD(num_classes=K),
        "softmax_lse_l2": SoftmaxRegressionGD(num_classes=K, penalty="l2", reg_lambda=0.5),
    }
    targets = {"linear": y_reg, "logistic": y_bin, "softmax": y_cls}
    results: Dict[str, float] = {}
    for name, model in cases.items():
        err = check_model_gradients(model, X, targets[name.split("_")[0]], seed=hash(name) % 1000)
        results[name] = err
        assert err < 1e-7, f"Gradient check FAILED for {name}: relative error {err:.3e}"

    # Generic minimize_gd on an anisotropic quadratic bowl J = 0.5 theta^T A theta
    A = np.diag([4.0, 0.5])
    res = minimize_gd(lambda t: (0.5 * float(t @ A @ t), A @ t), np.array([2.0, -3.0]), learning_rate=0.4, n_steps=200)
    assert np.linalg.norm(res.theta) < 1e-8, "minimize_gd failed to converge on a quadratic bowl"
    results["minimize_gd_quadratic_final_norm"] = float(np.linalg.norm(res.theta))

    # Batch GD vs closed-form SVD parity (standardized features)
    Xs = StandardScalerScratch().fit_transform(X)
    gd = LinearRegressionGD(learning_rate=0.3, epochs=600).fit(Xs, y_reg)
    w_svd, b_svd, _ = closed_form_linear(Xs, y_reg)
    parity = float(np.linalg.norm(np.vstack([gd.w, [[gd.bias]]]) - np.vstack([w_svd, [[b_svd]]])))
    assert parity < 1e-8, f"GD/SVD parity failed: {parity:.3e}"
    results["batch_gd_vs_svd_param_gap"] = parity

    # Ridge GD vs Ridge closed form
    lam = 7.0
    ridge = LinearRegressionGD(penalty="l2", reg_lambda=lam, learning_rate=0.3, epochs=800).fit(Xs, y_reg)
    w_r, b_r, _ = closed_form_linear(Xs, y_reg, l2_lambda=lam)
    ridge_gap = float(np.linalg.norm(np.vstack([ridge.w, [[ridge.bias]]]) - np.vstack([w_r, [[b_r]]])))
    assert ridge_gap < 1e-8, f"Ridge GD/closed-form parity failed: {ridge_gap:.3e}"
    results["ridge_gd_vs_closed_form_gap"] = ridge_gap

    # Proximal GD (ISTA) Lasso: verify KKT optimality conditions
    lam1 = 12.0
    lasso = LinearRegressionGD(penalty="l1", reg_lambda=lam1, learning_rate=0.3, epochs=3000).fit(Xs, y_reg)
    r = Xs @ lasso.w + lasso.bias - y_reg
    g_data = (Xs.T @ r / m).ravel()
    w_hat = lasso.w.ravel()
    nz = np.abs(w_hat) > 1e-12
    kkt = max(
        float(np.max(np.abs(g_data[nz] + lam1 / m * np.sign(w_hat[nz])), initial=0.0)),
        float(np.max(np.abs(g_data[~nz]) - lam1 / m, initial=-np.inf)),
    )
    assert kkt < 1e-8, f"Lasso KKT violation {kkt:.3e}"
    results["lasso_ista_kkt_violation"] = max(kkt, 0.0)

    # Softmax overflow immunity
    extreme = np.array([[1000.0, 1001.0, 999.0], [-1200.0, -1201.0, -1205.0]])
    assert np.isnan(softmax_naive(extreme)).any() and np.all(np.isfinite(softmax_stable(extreme)))
    assert np.allclose(softmax_stable(extreme).sum(axis=1), 1.0)
    assert np.all(np.isfinite(log_softmax_stable(extreme)))
    assert np.all(np.isfinite(sigmoid_stable(np.array([-1e4, 0.0, 1e4]))))

    if verbose:
        print("=" * 76)
        print("gradient-descent-from-scratch :: src/gd_engine.py VERIFICATION SUITE")
        print("=" * 76)
        for name, val in results.items():
            tag = "grad rel. err" if name in cases else "metric"
            print(f"  {name:<36} {tag:>14}: {val:.3e}   PASS")
        print("  softmax / sigmoid extreme-logit stability                     PASS")
        print("=" * 76)
    return results


if __name__ == "__main__":
    run_verification_suite(verbose=True)
    print("[SUCCESS] All gradient checks < 1e-7 and all solver parity tests passed.")

