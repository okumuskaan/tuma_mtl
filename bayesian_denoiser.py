import numpy as np # type: ignore

def log_complex_gaussian_likelihood(r, all_Covs, T, nP, keepdims=False):
    """
    Compute the log-likelihood of a complex Gaussian distribution.

    Parameters:
    r : np.ndarray
        Observation vector.
    all_Covs : np.ndarray
        Covariance matrices for the sampling-based approximation.
    T : np.ndarray
        Residual noise covariance.
    nP : float
        Power factor.
    keepdims : bool, optional
        Whether to keep dimensions of the result (default: False).

    Returns:
    np.ndarray
        Log-likelihood of the observation vector.
    """
    return (-((np.abs(r) ** 2) / (T + nP * all_Covs) + np.log(np.pi * (T + nP * all_Covs))).sum(axis=-1, keepdims=keepdims))

def log_sum_exp_trick(logai, i_axis, keepdims=True):
    """
    Numerically stable computation of log-sum-exp.

    Parameters:
    logai : np.ndarray
        Logarithm of input values.
    i_axis : int
        Axis over which to compute the log-sum-exp.
    keepdims : bool, optional
        Whether to keep the dimensions of the result (default: True).

    Returns:
    np.ndarray
        Log-sum-exp values.
    """
    maxlogai = np.max(logai, axis=i_axis, keepdims=True)
    res = maxlogai + np.log(np.exp(logai - maxlogai).sum(axis=i_axis, keepdims=True))
    return res if keepdims else np.squeeze(res, axis=i_axis)

def compute_logposteriors_with_logsumexptrick(y, all_Covs, T, nP, Ns, log_priors=0.0):
    """
    Compute the log-posterior probabilities using the log-sum-exp trick for numerical stability.

    Parameters:
    y : np.ndarray
        Observation vector.
    all_Covs : np.ndarray
        All covariance matrices for the sampling-based approximation.
    T : np.ndarray
        Residual noise covariance.
    nP : float
        Power factor.
    Ns : int
        Number of samples for sampling-based approximation.
    log_priors : float or np.ndarray
        Logarithm of prior probabilities.

    Returns:
    int, np.ndarray
        Estimated multiplicity and log-posterior probabilities.
    """
    logai = (-((np.abs(y)**2)/(T + nP * all_Covs) + np.log(np.pi * (T + nP * all_Covs))).sum(axis=-1) - np.log(Ns))
    maxlogai = np.max(logai, axis=-1, keepdims=True)
    logposteriors = (maxlogai[:, 0] + np.log(np.exp(logai - maxlogai).sum(axis=-1))) + log_priors
    est_type = np.argmax(logposteriors)
    return est_type, logposteriors

def normalize_posteriors(logposteriors):
    """
    Normalize log-posterior probabilities.

    Parameters:
    logposteriors : np.ndarray
        Log-posterior probabilities.

    Returns:
    np.ndarray
        Normalized posterior probabilities.
    """
    max_posterior = np.max(logposteriors)
    log_sumposteriors = max_posterior + np.log(np.exp(logposteriors - max_posterior).sum())
    return np.exp(logposteriors - log_sumposteriors)

def safe_log1mexp(x, epsilon=1e-12):
    """
    Numerically stable computation of log(1 - exp(x)).

    Parameters:
    x : np.ndarray
        Input values (x <= 0).
    epsilon : float, optional
        Small constant to avoid log(0) (default: 1e-12).

    Returns:
    np.ndarray
        Logarithm of 1 - exp(x).
    """
    x_clipped = np.clip(x, -np.inf, 0)  # Ensure x <= 0
    return np.log(1 - np.exp(x_clipped) + epsilon)

def bayesian_channel_multiplicity_estimation_sampbasedapprox(r, all_covs, taus, nP, log_priors=0.0):
    """
    Estimate channel multiplicity using sampling-based Bayesian approximation.

    Parameters:
    r : np.ndarray
        Effective observation vector.
    all_covs : np.ndarray
        Covariance matrices for sampling-based approximation.
    taus : np.ndarray
        Residual noise covariance.
    nP : float
        Power factor.
    log_priors : float or np.ndarray
        Logarithm of prior probabilities.

    Returns:
    np.ndarray
        Estimated signal for the given observations.
    """
    loglikelihoods_ai = log_complex_gaussian_likelihood(r, all_covs, taus, nP, keepdims=True)
    log_denum_ai = loglikelihoods_ai[1:]
    maxlog_denum_ai = np.max(log_denum_ai, axis=-2, keepdims=True)
    logdenum = maxlog_denum_ai + np.log(np.exp(log_denum_ai - maxlog_denum_ai).sum(axis=-2, keepdims=True))

    log_num_ai = np.log(np.sqrt(nP) * all_covs[1:]) - np.log(taus + nP * all_covs[1:]) + log_denum_ai
    maxlog_num_ai = np.max(log_num_ai, axis=-2, keepdims=True)
    lognum = maxlog_num_ai + np.log(np.exp(log_num_ai - maxlog_num_ai).sum(axis=-2, keepdims=True))

    log_leftpart = lognum - logdenum

    log_posteriors_ai = loglikelihoods_ai + np.expand_dims(log_priors, axis=(-1, -2))
    maxlog_posteriors_ai = np.max(log_posteriors_ai, axis=-2, keepdims=True)
    log_posteriors = maxlog_posteriors_ai + np.log(np.exp(log_posteriors_ai - maxlog_posteriors_ai).sum(axis=-2, keepdims=True))

    max_posterior = np.max(log_posteriors)
    log_sumposteriors = max_posterior + np.log(np.exp(log_posteriors - max_posterior).sum())
    log_rightpart = log_posteriors - log_sumposteriors

    est_X = r * (np.exp(log_rightpart[1:, 0]) * np.exp(log_leftpart[:, 0])).sum(axis=0)

    return est_X

def bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager(r, all_covs, taus, nP, log_priors=0.0):
    """
    Estimate channel multiplicity using sampling-based Bayesian approximation with Onsager correction.

    Parameters:
    r : np.ndarray
        Effective observation vector.
    all_covs : np.ndarray
        Covariance matrices for sampling-based approximation.
    taus : np.ndarray
        Residual noise covariance.
    nP : float
        Power factor.
    log_priors : float or np.ndarray
        Logarithm of prior probabilities.
    withOnsager : bool, optional
        Whether to include Onsager correction (default: True).

    Returns:
    np.ndarray, int, np.ndarray
        Estimated signal, estimated multiplicity, and Onsager reaction term.
    """
    loglikelihoods_ai = log_complex_gaussian_likelihood(r, all_covs, taus, nP, keepdims=True)
    log_denum_ai = loglikelihoods_ai[1:]
    logdenum = log_sum_exp_trick(log_denum_ai, i_axis=-2)

    log_num_ai = np.log(np.sqrt(nP) * all_covs[1:]) - np.log(taus + nP * all_covs[1:]) + log_denum_ai
    lognum = log_sum_exp_trick(log_num_ai, i_axis=-2)

    log_F_brk = lognum - logdenum

    log_posteriors_ai = loglikelihoods_ai + np.expand_dims(log_priors, axis=(-1, -2))
    log_posteriors = log_sum_exp_trick(log_posteriors_ai, i_axis=-2)

    log_sumposteriors = log_sum_exp_trick(log_posteriors, i_axis=0).reshape(-1)[0]
    log_rightpart = log_posteriors - log_sumposteriors

    log_G_rk = log_posteriors[1:] - log_sumposteriors

    H = (np.exp(log_rightpart[1:, 0]) * np.exp(log_F_brk[:, 0])).sum(axis=0)
    est_x = r * H
    est_k = np.argmax(log_posteriors)

    log_estB_rki = np.expand_dims(loglikelihoods_ai, axis=-1) + np.log(-np.expand_dims(np.conj(r), axis=-1)) - np.log(np.expand_dims(taus + nP * all_covs, axis=-1))
    log_estB_rk = log_sum_exp_trick(log_estB_rki, i_axis=1, keepdims=False)
    log_estA_brk = log_sum_exp_trick(
        np.expand_dims(np.log(np.sqrt(nP) * all_covs[1:]) - np.log(taus + nP * all_covs[1:]), axis=-2) + log_estB_rki[1:],
        i_axis=1,
        keepdims=False
    )
    log_estF_brk = log_estA_brk + safe_log1mexp(log_F_brk + log_estB_rk[1:] - log_estA_brk) - logdenum
    log_estC_rk = log_estB_rk + np.expand_dims(log_priors, axis=(-1, -2))
    log_estD_r = log_sum_exp_trick(log_estC_rk, i_axis=0)
    log_estG_rk = log_estC_rk[1:] + safe_log1mexp(log_G_rk + log_estD_r - log_estC_rk[1:]) - log_sumposteriors
    est_H_b = np.sum(np.exp(log_F_brk + log_estG_rk) + np.exp(log_estF_brk + log_G_rk), axis=0)
    Qum = np.diag(H) + r * est_H_b


    return est_x, est_k, Qum

def estimate_type_samplingbased_logsumexp(R, T, all_Covs, M, U, nP, log_priors):
    """
    Estimate message multiplicities using sampling-based log-sum-exp approximation.

    Parameters:
    R : list of np.ndarray
        Effective observations per zone.
    T : np.ndarray
        Residual noise covariance.
    all_Covs : np.ndarray
        Covariance matrices for sampling-based approximation.
    Mus : list of int
        Number of messages per zone.
    nP : float
        Power factor.
    log_priors : list of np.ndarray
        Log-prior probabilities.

    Returns:
    np.ndarray, list of np.ndarray
        Estimated multiplicities and posterior probabilities.
    """
    Mus = [M]*U
    est_k = np.zeros(sum(Mus), dtype=int)
    posteriors = []
    m = 0
    Kmax = all_Covs.shape[1]
    Ns = all_Covs.shape[-2]
    for u, Mu in enumerate(Mus):
        posteriors_u = np.zeros((Mu, Kmax))
        for mu in range(M):
            est_multiplicity, logposteriors = compute_logposteriors_with_logsumexptrick(R[u][mu], all_Covs[u], T, nP, Ns, log_priors[u][mu])
            posteriors_u[mu] = normalize_posteriors(logposteriors)
            est_k[m] = est_multiplicity
            m += 1
        posteriors.append(posteriors_u)
    return est_k, posteriors
