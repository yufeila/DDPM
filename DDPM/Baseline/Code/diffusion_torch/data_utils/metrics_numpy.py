"""
Classifier metrics using numpy/scipy.

Migrated from: diffusion_tf/tpu_utils/classifier_metrics_numpy.py
Migration notes:
- This file is mostly numpy/scipy based, minimal changes needed
- Removed TF-specific imports
- Functions compute FID and Inception Score from activations/logits
"""

import numpy as np
from scipy import linalg
from typing import Tuple


def classifier_score_from_logits(logits: np.ndarray) -> float:
    """
    Compute Inception Score from classifier logits.
    
    Corresponds to TF's classifier_score_from_logits().
    
    From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
    
    The Inception Score is computed as:
        exp(E_x[KL(p(y|x) || p(y))])
    where p(y|x) is the softmax output of the classifier.
    
    Args:
        logits: [N, num_classes] numpy array of classifier logits
    
    Returns:
        Inception Score (scalar)
    """
    assert len(logits.shape) == 2
    
    # Convert logits to probabilities
    # Stable softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    logits_shifted = logits - logits_max
    probs = np.exp(logits_shifted)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    
    # Compute KL divergence
    # p(y) is the marginal distribution
    p_y = np.mean(probs, axis=0, keepdims=True)
    
    # KL(p(y|x) || p(y)) = sum_y p(y|x) * log(p(y|x) / p(y))
    kl_divs = probs * (np.log(probs + 1e-10) - np.log(p_y + 1e-10))
    kl_divs = np.sum(kl_divs, axis=1)
    
    # Inception Score = exp(mean KL divergence)
    score = np.exp(np.mean(kl_divs))
    
    return float(score)


def frechet_classifier_distance_from_activations(
    activations1: np.ndarray,
    activations2: np.ndarray,
) -> float:
    """
    Compute Frechet Inception Distance (FID) from activations.
    
    Corresponds to TF's frechet_classifier_distance_from_activations().
    
    The FID is computed as:
        ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    where mu1, C1 are mean and covariance of activations1,
    and mu2, C2 are mean and covariance of activations2.
    
    Args:
        activations1: [N, D] numpy array of activations (e.g., from real images)
        activations2: [M, D] numpy array of activations (e.g., from generated images)
    
    Returns:
        FID score (scalar, lower is better)
    """
    assert len(activations1.shape) == 2
    assert len(activations2.shape) == 2
    assert activations1.shape[1] == activations2.shape[1]
    
    # Compute mean and covariance for both sets
    mu1 = np.mean(activations1, axis=0)
    mu2 = np.mean(activations2, axis=0)
    
    sigma1 = np.cov(activations1, rowvar=False)
    sigma2 = np.cov(activations2, rowvar=False)
    
    return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def _calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Calculate Frechet Distance between two multivariate Gaussians.
    
    Corresponds to TF's _calculate_frechet_distance().
    
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    
    Args:
        mu1: Mean of first Gaussian [D]
        sigma1: Covariance matrix of first Gaussian [D, D]
        mu2: Mean of second Gaussian [D]
        sigma2: Covariance matrix of second Gaussian [D, D]
        eps: Small value for numerical stability
    
    Returns:
        Frechet distance (scalar)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Check and fix numerical issues
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


def compute_statistics_from_activations(
    activations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance from activations.
    
    Useful for precomputing statistics for a dataset.
    
    Args:
        activations: [N, D] numpy array of activations
    
    Returns:
        mu: [D] mean vector
        sigma: [D, D] covariance matrix
    """
    assert len(activations.shape) == 2
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def frechet_distance_from_statistics(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """
    Compute FID from precomputed statistics.
    
    Args:
        mu1: [D] mean of first distribution
        sigma1: [D, D] covariance of first distribution
        mu2: [D] mean of second distribution
        sigma2: [D, D] covariance of second distribution
    
    Returns:
        FID score (scalar)
    """
    return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
