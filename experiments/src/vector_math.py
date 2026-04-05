"""Astigmatism vector mathematics: double-angle decomposition, centroids, errors."""

import numpy as np


def decompose_to_j0_j45(magnitude, meridian_deg):
    """Convert astigmatism (magnitude, meridian) to Cartesian (J0, J45) in double-angle space.

    J0 = magnitude * cos(2 * meridian)   — horizontal/vertical component
    J45 = magnitude * sin(2 * meridian)  — oblique component
    """
    rad = np.radians(2.0 * np.asarray(meridian_deg, dtype=float))
    mag = np.asarray(magnitude, dtype=float)
    return mag * np.cos(rad), mag * np.sin(rad)


def reconstruct_from_j0_j45(j0, j45):
    """Convert Cartesian (J0, J45) back to (magnitude, meridian_deg).

    Meridian is returned in the range [0, 180).
    """
    j0 = np.asarray(j0, dtype=float)
    j45 = np.asarray(j45, dtype=float)
    magnitude = np.sqrt(j0**2 + j45**2)
    meridian_deg = np.degrees(np.arctan2(j45, j0)) / 2.0
    # map to [0, 180)
    meridian_deg = meridian_deg % 180.0
    return magnitude, meridian_deg


def compute_centroid(magnitudes, meridians_deg):
    """Compute the vector centroid of a set of astigmatism vectors.

    Returns dict with centroid magnitude, meridian, mean J0/J45, SD, and n.
    """
    j0, j45 = decompose_to_j0_j45(magnitudes, meridians_deg)
    mean_j0 = np.mean(j0)
    mean_j45 = np.mean(j45)
    cent_mag, cent_mer = reconstruct_from_j0_j45(mean_j0, mean_j45)
    return {
        "centroid_mag": float(cent_mag),
        "centroid_meridian": float(cent_mer),
        "mean_J0": float(mean_j0),
        "mean_J45": float(mean_j45),
        "sd_J0": float(np.std(j0, ddof=1)),
        "sd_J45": float(np.std(j45, ddof=1)),
        "n": len(j0),
    }


def compute_confidence_ellipse(j0_arr, j45_arr, confidence=0.95, is_centroid=True):
    """Compute the 95% confidence ellipse for a set of (J0, J45) points.

    Args:
        is_centroid: If True, returns confidence ellipse of the centroid (scaled by 1/sqrt(n)).
                     If False, returns the data distribution ellipse.

    Returns:
        dict with center_x, center_y, width, height, angle_deg
    """
    from scipy.stats import chi2

    j0_arr = np.asarray(j0_arr, dtype=float)
    j45_arr = np.asarray(j45_arr, dtype=float)
    n = len(j0_arr)

    cov = np.cov(j0_arr, j45_arr)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # sort by descending eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    chi2_val = chi2.ppf(confidence, df=2)

    if is_centroid:
        scale = chi2_val / n
    else:
        scale = chi2_val

    width = 2.0 * np.sqrt(eigenvalues[0] * scale)
    height = 2.0 * np.sqrt(eigenvalues[1] * scale)
    angle_deg = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    return {
        "center_x": float(np.mean(j0_arr)),
        "center_y": float(np.mean(j45_arr)),
        "width": float(width),
        "height": float(height),
        "angle_deg": float(angle_deg),
    }


def vector_error(pred_j0, pred_j45, actual_j0, actual_j45):
    """Euclidean distance in J0/J45 space between predicted and actual vectors."""
    return np.sqrt(
        (np.asarray(pred_j0) - np.asarray(actual_j0)) ** 2
        + (np.asarray(pred_j45) - np.asarray(actual_j45)) ** 2
    )


def angular_error(pred_meridian, actual_meridian):
    """Minimum angular distance in degrees accounting for 0–180 wraparound."""
    diff = np.abs(np.asarray(pred_meridian, dtype=float) - np.asarray(actual_meridian, dtype=float))
    return np.minimum(diff, 180.0 - diff)
