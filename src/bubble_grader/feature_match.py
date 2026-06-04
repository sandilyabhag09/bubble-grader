"""Compute a photo → reference-sheet homography via ORB feature matching.

This is the marker-free alternative to ArUco detection. It assumes the
unmodified bubble sheet is text-rich and high-contrast (true for ACT-style
sheets), so ORB finds plenty of stable keypoints in both the reference
rasterization and a photograph of the filled sheet.

Pipeline:
  1. ORB.detectAndCompute on both images.
  2. Brute-force Hamming matching + Lowe's ratio test (0.75).
  3. RANSAC homography (reprojection threshold 5 px).
  4. Return H plus diagnostics so callers can fail loudly if matching is weak.
"""

import cv2
import numpy as np


class FeatureMatchError(RuntimeError):
    """Raised when feature matching can't produce a confident homography."""


def compute_homography(
    photo_gray: np.ndarray,
    reference_gray: np.ndarray,
    *,
    n_features: int = 5000,
    ratio_thresh: float = 0.75,
    min_inliers: int = 25,
    ransac_reproj_threshold: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Return (H, info) where H maps photo pixels → reference pixels.

    Raises FeatureMatchError if the match is too weak to trust.
    """
    if photo_gray.ndim != 2 or reference_gray.ndim != 2:
        raise FeatureMatchError("Both inputs must be grayscale (2-D)")

    orb = cv2.ORB_create(
        nfeatures=n_features, scaleFactor=1.2, nlevels=8, edgeThreshold=15,
    )
    kp_photo, des_photo = orb.detectAndCompute(photo_gray, None)
    kp_ref, des_ref = orb.detectAndCompute(reference_gray, None)
    if des_photo is None or des_ref is None or len(kp_photo) < 4 or len(kp_ref) < 4:
        raise FeatureMatchError(
            f"Not enough ORB keypoints (photo={len(kp_photo or [])}, ref={len(kp_ref or [])})"
        )

    # Cross-checked brute-force matching disabled so we can apply Lowe's ratio test.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pair_matches = bf.knnMatch(des_photo, des_ref, k=2)
    good = [m for m, n in (p for p in pair_matches if len(p) == 2)
            if m.distance < ratio_thresh * n.distance]
    if len(good) < min_inliers:
        raise FeatureMatchError(
            f"Only {len(good)} good matches after ratio test (need ≥ {min_inliers})"
        )

    src = np.float32([kp_photo[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj_threshold)
    if H is None:
        raise FeatureMatchError("RANSAC failed to estimate a homography")
    n_inliers = int(mask.sum())
    if n_inliers < min_inliers:
        raise FeatureMatchError(
            f"Only {n_inliers} RANSAC inliers (need ≥ {min_inliers})"
        )

    return H, {
        "n_keypoints_photo": len(kp_photo),
        "n_keypoints_reference": len(kp_ref),
        "n_good_matches": len(good),
        "n_inliers": n_inliers,
        "inlier_ratio": n_inliers / max(1, len(good)),
    }
