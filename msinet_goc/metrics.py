"""
Evaluation metrics for visual saliency prediction.
Metrics: KLD, CC, NSS, SIM, AUC-Judd
All functions accept numpy arrays (H, W) normalised to [0,1].
"""
import numpy as np


def _normalize(x, eps=1e-7):
    s = x.sum()
    return x / (s + eps)


def kld_metric(gt, pred, eps=1e-7):
    """KL Divergence (lower is better)."""
    gt   = _normalize(gt.astype(np.float64))
    pred = _normalize(pred.astype(np.float64))
    return float(np.sum(gt * np.log(eps + gt / (eps + pred))))


def cc_metric(gt, pred, eps=1e-7):
    """Pearson Correlation Coefficient (higher is better, max 1)."""
    gt   = gt.astype(np.float64).ravel()
    pred = pred.astype(np.float64).ravel()
    gt   -= gt.mean();   pred -= pred.mean()
    denom = np.sqrt((gt**2).sum() * (pred**2).sum()) + eps
    return float(np.sum(gt * pred) / denom)


def sim_metric(gt, pred):
    """Similarity / histogram intersection (higher is better, max 1)."""
    gt   = _normalize(gt.astype(np.float64))
    pred = _normalize(pred.astype(np.float64))
    return float(np.sum(np.minimum(gt, pred)))


def nss_metric(gt_fixations, pred, eps=1e-7):
    """
    Normalized Scanpath Saliency (higher is better).
    gt_fixations: binary map (1 = fixation location).
    pred        : continuous saliency map.
    """
    pred = pred.astype(np.float64)
    pred = (pred - pred.mean()) / (pred.std() + eps)
    fix  = gt_fixations.astype(bool)
    if fix.sum() == 0:
        return 0.0
    return float(pred[fix].mean())


def auc_judd(gt_fixations, pred):
    """
    AUC-Judd (higher is better).
    Treats all fixation pixels as positives and random pixels as negatives.
    """
    pred  = pred.astype(np.float64).ravel()
    fix   = gt_fixations.astype(bool).ravel()
    if fix.sum() == 0 or (~fix).sum() == 0:
        return 0.5

    pos_scores = pred[fix]
    neg_scores = pred[~fix]

    thresholds = np.unique(pos_scores)[::-1]
    tprs, fprs = [], []
    for t in thresholds:
        tprs.append((pos_scores >= t).mean())
        fprs.append((neg_scores >= t).mean())
    tprs = np.array([0.0] + tprs + [1.0])
    fprs = np.array([0.0] + fprs + [1.0])
    try:
        return float(np.trapezoid(tprs, fprs))
    except AttributeError:
        return float(np.trapz(tprs, fprs))


def evaluate_batch(gt_maps, pred_maps, gt_fixations=None):
    """
    Compute all metrics over a batch.
    gt_maps, pred_maps : (N, H, W) numpy arrays in [0, 1]
    gt_fixations       : (N, H, W) binary numpy arrays (required for NSS, AUC)
    Returns dict with mean values.
    """
    results = {k: [] for k in ["KLD", "CC", "SIM", "NSS", "AUC"]}
    for i in range(len(gt_maps)):
        results["KLD"].append(kld_metric(gt_maps[i],  pred_maps[i]))
        results["CC"] .append(cc_metric (gt_maps[i],  pred_maps[i]))
        results["SIM"].append(sim_metric(gt_maps[i],  pred_maps[i]))
        if gt_fixations is not None:
            results["NSS"].append(nss_metric(gt_fixations[i], pred_maps[i]))
            results["AUC"].append(auc_judd  (gt_fixations[i], pred_maps[i]))
    return {k: float(np.mean(v)) for k, v in results.items() if v}
