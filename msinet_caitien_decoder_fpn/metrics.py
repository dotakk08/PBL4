"""
Evaluation metrics: KLD, CC, SIM, NSS, AUC-Judd — giống baseline.
"""
import numpy as np

def _normalize(x, eps=1e-7):
    s = x.sum(); return x / (s + eps)

def kld_metric(gt, pred, eps=1e-7):
    gt   = _normalize(gt.astype(np.float64))
    pred = _normalize(pred.astype(np.float64))
    return float(np.sum(gt * np.log(eps + gt / (eps + pred))))

def cc_metric(gt, pred, eps=1e-7):
    gt   = gt.astype(np.float64).ravel(); pred = pred.astype(np.float64).ravel()
    gt  -= gt.mean();  pred -= pred.mean()
    denom = np.sqrt((gt**2).sum() * (pred**2).sum()) + eps
    return float(np.sum(gt * pred) / denom)

def sim_metric(gt, pred):
    gt   = _normalize(gt.astype(np.float64))
    pred = _normalize(pred.astype(np.float64))
    return float(np.sum(np.minimum(gt, pred)))

def nss_metric(gt_fixations, pred, eps=1e-7):
    pred = pred.astype(np.float64)
    pred = (pred - pred.mean()) / (pred.std() + eps)
    fix  = gt_fixations.astype(bool)
    if fix.sum() == 0: return 0.0
    return float(pred[fix].mean())

def auc_judd(gt_fixations, pred):
    pred = pred.astype(np.float64).ravel()
    fix  = gt_fixations.astype(bool).ravel()
    if fix.sum() == 0 or (~fix).sum() == 0: return 0.5
    
    pos_scores = pred[fix]
    neg_scores = pred[~fix]
    
    # Dùng sorting — O(n log n), không tạo matrix lớn
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels     = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    
    sorted_idx = np.argsort(all_scores)[::-1]
    labels_sorted = labels[sorted_idx]
    
    tps = np.cumsum(labels_sorted) / n_pos
    fps = np.cumsum(1 - labels_sorted) / n_neg
    
    tps = np.concatenate([[0.0], tps])
    fps = np.concatenate([[0.0], fps])
    
    try:    return float(np.trapezoid(tps, fps))
    except: return float(np.trapz(tps, fps))

def evaluate_batch(gt_maps, pred_maps, gt_fixations=None):
    results = {k: [] for k in ["KLD", "CC", "SIM", "NSS", "AUC"]}
    for i in range(len(gt_maps)):
        results["KLD"].append(kld_metric(gt_maps[i],  pred_maps[i]))
        results["CC"] .append(cc_metric (gt_maps[i],  pred_maps[i]))
        results["SIM"].append(sim_metric(gt_maps[i],  pred_maps[i]))
        if gt_fixations is not None:
            results["NSS"].append(nss_metric(gt_fixations[i], pred_maps[i]))
            results["AUC"].append(auc_judd  (gt_fixations[i], pred_maps[i]))
    return {k: float(np.mean(v)) for k, v in results.items() if v}
