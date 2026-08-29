import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
import config

def calculate_metrics(targets, probs, thresholds=None):
    """
    Computes diagnostic metrics: AUROC, F1, Precision, Recall, Sensitivity, Specificity, PR-AUC.
    Returns both macro averages and per-class metrics.
    """
    num_classes = targets.shape[1]
    
    # If thresholds not provided, default to 0.5 for all classes
    if thresholds is None:
        thresholds = [0.5] * num_classes
        
    per_class_results = {}
    
    macro_auroc = 0.0
    macro_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_sensitivity = 0.0
    macro_specificity = 0.0
    macro_pr_auc = 0.0
    
    valid_classes_auc = 0
    valid_classes_pr = 0
    
    for c in range(num_classes):
        c_name = config.PATHOLOGIES[c]
        t = targets[:, c]
        p_prob = probs[:, c]
        thresh = thresholds[c]
        p_bin = (p_prob >= thresh).astype(int)
        
        # Calculate True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((t == 1) & (p_bin == 1))
        fp = np.sum((t == 0) & (p_bin == 1))
        tn = np.sum((t == 0) & (p_bin == 0))
        fn = np.sum((t == 1) & (p_bin == 0))
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp + 1e-8)
        
        # Precision, Recall/Sensitivity, F1
        precision = precision_score(t, p_bin, zero_division=0)
        recall = recall_score(t, p_bin, zero_division=0) # Recall is identical to sensitivity
        f1 = f1_score(t, p_bin, zero_division=0)
        
        # AUROC (handle single class case)
        try:
            if len(np.unique(t)) > 1:
                auroc = roc_auc_score(t, p_prob)
                macro_auroc += auroc
                valid_classes_auc += 1
            else:
                auroc = 0.5  # default if no positives/negatives in set
        except Exception:
            auroc = 0.5
            
        # PR-AUC
        try:
            prec_curve, rec_curve, _ = precision_recall_curve(t, p_prob)
            pr_auc = auc(rec_curve, prec_curve)
            if not np.isnan(pr_auc):
                macro_pr_auc += pr_auc
                valid_classes_pr += 1
            else:
                pr_auc = 0.0
        except Exception:
            pr_auc = 0.0
            
        per_class_results[c_name] = {
            "AUROC": float(auroc),
            "F1": float(f1),
            "Precision": float(precision),
            "Recall": float(recall),
            "Sensitivity": float(recall),
            "Specificity": float(specificity),
            "PR-AUC": float(pr_auc),
            "Threshold": float(thresh)
        }
        
        macro_f1 += f1
        macro_precision += precision
        macro_recall += recall
        macro_sensitivity += recall
        macro_specificity += specificity
        
    # Calculate Macro Averages
    macro_results = {
        "AUROC": float(macro_auroc / valid_classes_auc) if valid_classes_auc > 0 else 0.5,
        "F1": float(macro_f1 / num_classes),
        "Precision": float(macro_precision / num_classes),
        "Recall": float(macro_recall / num_classes),
        "Sensitivity": float(macro_sensitivity / num_classes),
        "Specificity": float(macro_specificity / num_classes),
        "PR-AUC": float(macro_pr_auc / valid_classes_pr) if valid_classes_pr > 0 else 0.0,
    }
    
    return macro_results, per_class_results

def optimize_thresholds(targets, probs):
    """
    Grid-searches the optimal F1 threshold per class on the validation set.
    """
    num_classes = targets.shape[1]
    optimal_thresholds = []
    
    for c in range(num_classes):
        t = targets[:, c]
        p_prob = probs[:, c]
        
        best_thresh = 0.5
        best_f1 = 0.0
        
        # Test thresholds from 0.02 to 0.98
        for thresh in np.arange(0.02, 0.98, 0.02):
            p_bin = (p_prob >= thresh).astype(int)
            f1 = f1_score(t, p_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        optimal_thresholds.append(float(best_thresh))
        
    return optimal_thresholds
