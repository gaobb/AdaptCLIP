"""Efficient metric evaluator for anomaly detection."""

from collections import defaultdict

import torch
from torch.nn import functional as F

from metrics import AUPR, AUPRO, AUROC, F1Max, OverkillEscape


class Evaluator:
    """Evaluator for computing anomaly detection metrics.

    Args:
        device: Device to run metrics on (cuda or cpu)
        metrics: List of metric names to compute. If empty, uses default metrics.
    """

    def __init__(self, device, metrics=None, sample_level=False):
        if metrics is None or len(metrics) == 0:
            self.metrics = [
                'I-AUROC', 'I-AP', 'I-F1max',
                'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'
            ]
        else:
            self.metrics = metrics
        self.sample_level = sample_level
        self.aupr = AUPR().to(device)
        self.aupro = AUPRO().to(device)
        self.auroc = AUROC().to(device)
        self.f1max = F1Max().to(device)
        self.overkill_escape_2 = OverkillEscape(escape_rate=2).to(device)
        self.overkill_escape_5 = OverkillEscape(escape_rate=5).to(device)
        self.overkill_escape_10 = OverkillEscape(escape_rate=10).to(device)

    def run(self, results, cls_name, logger=None):
        """Run evaluation metrics for a specific class.

        Args:
            results: Dictionary containing predictions and ground truth
            cls_name: Name of the class to evaluate
            logger: Optional logger for output

        Returns:
            dict: Dictionary of metric names and their values
        """
        idxes = results['cls_names'] == cls_name

        gt_px = results['gt_masks'][idxes]
        pr_px = results['pr_masks'][idxes]

        gt_sp = results['gt_anomalys'][idxes]
        pr_sp = results['pr_anomalys'][idxes]

        # --- compute sample-level results ---
        if self.sample_level:
            s_ids = results['sample_ids'][idxes]
            sample_data = defaultdict(lambda: {'gt': [], 'pr': []})
            for i, sid in enumerate(s_ids):
                sample_data[sid]['gt'].append(gt_sp[i])
                sample_data[sid]['pr'].append(pr_sp[i])

            gt_s = torch.stack([torch.stack(sample_data[sid]['gt']).max() for sid in sample_data])
            pr_s = torch.stack([torch.stack(sample_data[sid]['pr']).max() for sid in sample_data])


        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)

        eval_results = {}
        for metric in self.metrics:
            if metric.startswith('S-AUROC'):
                eval_results[metric] = self.auroc(pr_s, gt_s).item()

            if metric.startswith('I-AUROC'):
                eval_results[metric] = self.auroc(pr_sp, gt_sp).item()

            elif metric.startswith('P-AUROC'):
                eval_results[metric] = self.auroc(pr_px.ravel(), gt_px.ravel()).item()

            elif metric.startswith('S-AP'):
                eval_results[metric] = self.aupr(pr_s, gt_s).item()

            elif metric.startswith('I-AP'):
                eval_results[metric] = self.aupr(pr_sp, gt_sp).item()

            elif metric.startswith('P-AP'):
                eval_results[metric] = self.aupr(pr_px.ravel(), gt_px.ravel()).item()

            elif metric.startswith('S-F1max'):
                eval_results[metric] = self.f1max(pr_s, gt_s).item()

            elif metric.startswith('I-F1max'):
                eval_results[metric] = self.f1max(pr_sp, gt_sp).item()

            elif metric.startswith('P-F1max'):
                eval_results[metric] = self.f1max(pr_px.ravel(), gt_px.ravel()).item()

            elif metric.startswith('P-AUPRO'):
                eval_results[metric] = self.aupro(pr_px, gt_px).item()

            elif metric.startswith('I-Overkill@2'):
                eval_results[metric] = self.overkill_escape_2(pr_sp, gt_sp).item()

            elif metric.startswith('I-Overkill@5'):
                eval_results[metric] = self.overkill_escape_5(pr_sp, gt_sp).item()

            elif metric.startswith('I-Overkill@10'):
                eval_results[metric] = self.overkill_escape_10(pr_sp, gt_sp).item()

            elif metric.startswith('S-Overkill@2'):
                eval_results[metric] = self.overkill_escape_2(pr_s, gt_s).item()

            elif metric.startswith('S-Overkill@5'):
                eval_results[metric] = self.overkill_escape_5(pr_s, gt_s).item()

            elif metric.startswith('S-Overkill@10'):
                eval_results[metric] = self.overkill_escape_10(pr_s, gt_s).item()

        return eval_results
