import os

import cv2
import numpy as np

from .utils import normalize


def visualizer(pathes, ori_img, anomaly_map, img_size, save_path, cls_name, img_mask, max=None, min=None):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        ori = (ori_img[idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        vis = cv2.cvtColor(cv2.resize(ori, (img_size[0], img_size[1])), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx], max_value=max, min_value=min)
        vis = apply_ad_scoremap(vis, mask)

        # 可视化 GT
        gt_mask = img_mask[idx].squeeze(0).numpy().astype(np.uint8)  # 去掉第一个维度并转换为 uint8
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(vis, contours, isClosed=True, color=(0, 255, 0), thickness=2)  # 红色多边形框

        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
