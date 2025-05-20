
# AdaptCLIP

[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/csgaobb/AdaptCLIP)

> Official PyTorch Implementation of [AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection](https://www.arxiv.org/pdf/2505.09926), 2025.



## Introduction 
Universal visual anomaly detection aims to identify anomalies from novel or unseen vision domains without additional fine-tuning, which is critical in open scenarios. 

- Adaptive visual and textual representations should be learned alternately rather than jointly.
- Comparative learning should incorporate contextual and aligned residual features rather than relying solely on residual features.

## AdaptCLIP Framework

![AdaptCLIP](https://arxiv.org/html/2505.09926v2/x2.png)

<div style="display: flex; justify-content: space-between;">
  <img src="https://arxiv.org/html/2505.09926v2/extracted/6447805/figures/AdaptCLIP-PSCode.png" alt="Image 1" style="width: 40%;"  />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <!-- 插入 6 个空格 -->
  <img src="https://arxiv.org/html/2505.09926v2/x1.png" alt="Image 2" style="width: 50%;"  />
</div>


## Ablation Studies

| No. | Methods      | Shots | TA    | VA    | PQA         | MVTec         | VisA         |
|-----|--------------|-------|-------|-------|-------------|---------------|--------------|
| 0   | baselines    | 0     | ✗     | ✗     | ✗           | 91.1 / 33.0   | 82.1 / 18.0  |
| 1   | baselines    | 0     | ✓     | ✗     | ✗           | 92.2 / 31.4   | 82.9 / 19.7  |           
| 2   | baselines    | 0     | ✗     | ✓     | ✗           | 90.5 / 39.4   | 81.0 / 22.1  |
| 3   | joint        | 0     | ✓     | ✓     | ✗           | 89.3 / 36.2   | 81.6 / 21.5  |
| 4   | **alternating**  | 0     | ✓     | ✓     | ✗           | 93.5 / 38.3   | 84.8 / 26.1  |
| 5   | w/o context  | 1     | ✗     | ✗     | ✓           | 62.6 / 7.0    | 85.3 / 28.7  |
| 6   | **w context**    | 1     | ✗     | ✗     | ✓           | 88.1 / 50.2   | 88.9 / 38.1  |
| 7   | **AdaptCLIP**    | 1     | ✓     | ✓     | ✓           | 94.2 / 52.5   | 92.0 / 38.8  |

## Complexity and Efficiency Comparisons
| Shots | Methods              | CLIP Models         | Input Size    | # Params (M)       | Inf. Time (ms) |
|-------|----------------------|---------------------|---------------|--------------------|----------------|
| 0     | WinCLIP [16]         | ViT-B-16+240        | 240×240       | 208.4 + 0.0        | 201.3          |
| 0     | WinCLIP [16]         | ViT-B-16+240        | 512×512       | 208.4 + 0.0        | 3912.6         |
| 0     | AdaCLIP [6]          | ViT-L/14@336px      | 518×518       | 428.8 + 10.7       | 212.0          |
| 0     | AnomalyCLIP [53]     | ViT-L/14@336px      | 518×518       | 427.9 + 5.6        | 154.9          |
| 0     | **AdaptCLIP-Zero**       | ViT-B-16+240        | 512×512       | 208.4 + 0.4        | 49.9           |
| 0     | **AdaptCLIP-Zero**       | ViT-L/14@336px      | 518×518       | 427.9 + 0.6        | 162.2          | 
| 1     | WinCLIP+ [16]        | ViT-B-16+240        | 240×240       | 208.4 + 0.0        | 339.5          |
| 1     | WinCLIP+ [16]        | ViT-B-16+240        | 512×512       | 208.4 + 0.0        | 7434.9         |
| 1     | InCtrl [54]          | ViT-B-16+240        | 240×240       | 208.4 + 0.3        | 337.0          |
| 1     | AnomalyCLIP+ [53]    | ViT-L/14@336px      | 518×518       | 427.9 + 5.6        | 158.6          |
| 1     | **AdaptCLIP**            | ViT-B-16+240        | 512×512       | 208.4 + 1.4        | 54.0           |
| 1     | **AdaptCLIP**            | ViT-L/14@336px      | 518×518       | 427.9 + 1.8        | 168.2          |


## ToDo List
- [ ] release pre-trained AdaptCLIP models
- [ ] deploy online AdaptCLIP Demo on [HuggingFace Space](https://huggingface.co/spaces/csgaobb/AdaptCLIP)
- [ ] open testing code
- [ ] open training code




