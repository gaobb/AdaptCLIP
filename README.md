
# AdaptCLIP

[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/csgaobb/AdaptCLIP)

> Official PyTorch Implementation of [AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection](https://www.arxiv.org/pdf/2505.09926), 2025.

# News 🎉
- [2025-12-15] AdaptCLIP achieves **88.1%** I-AUROC, **93.0%** P-AUROC, and **52.5%** P-AUPR using only 4 training-free normal samples on the large-scale [Real-IAD Variety](https://arxiv.org/pdf/2511.00540), surpassing the state-of-the-art multi-class AD model ([Dinomaly](https://github.com/guojiajeremy/Dinomaly): 85.4% I-AUROC, 91.5% P-AUROC, and 42.8% P-AUPR) that utilizes full normal training images. 

Note: The 1-shot and 2-shot AdaptCLIP results we reported in [Real-IAD-Variety paper](https://arxiv.org/pdf/2511.00540) are lower than the results below, due to our incorrect setting (**pq_context=False**). We will update the new results in the next version with **pq_context=True**.


| Methods | Shot | I-AUROC | P-AUROC | P-AUPR |
|------|------|---------|---------|--------|
| AdaptCLIP-Zero | 0    | 73.0          | 89.2       | 36.2        |
| AdaptCLIP      | 1    | 84.3 ± 0.1    | 92.5 ± 0.1 | 48.9 ± 0.4  |
| AdaptCLIP      | 2    | 86.4 ± 0.1    | 92.8 ± 0.0 | 50.8 ± 0.1  |
| AdaptCLIP      | 4    | **88.1 ± 0.1**    | **93.0 ± 0.0** | **52.5 ± 0.2**  |
| Dinomaly       | full | 85.4          | 91.5       | 42.8        |

## Introduction 
Universal visual anomaly detection aims to identify anomalies from novel or unseen vision domains without additional fine-tuning, which is critical in open scenarios. To this end, we present a **simple yet effective AdaptCLIP** based on **two key insights**:

- Adaptive visual and textual representations should be learned alternately rather than jointly.
- Comparative learning should incorporate contextual and aligned residual features rather than relying solely on residual features.

## AdaptCLIP Framework

![AdaptCLIP](https://arxiv.org/html/2505.09926v2/x2.png)

<div style="display: flex; justify-content: space-between;">
  <img src="https://arxiv.org/html/2505.09926v2/extracted/6447805/figures/AdaptCLIP-PSCode.png" alt="Image 1" style="width: 40%;"  />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <!-- 插入 6 个空格 -->
  <img src="https://arxiv.org/html/2505.09926v2/x1.png" alt="Image 2" style="width: 50%;"  />
</div>


### Training and Testing

- Generate `meta.json`. 
Please refer to [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) for more details.
```bash
  python dataset/mvtec.py # for MVTec
  python dataset/visa.py  # for VisA
```
- Test
Please download pre-train AdaptCLIP models from [huggingface](https://huggingface.co/csgaobb/AdaptCLIP) to `./adaptclip_checkpoints`, and then run the following command:
```bash
bash scripts/test_adaptclip.sh
```

- Train
```bash
bash scripts/train_adaptclip.sh
```

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

Note:  Following previous works, we use AUROC for image-level anomaly classification and AUPR for pixel-level anomaly segmentation in our main paper.
Here, we emphasize that AUPR is better for anomaly segmentation, where the imbalance issue is very extreme between normal and anomaly pixels, as pointed out in [VisA paper (ECCV 2022)](https://arxiv.org/pdf/2207.14315).  In Appendix, we also provide detailed comparisons using all metrics, including AUROC, AUPR, and F1max.

## Complexity and Efficiency Comparisons
| Shots | Methods              | CLIP Models         | Input Size    | # F+L Params (M)      | Inf. Time (ms) |
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

Note: F means Frozen Parameters (M) and L means Learnable Parameters (M)

## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{adaptclip,
  title={AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection},
  author={Gao, Bin-Bin and Zhou, Yue and Yan, Jiangtao and Cai, Yuezhi and Zhang, Weixi and Wang, Meng and Liu, Jun and Liu, Yong and Wang, Lei and Wang, Chengjie},
  booktitle={AAAI}
  year={2026}
}
```

Please also consider citing the following related works from our team if you find them useful in your research:
```
# Anomaly Detection Benchmarks
@inproceedings{real-iad,
        title={Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection},
        author={Wang, Chengjie and Zhu, Wenbing and Gao, Bin-Bin and Gan, Zhenye and Zhang, Jiangning and Gu, Zhihao and Qian, Shuguang and Chen, Mingang and Ma, Lizhuang},
        booktitle={CVPR},
        year={2024}
      }
@inproceedings{real-iad-d3,
        title={Real-IAD D3: A Real-World 2D/Pseudo-3D/3D Dataset for Industrial Anomaly Detection},
        author={Zhu, Wenbing and Wang, Lidong and Zhou, Ziqing and Wang, Chengjie and Pan, Yurui and Zhang, Ruoyi and Chen, Zhuhao and Cheng, Linjie and Gao, Bin-Bin and Zhang, Jiangning and others},
        booktitle={CVPR},
        year={2025}
      }
@article{real-iad-variety,
      title={Real-IAD Variety: Pushing Industrial Anomaly Detection Dataset to a Modern Era},
      author={Zhu, Wenbing and Wang, Chengjie and Gao, Bin-Bin and Zhang, Jiangning and Jiang, Guannan and Hu, Jie and Gan, Zhenye and Wang, Lidong and Zhou, Ziqing and Cheng, Linjie and others},
      journal={arXiv preprint arXiv:2511.00540},
      year={2025}
}
    
# Multi-class Anomaly Detection
@inproceedings{onenip,
  title={Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt},
  author={Gao, Bin-Bin},
  booktitle={ECCV},
  year={2024}
}

# Few-shot Anomaly Generation
@inproceedings{anogen,
  title={Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation},
  author={Gui, Guan and Gao, Bin-Bin and Liu, Jun and Wang, Chengjie and Wu, Yunsheng},
  booktitle={ECCV},
  year={2024}
}

# One-shot Anomaly Segmentation
@inproceedings{metauas,
  title={MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning},
  author={Gao, Bin-Bin},
  booktitle={NeurIPS},
  year={2024}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gaobb/AdaptCLIP&type=Timeline)](https://www.star-history.com/#gaobb/AdaptCLIP&Timeline)


