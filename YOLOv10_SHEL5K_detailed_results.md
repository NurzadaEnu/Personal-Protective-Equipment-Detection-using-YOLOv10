# YOLOv10 — Detailed Results (SHEL5K Dataset)

Environment:Ultralytics 8.2.74 · Python 3.10.13 · Torch 2.0.1 · CUDA: Tesla V100-SXM2-32GB (31744 MiB)

These are the exact validation results extracted from screenshots of the training logs (train16–train25).

---

YOLOv10n
- **Epochs:** 152  
- **Training time:** 0.55 h  
- **Parameters:** 2,695,586  
- **GFLOPs:** ≈ 8.3  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.882 | 0.826 | 0.885 | 0.522 |
| person | 0.902 | 0.837 | 0.896 | 0.530 |
| vest | 0.882 | 0.799 | 0.846 | 0.496 |
| helmet | 0.917 | 0.843 | 0.913 | 0.539 |

---

 YOLOv10m
- **Epochs:** 119  
- **Training time:** 1.05 h  
- **Parameters:** 16,453,858  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.883 | 0.851 | 0.898 | 0.533 |
| person | 0.889 | 0.857 | 0.907 | 0.545 |
| vest | 0.853 | 0.814 | 0.857 | 0.507 |
| helmet | 0.908 | 0.883 | 0.931 | 0.548 |

---

 YOLOv10l
- **Epochs:** 155  
- **Training time:** 2.08 h  
- **Parameters:** 25,720,994  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.891 | 0.838 | 0.894 | 0.537 |
| person | 0.890 | 0.842 | 0.895 | 0.547 |
| vest | 0.849 | 0.799 | 0.861 | 0.517 |
| helmet | 0.935 | 0.873 | 0.927 | 0.547 |

---

YOLOv10x
- **Epochs:** 71  
- **Training time:** 0.82 h  
- **Parameters:** 31,589,858  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.889 | 0.848 | 0.907 | 0.538 |
| person | 0.879 | 0.842 | 0.901 | 0.545 |
| vest | 0.858 | 0.827 | 0.887 | 0.510 |
| helmet | 0.929 | 0.876 | 0.932 | 0.558 |

---

**Summary:**  
YOLOv10x achieved the best global performance on SHEL5K with mAP@50–95 = 0.538, while maintaining competitive precision and recall across PPE classes.
