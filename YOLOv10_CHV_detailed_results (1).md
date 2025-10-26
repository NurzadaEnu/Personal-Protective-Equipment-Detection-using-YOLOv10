# YOLOv10 — Detailed Results (CHV Dataset)

**Environment:** Ultralytics 8.2.91 · Python 3.12.4 · Torch 2.4.1+cu121 · Quadro RTX 6000 (24 GB)

This document summarizes YOLOv10 (n, s, m, b, l, x) results on the CHV dataset, including detailed metrics by class.

---

 YOLOv10n
- **Epochs:** 243  
- **Training time:** 1.15 h  
- **Parameters:** 2,696,756  
- **GFLOPs:** 8.2  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.934 | 0.791 | 0.890 | 0.540 |
| person | 0.949 | 0.842 | 0.916 | 0.562 |
| vest | 0.892 | 0.737 | 0.819 | 0.485 |
| blue helmet | 0.973 | 0.744 | 0.883 | 0.545 |
| red helmet | 0.884 | 0.792 | 0.887 | 0.560 |
| white helmet | 0.966 | 0.853 | 0.935 | 0.600 |
| yellow helmet | 0.941 | 0.779 | 0.903 | 0.490 |

---

YOLOv10s
- **Epochs:** 419  
- **Training time:** 1.96 h  
- **Parameters:** 8,039,604  
- **GFLOPs:** 24.5  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.920 | 0.851 | 0.911 | 0.561 |
| person | 0.942 | 0.876 | 0.932 | 0.584 |
| vest | 0.863 | 0.795 | 0.871 | 0.540 |
| blue helmet | 0.975 | 0.794 | 0.898 | 0.563 |
| red helmet | 0.867 | 0.896 | 0.923 | 0.564 |
| white helmet | 0.942 | 0.919 | 0.940 | 0.622 |
| yellow helmet | 0.931 | 0.828 | 0.902 | 0.495 |

---

 YOLOv10m
- **Epochs:** 216  
- **Training time:** 1.16 h  
- **Parameters:** 16,457,332  
- **GFLOPs:** 63.4  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.878 | 0.845 | 0.893 | 0.551 |
| person | 0.916 | 0.876 | 0.920 | 0.566 |
| vest | 0.840 | 0.770 | 0.833 | 0.506 |
| blue helmet | 0.936 | 0.893 | 0.912 | 0.569 |
| red helmet | 0.788 | 0.833 | 0.866 | 0.529 |
| white helmet | 0.898 | 0.899 | 0.941 | 0.619 |
| yellow helmet | 0.891 | 0.803 | 0.884 | 0.516 |

---

YOLOv10b
- **Epochs:** 500  
- **Training time:** 3.12 h  
- **Parameters:** 20,420,404  
- **GFLOPs:** 98.0  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.921 | 0.836 | 0.907 | 0.574 |
| person | 0.953 | 0.835 | 0.932 | 0.595 |
| vest | 0.894 | 0.738 | 0.840 | 0.533 |
| blue helmet | 1.000 | 0.843 | 0.928 | 0.573 |
| red helmet | 0.801 | 0.875 | 0.897 | 0.568 |
| white helmet | 0.951 | 0.919 | 0.952 | 0.654 |
| yellow helmet | 0.929 | 0.807 | 0.892 | 0.523 |

---

YOLOv10l
- **Epochs:** 242  
- **Training time:** 2.21 h  
- **Parameters:** 25,725,620  
- **GFLOPs:** 126.4  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.878 | 0.849 | 0.903 | 0.553 |
| person | 0.869 | 0.873 | 0.914 | 0.576 |
| vest | 0.819 | 0.772 | 0.846 | 0.516 |
| blue helmet | 0.954 | 0.847 | 0.933 | 0.538 |
| red helmet | 0.789 | 0.860 | 0.887 | 0.550 |
| white helmet | 0.919 | 0.917 | 0.946 | 0.648 |
| yellow helmet | 0.916 | 0.828 | 0.891 | 0.493 |

---

 YOLOv10x
- **Epochs:** 310  
- **Training time:** 3.19 h  
- **Parameters:** 31,595,636  
- **GFLOPs:** 169.8  

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|:------|-----------:|-------:|-------:|-----------:|
| all | 0.905 | 0.847 | 0.905 | 0.566 |
| person | 0.936 | 0.873 | 0.933 | 0.578 |
| vest | 0.877 | 0.789 | 0.858 | 0.538 |
| blue helmet | 0.951 | 0.795 | 0.917 | 0.546 |
| red helmet | 0.772 | 0.875 | 0.884 | 0.562 |
| white helmet | 0.943 | 0.919 | 0.938 | 0.659 |
| yellow helmet | 0.951 | 0.833 | 0.898 | 0.510 |

---

**Summary:** YOLOv10x achieved the best global accuracy (mAP@50–95 = 0.566) on the CHV dataset.
