# CaroNet-Dynamic-2.0
Implementation of "Human‒machine interaction based on real-time explainable deep learning for higher accurate grading of carotid stenosis from transverse B-mode scan videos." in Pytorch -- for model inference.

### Our work has been accepted by European Journal of Radiology (EJR) 2023! 

#### Our human‒machine interactive online demo is available! [[Demo](https://xinruizhou-caronet-dynamic-v2.hf.space)]

## Method Workflow
![Workflow](https://github.com/user-attachments/assets/d96445ca-3217-4ca3-90ca-5575fbd307a2)

## Usage

### 1. Prepare testing video data.

<details>
  
  <summary>Directory structure for data.</summary>
  
  ```
    data/
  
    ├── video01/
    │   ├── 1.png
    │   ├── 2.png
    │   └── x.png
    ├── video02/
    │   ├── 1.png
    │   ├── 2.png
    │   └── y.png
    └── ...
  ```
</details>

### 2. Inference (detector -> segmentor -> key fragment extractor -> classifier).

```
$ python inference.py
```

# Acknowledgments
Our code is based on [YOLOv5](https://github.com/ultralytics/yolov5) and [Conformer](https://ieeexplore.ieee.org/document/10040235). We appreciate the authors for their great work.

