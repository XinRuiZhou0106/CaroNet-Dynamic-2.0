# CaroNet-Dynamic-2.0
Implementation of "Human‒machine interaction based on real-time explainable deep learning for higher accurate grading of carotid stenosis from transverse B-mode scan videos." in Pytorch -- for model inference.

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
