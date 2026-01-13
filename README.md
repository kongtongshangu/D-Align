# Low-Frequency Perspective on Visual Domain Prompt for Continual Test-Time Adaptation
![LVDP framework](img/LVDP.png)

## Overview
Continuous Test-Time Adaptation (CTTA) aims to adapt the pre-trained model to continuously evolving, unlabeled target domain. Recent advances in this field have demonstrated that visual domain prompts (VDPs) offer a novel solution by 
rerepresenting input data through learned image-level domain prompts. However, existing VDP methods primarily rely on teacher-student pseudo-labeling schemes to extract knowledge from unlabeled target domains. Under dynamic distribution shifts, 
such approaches often suffer from miscalibrated predictions and noisy pseudo-labels, resulting in error accumulation and catastrophic forgetting. To address these challenges, we propose Low-Frequency Perspective on VisualDomain Prompt (LVDP) 
for CTTA. Specifically, we introduce a low-frequency-guided sample selection strategy that filters out samples dominated by high-frequency components, guided by posterior prediction discrepancy. To further en
rich the target distribution and reduce the model’s reliance on high-frequency cues, we employ a cross-class mixed frequency augmentation method that selectively injects high frequency components from other classes while preserving
low-frequency content. In addition, we alleviate catastrophic forgetting by incorporating low-rank and high-rank branches into the model and dynamically allocating domain knowledge between them based on prediction discrepancy. Extensive 
experiments on four widely used benchmarks demonstrate the effectiveness of LVDP in both classification and segmentation CTTA tasks.

## Installation
Please create and activate the following conda envrionment.
```bash  
# It may take several minutes for conda to solve the environment
cd LVDP/Classification
conda update conda
conda env create -f environment.yml
conda activate lvdp
```

## Data links
ImageNet-C [Download](https://zenodo.org/records/2235448#.Yj2RO_co_mF)

## Classification Experiments
### ImageNet-to-ImageNetC task
We release the code of the baseline method based on vit.
Our source LVDP model is [here](https://drive.google.com/file/d/1lJ3q0B5Y1f2L5pg8TFGgFFyXh9GbvCV6/view?usp=drive_link)
```bash
cd imagenet-c
bash ./bash/source.sh # Source model directly test on target domain
bash ./bash/lvdp.sh # LVDP
```

### Cifar10-to-Cifar10C task
Please load the source model from [here](https://drive.google.com/file/d/1b2zkX3eO_Z7RR-EZnUQ2YDGG7pqJM_bX/view?usp=drive_link)

Our source LVDP model is [here](https://drive.google.com/file/d/1gY5fCkefjD1gedVXQN-IZ2o_VOIoNtOz/view?usp=drive_link)
```bash
cd cifar-c
bash ./bash/cifar10c/source.sh # Source model directly test on target domain
bash ./bash/cifar10c/lvdp.sh # LVDP
```

### Cifar100-to-Cifar100C task
Please load the source model from [here](https://drive.google.com/file/d/1vgl_zGc0tM4_pyYZQiZfr3BZiaRfdspp/view?usp=drive_link)

And our source LVDP model is [here](https://drive.google.com/file/d/1I0mpOdALxZ6MCJWd79otWLCTRs71jLK6/view?usp=drive_link)
```bash
cd cifar-c
bash ./bash/cifar10c/source.sh # Source model directly test on target domain
bash ./bash/cifar10c/lvdp.sh # LVDP
```

## Segmentation Experiments
Please note that the Conda virtual environment required for the segmentation task is different from that required for the classification task, and a new one needs to be created.
```bash  
# It may take several minutes for conda to solve the environment
cd LVDP
conda update conda
conda env create -f environment.yml
conda activate lvdp
```
Our source LVDP model is [here](https://drive.google.com/file/d/1QxzHJ7ZHAn5ytcAXhNfw5yWXaQ9V0qlL/view?usp=drive_link)
```bash
cd LVDP/Segmentation
bash ./bash/lvdp.sh
```

## Acknowledgement
- ViDA code is heavily used. [official](https://github.com/Yangsenqiao/vida)
- Robustbench [official](https://github.com/RobustBench/robustbench)
- SVDP code is heavily used. [official](https://github.com/Anonymous-012/SVDP)
- SegFormer [official](https://github.com/NVlabs/SegFormer)
