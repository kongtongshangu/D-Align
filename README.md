# Continuous Test-Time Adaptation via Dual Alignment
![LVDP framework](img/framework.png)

## Overview
Continual test-time adaptation (CTTA) seeks to adapt a source-pretrained model to continuously shifting target distributions. Recent advances in this field either minimize prediction entropy on streaming target samples, which is efficient but prone to error accumulation, or rely on teacher–student pseudo-labeling, which provides more stable predictions at the cost of high computational overhead. To alleviate these issues, in this paper, we propose Dual Alignment (D-Align), a method that jointly optimizes correlation alignment and masked consistency alignment for stable and efficient online adaptation. Specifically, correlation alignment constructs a reliable pseudo-source domain by selecting resilient target samples, maintaining their diversity via a dynamic memory bank, and aligning nonlinear feature relations through image-level visual domain prompt–driven alignment, enabling effective feature alignment without costly backpropagation. Masked consistency alignment enforces prediction consistency across multiple masked views of each target sample, promoting robust and discriminative target representations. Using multiple standard benchmarks, the experiments have verified the effectiveness of D-Align under dynamic domain shifts, in comparison with state-of-the-art methods. 

## Installation
Please create and activate the following conda envrionment.
```bash  
# It may take several minutes for conda to solve the environment
cd D-Align/imagenet
conda update conda
conda env create -f environment.yml
conda activate d_align
```

## Data links
ImageNet-C [Download](https://zenodo.org/records/2235448#.Yj2RO_co_mF)

## ImageNet-to-ImageNetC task
```bash
cd imagenet-c
bash ./bash/d_align.sh # D-Align
```

## Acknowledgement
- ViDA code is heavily used. [official](https://github.com/Yangsenqiao/vida)
- Robustbench [official](https://github.com/RobustBench/robustbench)
