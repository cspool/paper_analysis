# A Artifact Appendix

#### A.1 Abstract

We provide step-by-step instructions to reproduce the major experiments and results of FlashPS, validating major claims regarding performance improvements and image quality preservation.

To simplify reproducibility, we provide an off-the-shelf Docker image, jiangxiaoxiao/flashps, which includes all the dependencies and configurations required to run the experiments. This eliminates the need for complex environment setup.

#### A.2 Description & Requirements

A.2.1 How to access. GitHub: [https://github.com/Sylvia-](https://github.com/Sylvia-16/FlashPS)[16/FlashPS;](https://github.com/Sylvia-16/FlashPS) Zenodo: [https://zenodo.org/records/17176576.](https://zenodo.org/records/17176576)

A.2.2 Hardware dependencies. During artifact evaluation, we provided an AWS EC2 instance with 8 A10 GPUs to validate the following scripts.

A.2.3 Software dependencies. Please use the image template pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel as the base image, then configure the Conda environment within this image.

```
1 # Create conda env
2 conda create -n flashps python =3.10
3 conda activate flashps
4 # Git clone this repo
5 git clone https :// github . com / Sylvia -16/ FlashPS . git
6 pip install -r requirements . txt
7 # Install our customized diffusers package
8 cd diffusers && pip install -e .
```

