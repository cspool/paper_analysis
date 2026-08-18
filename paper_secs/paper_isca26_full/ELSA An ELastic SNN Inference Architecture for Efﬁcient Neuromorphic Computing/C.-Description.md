# *C. Description*

- *1) How to access:* We archive the source code at [https://](https://zenodo.org/records/19449728) [zenodo.org/records/19449728.](https://zenodo.org/records/19449728) We recommend you access the provided anonymous GitHub repository: [https://github.com/](https://github.com/Intelligent-Computing-Research-Group/ELSA#) [Intelligent-Computing-Research-Group/ELSA#](https://github.com/Intelligent-Computing-Research-Group/ELSA#) for the latest version.
- *2) Hardware dependencies:* We evaluate the SNN models with two types of server configuration: ① ELSA simulator: a server equipped with an AMD EPYC 9334 32-Core Processor. ② ELSA algorithm evaluator: a server equipped with eight NVIDIA 4090 GPUs.

- *3) Software dependencies:* The experiments rely on the following software components.
  - Ubuntu 22.04.3 LTS
  - Python 3.10
  - PyTorch 2.4.1
  - Anaconda 24.5.0
  - GCC 11.4.0
  - CUDA 12.2
- *4) Data sets and models:* The evaluated image classification models with the ImageNet dataset [\[47\]](#page-15-5), CIFAR10 dataset, and CIDAR100 dataset. We evaluate the SNN models including VGG-16 [\[42\]](#page-15-0), ResNet-18 [\[43\]](#page-15-1),ResNet-34 [\[43\]](#page-15-1), ResNet-50 [\[43\]](#page-15-1), and ViT (vision transformer) [\[74\]](#page-16-0).

