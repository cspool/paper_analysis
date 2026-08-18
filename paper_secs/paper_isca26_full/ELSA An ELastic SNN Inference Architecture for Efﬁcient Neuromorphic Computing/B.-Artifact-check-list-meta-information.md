# *B. Artifact check-list (meta-information)*

- Compilation: GCC: 11.4.0
- Model: VGG-16, ResNet18, ResNet34, ResNet50, and ViT Small.
- Data set: ImageNet, CIFAR10, CIFAR100.
- Run-time environment: Ubuntu 22.04.3 LTS, CUDA 12.2, and PyTorch 2.4.1.
- Hardware: A server with an AMD EPYC 9334 32-Core Processor and eight NVIDIA 4090 GPUs.
- Output: SNN accuracy, ELSA energy, performance, and area.
- How much disk space is required (approximately)?: 20GB.
- How much time is needed to prepare the workflow (approximately)?: It takes about 30 minutes to prepare the environment.
- How much time is needed to complete experiments (approximately)?: Obtaining ELSA PPA metrics requires approximately 8 hours, and evaluating the SNN model accuracy also takes around 8 hours. Using the pre-generated tracer files, the fast evaluation of ELSA PPA metrics can be completed in under one minute.
- Publicly available: Our framework is publicly available on GitHub [https://github.com/](https://github.com/Intelligent-Computing-Research-Group/ELSA#) [Intelligent-Computing-Research-Group/ELSA#.](https://github.com/Intelligent-Computing-Research-Group/ELSA#)
- Data licenses: The datasets are publicly available through their original licensing terms.
- Archived: [https://zenodo.org/records/19449728.](https://zenodo.org/records/19449728)

