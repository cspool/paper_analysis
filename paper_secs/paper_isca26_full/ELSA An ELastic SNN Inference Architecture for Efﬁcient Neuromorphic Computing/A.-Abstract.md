# *A. Abstract*

Our artificial evaluation has two major parts: the evaluation of the SNN model accuracy and the performance of the ELSA.

We evaluate our results using SNN models on standard image classification tasks. The evaluation encompasses five representative models: VGG16, ResNet18, ResNet34, ResNet50, and ViT-Small, and three widely-used datasets: CIFAR-10, CIFAR-100, and ImageNet. To facilitate reproducibility, we provide validation scripts and pre-trained checkpoints for all models, allowing rapid accuracy verification.

For assessing ELSA performance, we adopt a two-path evaluation strategy: a slow path and a fast path. In the slow path, the simulator generates energy and latency tracer files, from which power, performance, and area (PPA) metrics are computed. This process requires approximately 8 hours. In the fast path, we provide the pre-generated tracer files, allowing direct computation of PPA metrics within one minute.

All experiments are conducted on an Ubuntu server equipped with eight NVIDIA RTX 4090 GPUs, ensuring consistent and high-performance evaluation across all models.

