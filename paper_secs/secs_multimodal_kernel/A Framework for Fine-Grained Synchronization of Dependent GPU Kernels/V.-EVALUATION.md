# V. EVALUATION

<span id="page-7-0"></span>We now evaluate the performance of cuSync against stateof-the-art baselines using large open source ML models as our workloads.

## *A. Experimental Setup*

We run our experiments on a machine with a 2.60GHz 12-core Intel Xeon E5-2690 CPU with 448GB RAM and 8 NVIDIA Tesla V100 32GB GPUs connected with NVLINK. We use CUDA 12.2 and report the average time of 20 executions after a warmup of 5 executions.

ML Models We used cuSync to synchronize CUDA kernels of four ML models: MegatronLM GPT-3 145 Billion [\[12\]](#page-12-0), LLaMA 65.2 Billion [\[15\]](#page-12-1), ResNet-38 [\[6\]](#page-11-1), and VGG-19 [\[13\]](#page-12-2). We used the GeMM and Conv2D CUDA kernels of NVIDIA CUTLASS 3.1 (Figure [2b\)](#page-2-0). We fuse the pointwise computations with GeMM and Conv2D kernels and developed a fused kernel of Softmax and Dropout in the Attention. We evaluate the reduction in inference times of these models using cuSync on batch sizes from 1 to the largest supported batch size by each model.

Baselines We consider the following baselines: StreamSync is the CUDA stream synchronization. Stream-K [\[10\]](#page-11-0) partitions the last thread block wave of GeMM and Conv2D among all SMs to improve the GPU utilization.

## *B. Ease of Programming*

Table [III](#page-7-1) shows that the number of lines added and changed to support fine-grained synchronization of GeMM, Conv2D, and Softmax-Dropout kernels using cuSync are negligible compared to the lines of code of these kernels. Thus, the cuSync approach enables diverse synchronizations of tile based computation kernels through few modifications.

## *C. Applicability in ML Models Inference*

We now discuss the applicability of cuSync in improving the performance of ML models from the perspective of kind of computations and the average utilization of GPU. First, ML models majorly consists of tile based GPU kernels, such as GeMM and Conv2Ds. Since cuSync supports any tile based kernel, we can use cuSync to synchronize kernels of ML models. Second, since the number of waves of each kernel increases with the batch size, the average utilization of all waves also increases. However, each ML model supports a maximum batch size limit during both training and inference phases. For example, the maximum token length supported by GPT-3 and LLaMA is 2048. We show in our experiments that even for this maximum batch size, GPU kernels suffer from low number of waves leading to low average utilization. In summary, cuSync is applicable to diverse ML models because ML models largely contains tile based kernels and the maximum batch size supported by ML models still suffers from under-utilization.

