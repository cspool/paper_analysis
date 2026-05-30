# <span id="page-7-0"></span>5 EXPERIMENTS

## 5.1 General Setup

Models. We focus our experiments on the SwitchTransformer [\(Fedus et al.,](#page-10-0) [2022\)](#page-10-0) family of models. Our primary target is the very largest variant, c2048, with around 1.6 trillion parameters, but we also consider the comparatively small base128 (7B params) and large128 (26B params) versions for testing and ablations. We chose the SwitchTransformer family as it contains the largest publicly-available model, which also features a similar or higher number of training tokens to parameters ratio than potential alternatives like [Artetxe et al.](#page-10-0) [\(2022\)](#page-10-0). Further, those models are also among the most popular massive MoEs, with several implementations across frameworks [\(Wolf et al.,](#page-11-0) [2019;](#page-11-0) [Shazeer](#page-11-0) [et al.,](#page-11-0) [2018;](#page-11-0) [Google,](#page-10-0) [2023\)](#page-10-0).

Framework. As accessibility is a major goal of our work, we build our code-base around the PyTorch-backend of the highly popular HuggingFace [\(Wolf et al.,](#page-11-0) [2019\)](#page-11-0) framework, which brings a number of additional challenges. First, we find that the largest model variants require a handful of bugfixes, primarily configuration and model setup changes, in order to run properly. We suspect that this is because their enormous sizes have rendered extensive testing very difficult. Second, we observed a major inefficiency in the context of generative inference for models with a large number of experts: the HuggingFace implementation will perform several (empty) CUDA calls for potentially 1000s of experts to which no token is routed, accumulating large overheads. We modify the implementation (also for baselines) to skip such unnecessary calls, leading to > 10× speedup for large models. We apply all changes to the HuggingFace framework only dynamically at runtime, so that our code can be run directly with an official installation.

Datasets. SwitchTransformers have been trained for a Masked-Language-Modelling (MLM) objective [\(Raffel](#page-11-0) [et al.,](#page-11-0) [2020b\)](#page-11-0) on the C4 dataset [\(Raffel et al.,](#page-11-0) [2020a\)](#page-11-0). Similar to most works in the area of LLM quantization [\(Yao et al.,](#page-12-0) [2022;](#page-12-0) [Frantar et al.,](#page-10-0) [2022;](#page-10-0) [Dettmers & Zettlemoyer,](#page-10-0) [2022\)](#page-10-0), we focus on general *upstream* compression directly on this pretraining task/dataset combination. Consequently, our evaluation focuses on validation performance for C4/MLM, where we use the public reproduction of C4 on HuggingFace as well as their replication of the original masking procedure. Calibration data for compression is taken, in order, from the first two shards of the training set. For efficiency, we primarily evaluate on 128 samples (corresponding to the average loss over > 10K tokens, which is quite stable) from the first shard of the validation set, but we also perform some evaluations other datasets.

Hardware. All compression experiments, including those for the very largest models, can be performed in less than a day on a single NVIDIA A6000 with 48GB of GPU memory.

However, efficiently compressing trillion parameter models using a large number of calibration samples requires a few 100GBs of (CPU) RAM; the original 1.6T model itself also occupies > 3 TB disk storage.

## 5.2 Compression Results

Accuracy. We begin by quantizing all SwitchTransformer models to 2-bit and ternary precision, and evaluating their validation loss. Our default number of calibration samples is 10K for 128 experts and 160K for 2048, but we also consider using 0.5× and 2× as many samples. In addition to using our efficient QMoE framework discussed in Section [3,](#page-2-0) we also consider a standard round-to-nearest (RTN) baseline [\(Dettmers et al.,](#page-10-0) [2022\)](#page-10-0). We simulate the latter by fixing Hessians to the identity matrix, thus applying precisely the same quantization settings and evaluation protocol. Table 4 summarizes our results.

Perhaps surprisingly, vanilla rounding (RTN) does not lead to a complete model collapse even at ternary precision, emphasizing the high robustness of large MoEs to quantization. Nevertheless, the loss increases are quite significant for smaller models at 2-bit and far too large to be useful at ternary precision. In contrast, using data-dependent quantization, 2-bit is achievable at minimal loss (1.7% relative on c2048) and ternary at only a small increase (6.7% relative on c2048). This demonstrates not only the effectiveness of such advanced quantization methods in this context, but also shows that extremely low-bit compression is indeed practical for massive MoEs.

|           | base128 |      | large128 |      | c2048 |      |
|-----------|---------|------|----------|------|-------|------|
| method    | 2bit    | tern | 2bit     | tern | 2bit  | tern |
| BF16      |         | 1.73 |          | 1.55 |       | 1.18 |
| RTN       | 2.27    | 4.54 | 1.96     | 2.79 | 1.33  | 2.15 |
| QMoE 0.5x | 1.78    | 2.11 | 1.54     | 1.70 | 1.22  | 1.27 |
| QMoE 1.0x | 1.76    | 1.99 | 1.56     | 1.69 | 1.20  | 1.26 |
| QMoE 2.0x | 1.76    | 1.93 | 1.57     | 1.64 | 1.21  | 1.26 |

Table 4. Comparing C4 validation losses for 2-bit and ternary (tern) quantized SwitchTransformers. "QMoE 0.5x" indicates that only half of the default number of calibration samples are used.

Additionally, we conduct evaluations on Arxiv, GitHub, StackeEchange and Wikipedia data sampled from RedPajama [\(Computer,](#page-10-0) [2023\)](#page-10-0). Even though only < 0.01% of our C4 calibration data originates from those websites, the compressed model still preserves performance almost as well as on the core of the distribution (see Table [5\)](#page-8-0).

In terms of calibration data, we see that increasing the amount of samples generally improves performance slightly, most noticeably for ternary quantization, but there is also some noise in the process, especially at 2-bit.

<span id="page-8-0"></span>

| bits  | arxiv | github | stackexch. | wiki |
|-------|-------|--------|------------|------|
| BF16  | 1.31  | 0.99   | 1.15       | 1.20 |
| 2-bit | 1.34  | 1.05   | 1.17       | 1.24 |
| tern  | 1.42  | 1.13   | 1.22       | 1.32 |

Table 5. Additional evaluations for the c2048 model.

Compression. Next, we investigate the actual compression rates that are achieved by further compressing ternary models using our scheme introduced in Section [4.](#page-4-0) We consider both compression relative to just the MoE modules (the model parts we quantize) as well as to the full model and all its metadata. The compression rates and overall checkpoint sizes are listed in Table 6.

| model    | moe-only | full   | bf16 | size [GB]<br>ours |
|----------|----------|--------|------|-------------------|
| base128  | 17.06×   | 11.76× | 14.9 | 1.27              |
| large128 | 18.34×   | 13.32× | 52.7 | 3.96              |
| c2048    | 20.07×   | 19.81× | 3142 | 158.6             |

Table 6. Compression rates and sizes for ternary models.

In general, measuring only relative to parts we compress (moe-only), all sizes achieve > 16× compression rate and thus < 1 bits per parameter storage. On c2048, even the overall rate, including all uncompressed dense layers, remains at 19.81×, corresponding to *0.807 bits per parameter*, reducing the checkpoint size from 3142GB to 158.6GB. One can also observe that compression rates increase with model size, which is for two reasons: (a) natural sparsity increases while our encoding dictionary is also optimized for c2048 (see Section [4\)](#page-4-0), and (b) weight distributions become closer to independent for larger layer sizes.

Runtime. Finally, we evaluate how long it takes to produce compressed models on a single A6000 GPU, for different amounts of calibration data. The results are shown in Table 7. Smaller models can be compressed in less than an hour and even c2048 in less than a day, confirming the high efficiency of QMoE. The runtime increase from large128 to c2048 is roughly proportional to the difference in size, despite the latter using 16× more samples. This is because the number of samples per expert stays constant and the expert size increases only slightly. Finally, we note that simply (iteratively) loading the original 1.6T model into RAM takes close to 5 hours on our slow disk storage.

| model    | 5K/80K  | 10K/160K | 20K/320K |
|----------|---------|----------|----------|
| base128  | 8.4min  | 14.0min  | 21.6min  |
| large128 | 22.0min | 30.2min  | 45.2min  |
| c2048    | 13.3h   | 16.0h    | 20.8h    |

Table 7. Compression runtime for different calibration data size.

## 5.3 Runtime Results

Individual Layers. Our kernel performance evaluation starts with a direct (isolated) comparison of our compressed matrix-vector product kernels (see Section [4\)](#page-4-0) against Py-Torch's standard (uncompressed) bfloat16 cuBLAS kernels. Figure [5](#page-9-0) (Left) shows the time taken by our compressed kernels relative to bfloat16, for the matrix shapes found in our MoEs, on two different GPUs. While our kernels have to perform a lot less slow (global) memory reads than the bfloat16 baseline due to lower storage costs, they need to spend much more compute for complex unpacking of the heavily-compressed weights. Nevertheless, executing our compressed kernels takes less time than the close to ideal bfloat16 baseline in all cases, with up to 35% speedup on specific matrix shapes. We note that these are very lowlatency operations, with the smallest matrix taking < 0.02 milliseconds and the largest < 0.05.

End-to-End Execution. Finally, we also benchmark our kernels end-to-end in HuggingFace on the real weights of our compressed MoE models. We consider an individual user application, like [\(Frantar et al.,](#page-10-0) [2022;](#page-10-0) [Leviathan et al.,](#page-11-0) [2023;](#page-11-0) [Park et al.,](#page-11-0) [2022\)](#page-11-0), where a single prompt (sampled from C4) should be processed to generate a 128-token response. As actually running the bfloat16 version of the c2048 model would require > 65 A6000 and > 130 3090 GPUs (versus 4 and 8, respectively, for sub-1-bit compressed weights) we have to estimate its runtime. We do this by having all experts in a layer point to the same weight data (resolving memory issues), which allows us to collect timings with precisely the same overheads as for our compressed models. However, this is a highly optimistic estimate since real execution would require close to 20× more GPUs, with corresponding communication overheads, and our numbers should thus be viewed as a lower bound.

The results, shown in Figure [5](#page-9-0) (Right), demonstrate that end-to-end execution of compressed models is only < 5% slower than standard (uncompressed) execution. This slight slow-down despite faster per-layer timings is due to the fact that the encoder may sometimes route multiple tokens to the same expert. Our current implementation naively executes a separate matrix-vector product for each token, while the baseline performs a much more efficient joint matrix multiplication. For applications where this is a significant bottleneck, one could easily introduce an inner loop over tokens into our kernel (Listing [1,](#page-6-0) line 30), or fully decompress first, followed by a standard matmul, for large token counts.

