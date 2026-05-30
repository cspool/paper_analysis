# B.2.2 Benchmarking Across Input Lengths (Section [4.3\)](#page-6-2)

For Section 4.3, we evaluate the performance, speed, and report the compute of all the methods across different input lengths using the Llama-3.1-8B model and the RULER benchmark. We run 50 samples per task for performance evaluation, while for speed testing, we run 20 samples per task. For STARATTN, we set both the block size and the anchor length to n/H, where n is the input length and H is the sequence parallelism size. For APB, we use the hyperparameters listed in Table [8.](#page-13-1) We align other hyperparameters to the end-to-end benchmark. We visualize the compute of each method following Table [9.](#page-16-0)

#### B.2.3 Ablation Studies (Section [4.4\)](#page-6-3)

Here, we elaborate on the settings in Section [4.4.](#page-6-3)

Each Component's Contribution. We conduct the ablation study of each component using Llama-3.1-8B. We conduct experiments of different settings on E.MC from ∞Bench using the first 50 samples to assess the impact of different combinations on performance. We use the hyperparameters of n = 128K, l<sup>b</sup> = 32K, l<sup>a</sup> = 4K, l<sup>p</sup> = 2K.

Wall-time Breakdown Analysis. The prefill and decoding time is tested with Llama-3.1-8B on SG1 from RULER. For the breakdown analysis of the prefill time, we run all the methods on synthetic random input. All the experiments are conducted under a 128K input length, and the hyperparameters are kept the same as in previous experiments.

Distributed Settings. We test the performance under different host numbers (sequence parallelism size) with Llama-3.1-8B. We set H across {2, 4, 6, 8} and evaluate the performance of STARATTN and APB on E.MC from ∞Bench. We set the sequence length to 32K and 128K, and test on the first 50 samples.

#### B.3 System Environment

All training and evaluation are conducted on workstations equipped with 8× NVIDIA A800-80GB GPUs and 104 Intel® Xeon® Platinum 8470 CPUs, running CentOS Linux 7 (Core). The GPUs within each machine are interconnected using third-generation NVLink technology, while cross-machine communication is facilitated via HDR InfiniBand. We perform the training of the retaining head and all experiments—except for FLASHATTN and MINFERENCE tests—on 8 GPUs. The experiments for FLASHATTN and MINFER-ENCE are conducted on a single GPU.

