# D. Other Hardware Components

Frontend and Writeback. The *frontend* module comprises two components: the *issue logic*, responsible for fetching and generating instructions, and the *data logic*, which loads input data from either DRAM or SRAM depending on the issued instruction. The issue logic retrieves loop parameters from the instruction, specifying the repetition of the instruction and how to update the metadata used by the data logic in each iteration. This metadata determines the column and row addresses of D-SRAM accessed by the data logic. Column addresses are selected as a range, specified by the start index and the number of columns to access. The row addresses for selected columns are generated from one of multiple predefined patterns, such as diagonal or strided access, also determined by the metadata.

The *writeback* module resembles the standalone version of the data logic in the *frontend* module, but with a different set of access patterns necessary for SRAM write operations. When storing the output of each CCU, all elements of the output vector are written to different banks in the D-SRAM, avoiding bank conflicts and issues of data consistency.

**Data Manipulation Unit (DMU).** As illustrated in Fig. 6(a), each PE within VGA consists of an *upper* DMU and a

lower DMU. DMUs remove the need for I/O-heavy data formatting operations by utilizing predefined control networks and a shifter. They perform bit-reversal permutation, padding insertion/removal and broadcast operations on the input data of the next module. For the *upper* DMU, the next module is either CCUs or the DMA engine, while for *lower* DMU, it is the *writeback* module.

**DMA Engine.** Each PE has a DMA engine with an I/O width of 64B. The frontend sends DMA instructions to the DMA engine, triggering access to the host memory system. These DMA instructions contain the offset from the base address, while the base addresses are stored inside the DMA engine. As the DMA engine and the *core* share hardware units for accessing D-SRAM, overlapping DMA with computation may cause structural hazards. To alleviate this issue, DMA is scheduled during Projection and Update modes, which have long regions where only the D-SRAM read or write path is active. If a DMA operation lasts longer than these regions, the *frontend* pauses the DMA engine, completes the current computation mode, and then restarts the remaining DMA operations afterward. In such cases, the DMA time is exposed to latency, as shown in Fig. 9. Since there are more data to read from the host accelerator's memory than to write, DMA read (D-SRAM write) time is more likely to be exposed to latency.

## E. System-level Issues

**Instruction Set Architecture.** VGA uses two types of 16B instructions: configuration and executable instructions. Configuration instructions set up TLBs and the addresses required for DMA operations, and alter the global states of the CCU. Executable instructions encode the behavior of each module in a PE. They include information such as D-SRAM addresses, read/write access patterns of input/output data, DMU operations, and CCU modes. These instructions do not include data, as data is passed through separate vector registers.

VGA Initialization. Before the first execution on VGA, the host accelerator must perform initialization. First, it allocates memory regions for data such as Q, K, V, output matrix, instructions, and SSM parameters. The size of each memory region is set to cover sequence lengths up to a predetermined maximum. Next, the page table entries (PTEs) for these regions are retrieved and stored in a reserved memory region that can be directly accessed by VGA. Triggered by a memorymapped register, the DMA engine in each PE populates its TLBs with these PTEs. Each DMA engine can hold up to 32 TLB entries, which covers up to 64MB per PE when using a 2MB page on GPUs. With 128 PEs, VGA can access up to an 8GB memory space. This capacity is sufficient to support sequence lengths of up to 1M, assuming a hidden dimension of h = 768, as the size of weight matrices used in the ROI region is small compared to conventional LLMs. The cost of initialization can be amortized, as initialized memory regions can be reused during inference.

Communication with Host Accelerator. As VGA and the host processor share the main memory, it is used as a ren-

dezvous point for all GPU/TPU and VGA communication, without concurrent memory access by either. GPUs/TPUs use a write-through policy for data pages to send to VGA. When VGA writes its output to the main memory, the GPU/TPU cache invalidates any stale cached copy of the output.

Offloading Workload. The host accelerator offloads the ROI of the H3 layer by sending the input length and layer number to VGA through memory-mapped registers. This information is broadcast to all PEs, triggering the execution of VGA instructions. Each PE signals the completion of computation to the host through its private completion flag register. The workload is parallelized across the hidden dimension (h). Each PE is assigned consecutive  $\frac{h}{\# \text{ of PEs}}$  hidden dimensions and processes input sequences of one hidden dimension at a time. This eliminates the need for synchronization between PEs, as computation on each hidden dimension is independent and PEs write to non-overlapping memory addresses.

## V. EVALUATION

## A. Methodology

**Models and Datasets.** The performance evaluation is conducted on the inference of two models. The first model is a GPT-125M like natural language model taken from the official H3 GitHub repository [18]. This model, referred to as H3-GPT, replaces 12 self-attention blocks in GPT-125M with H3 blocks. Retaining the same hyperparameters as the original repository, H3 blocks use a hidden dimension of h=768 and the SSM state vector dimension of m=64. We evaluate the model on input queries from the WikiText-103 dataset parsed to sequence lengths ranging from 8K to 128K. The second model, denoted as H3-Speech [24], is composed of 6 unidirectional H3 layers and targets raw speech classification on SC10 [50]. Its H3 layers use a configuration of h=128 and m=64. The lack of FFN layers in this model presents opportunities for additional performance gains for VGA.

The evaluation employs the maximum possible power-of-two batch size on each device. Specifically, for H3-GPT and H3-Speech on GPU, batch sizes of 8 and 16 are employed, respectively. On TPU, batch sizes of 2 and 16 are utilized for H3-GPT and H3-Speech, respectively. Note that VGA conducts the *exact* computation of ROI without any approximation. Therefore, the use of VGA does not incur any accuracy drop, and its inference latency is independent of the input data contents.

**GPU Baseline.** All GPU baselines are measured using a single NVIDIA A100-40GB GPU. The self-attention baseline utilizes FlashAttention2 [13], the state-of-the-art conventional self-attention mechanism on GPU. The official GitHub repository of H3 [18] does not provide the implementation of the state passing algorithm. Thus, we faithfully implemented the state passing algorithm without Vandermonde matrix generation, using a custom FFT convolution CUDA kernel from the repository, utilizing it as the baseline for H3.

**TPU Baseline.** In our TPU evaluation, we use a single core of the TPUv3 chip, which is half of a TPUv3 chip.

![](_page_8_Figure_8.jpeg)

Fig. 10. ROI speedup of two models on GPU and TPU, with respect to the execution of H3 on each platform.

A TPU core is the basic unit of computation for TPUv3. Although sharing a single die, two cores have completely separate DRAM and compute units, and can access each other through an on-chip interconnect router [40]. As there is no publicly available implementation of FlashAttention for TPU, we employ the multi-head attention layer provided by TensorFlow2. However, since this implementation is not as memory-efficient as FlashAttention2, evaluating sequences longer than 8K causes an out-of-memory (OOM) error. For the H3 software baseline, we used a ported version of H3 with the state passing algorithm.

VGA Simulation. For the evaluation of VGA, we simulate the ROI latency using a custom cycle-accurate simulator. The custom simulator is integrated with Ramulator2 [36] to simulate DRAM access latency. This simulation includes both the execution of ROI and the data transfer time between VGA and DRAM. To calculate the inference latency when utilizing VGA, we replaced the ROI latency of the GPU/TPU with that of VGA. In Ramulator2, we configure five 1.2GHz chips (1555GB/s) and two 0.9GHz chips (450GB/s), all each with an 8GB HBM2 chip, for the GPU and TPU, respectively, to match the known DRAM bandwidth of the host accelerator [2], [31]. VGA uses k = 32 CCUs in the core of each PE. In order to conduct 2D-FFT as a square matrix, the length of each input and output chunk l is set to 2048. The number of PEs is selected by analyzing the ROI latency across different numbers of PEs, as shown in Fig. 13. For evaluation, we employ VGA with 128 PEs for the GPU, and with 32 PEs for the TPU.

**Area/Power Comparison.** The RTL implementation of VGA is synthesized using TSMC 40nm technology at a frequency of 1GHz. To compare the area and power consumption of VGA with the GPU and TPU, we scaled VGA down to 7nm [2] and 16nm [30] using the scaling factor equation [46] to match the technology of each accelerator. The area and TDP of a single A100 40GB GPU are set to 826  $mm^2$  and 400 W [2], while for a single TPUv3 core, we used half the area and power of a single chip which is 324  $mm^2$  and 225 W [31].

![](_page_9_Figure_0.jpeg)

Fig. 11. (a) ROI latency of VGA and GPU. (b) Per-operation speedups of VGA over GPU. H3-GPT with a sequence length of 128K is used for latency measurement

## B. ROI Speedups

Fig. 10 illustrates the relative ROI latency across platforms and models as the sequence length increases. All values are normalized to the latency of the H3 layer with state passing executed on each platform. For FlashAttention2 and self-attention used in GPU and TPU, respectively, we measure the time spent on the matrix multiplication of Q and  $K^{\rm T}$ , the softmax operation to acquire the score matrix S, and the multiplication of S and V.

VGA on GPU. With an increase in input sequence length from 8K to 128K, VGA achieves a 4.89× and 4.63× ROI latency speedup at the 128K sequence length for H3-GPT and H3-Speech models, respectively. Compared to FlashAttention2 on the H3-GPT model, VGA attains a maximum 48.2× speedup at the 128K sequence length. The speedup in ROI gradually increases and then levels off as the sequence length increases. While the latency of the GPU scales relatively linearly with the sequence length, VGA exhibits sub-linear increases in latency at shorter sequence lengths, resulting in an increase in speedup. This is because, in VGA, the latency for loading parameters of the next hidden dimension is noticeable at shorter sequence lengths but becomes negligible at longer sequence lengths.

VGA on TPU. On a TPU, VGA achieves  $28.1 \times$  and  $28.7 \times$  speedup for the ROI latency at the 128K sequence length in the H3-GPT and H3-Speech models, respectively. In the H3-GPT model, self-attention encounters an OOM for sequences longer than 8K. The H3-GPT model exhibits a different trend for ROI speedup as the sequence length increases. Since the batch size is selected for inputs with a sequence length of 128K, only the longer sequence inputs fully utilize the compute resources on the TPU, resulting in a greater ROI speedup at shorter sequence lengths.

VGA ROI Breakdown. A breakdown of the accelerator's ROI latency is as follows: The FFTConv, Output Projection, and State Update operations consume 47%, 21% and 24%, respectively. In the FFTConv operation, FFT/IFFT operations account for 69%, while CTF generation and multiplication with FFT results take 25%. Pointwise multiplication of the filter with FFT results occupies the remaining 6%. In both the Output Projection and State Update operations, the majority of time (100% and 94%, respectively) is spent on matrix-vector multiplications, with minimal time for pointwise operations. VGA maintains an average FLOPS utilization of 78%.

![](_page_9_Figure_7.jpeg)

Fig. 12. (a) End-to-end speedup of two models across platforms, normalized to the execution time of H3 on each platform. (b) Portion of ROI in two models across different platforms.

Fig. 11 shows the latency breakdown of H3-GPT with a 128K input sequence length for the the GPU and VGA. FFTconv, state passing, pointwise add and mult are  $3.85 \times$ ,  $2.93 \times$ ,  $65 \times$ , and  $56 \times$  faster in VGA than the GPU. Pointwise operations benefit the most from kernel fusion due to their small amount of compute.

## C. Model Speedups

Fig. 12(a) shows the end-to-end latency of two models across each platform. The latency is normalized with the latency of the H3 model on each platform. The models with FlashAttention2 and self-attention use the same configuration as the H3-GPT model. When the ROI is delegated to the accelerator, the H3-GPT model achieves  $1.7 \times$  and  $7.4 \times$  latency speedup over software at a sequence length of 128K on the GPU and TPU, respectively. In comparison to the model using FlashAttention2 with identical configurations, the combination of H3 block and VGA achieves 8.8× speedup at a sequence length of 128K on the GPU. For the H3-Speech model, there are  $2.4\times$  and  $14.9\times$  latency speedups over software at a sequence length of 128K on the GPU and TPU, respectively. ROI Portion of TPU/GPU. Comparing GPUs to TPUs, the portion of ROI is larger for TPU than for GPU in both models, as seen in Fig. 12(b). This distinction arises from their different architectural focus. GPUs offer greater flexibility, supporting a wide range of operations beyond dense matrix multiplications. In contrast, TPUs prioritize compute-intensive dense matrix multiplications, with less emphasis on operations requiring irregular memory access patterns like FFT or memory-bound operations, such as the pointwise operations within the ROI. Therefore, by offloading the ROI to VGA, TPUs can experience greater performance improvements than GPUs in terms of the model latency.

## D. Source of Efficiency

**Memory Traffic Reduction.** The memory traffic reduction is the key to the latency improvement of VGA. Compared to the H3-GPT model on a GPU with a sequence length of 128K, VGA achieves a memory traffic reduction of 9.7×. This reduction enables greater arithmetic intensity and consequently higher achievable throughput. It is attributed to the fully fused ROI computation, made feasible by the reconfigurability and the on-the-fly Vandermonde matrix generation.

![](_page_10_Figure_0.jpeg)

Fig. 13. (a) ROI speedup of the H3-GPT model at a sequence length of 128K across varying numbers of PEs used in VGA, normalized to the ROI latency of H3 on each platform. (b) ROI speedup of the H3-GPT/Speech model at a sequence length of 128K across different batch sizes, normalized to the ROI latency of H3 on each platform.

**SRAM Footprint Reduction.** VGA improves area and power efficiency by minimizing the SRAM footprint for the fully fused kernel through on-the-fly matrix generation. Only 5 rows and columns are needed for generating the entire  $\mathbf{M}_{xy}$  and  $\mathbf{M}_{ux}$  matrices, respectively, with full pipeline efficiency. This reduction in memory footprint results in a 410× decrease compared to using full matrices. Overall, when also considering memory space for input and filter vectors, the total SRAM capacity is reduced by  $5 \times$  compared to storing full matrices. VGA without Matrix Generation. In a hypothetical scenario where the PE design of VGA is altered to store full matrices in SRAM and utilizes CCUs with greater flexibility, the throughput for CTFGen, Projection, and Update modes can be doubled. This enhanced throughput can translate into a 1.4× increase in the ROI speedup over the current PE design, when not accounting for the increased DMA time. Nevertheless, implementing such a configuration is less efficient compared to scaling the number of current PEs in VGA, as this approach incurs a 2.9× increase in area and a 3.9× increase in power due to a larger SRAM requirement.

## E. Sensitivity Studies

Fig. 13(a) illustrates the ROI speedup of the H3-GPT model using VGA, compared to the H3 implementation on different platforms, considering varying numbers of PEs within VGA. As the number of PEs is increased from 16 to 256, the ROI speedup increases from  $0.6 \times$  to  $8.7 \times$  when VGA is used with a GPU. For integration with a GPU, VGA with 128 PEs is chosen, as the linear scaling behavior is lost beyond this point.

In the case of TPU, the latency improvement increases from 14.1× to 57.4×. For TPU usage, VGA with 32 PEs is employed since the end-to-end model speedup improvement diminishes noticeably beyond 32 PEs. The non-linear scaling behavior observed when VGA is used with GPU or TPU is attributed to the DRAM bandwidth constraint.

Fig. 13(b) shows the influence of the batch size on the accelerator's speedup. Overall, ROI speedup remains consistent across batch sizes on both GPU and TPU for both models. The ROI speedup of H3-Speech on the GPU exhibits higher values at small batch sizes but plateaus as the batch size increases.

TABLE I Area and Power Breakdown

| Single PE               |                         |              |                |                         |              |  |  |  |
|-------------------------|-------------------------|--------------|----------------|-------------------------|--------------|--|--|--|
| Components              | Area (mm <sup>2</sup> ) | Power<br>(W) | Components     | Area (mm <sup>2</sup> ) | Power<br>(W) |  |  |  |
| D-SRAM                  | 2.61                    | 1.25         | I-SRAM         | 0.05                    | 0.05         |  |  |  |
| Core<br>(32 CCUs)       | 2.35                    | 0.27         | Others         | 0.14                    | 0.07         |  |  |  |
| DMA Engine              | 0.08                    | 0.01         | Total (40-nm)  | 5.54                    | 1.72         |  |  |  |
| Frontend &<br>Writeback | 0.03                    | 0.01         | Scaled (16-nm) | 1.32                    | 0.59         |  |  |  |
| DMUs                    | 0.28                    | 0.06         | Scaled (7-nm)  | 0.41                    | 0.32         |  |  |  |
| VGA                     |                         |              |                |                         |              |  |  |  |
| 32 PEs (16-nm)          | 42.35                   | 18.92        | 128 PEs (7-nm) | 52.82                   | 41.10        |  |  |  |

TABLE II
SPEEDUP OF CONVOLUTION RELATIVE TO PARALLEL SCAN IN SSM

| SeqLen (tokens) |      |      |      |      |      |
|-----------------|------|------|------|------|------|
| Speedup (times) | 3.38 | 3.37 | 3.19 | 3.23 | 3.34 |

## F. Area/Power Analysis

As shown in Table I, the total area and power consumption of VGA with 128 PEs, scaled to 7nm, amount to 52.82  $mm^2$  and 41.11 W, approximately 6.4% and 10.28% of a single A100 GPU, respectively. Meanwhile, for VGA with 32 PEs scaled to 16nm, the total area and power consumption are 42.35  $mm^2$  and 18.92 W, making up 13.1% and 8.4% of a single TPUv3 core. Further detailed breakdowns for each PE are available in the same table. The primary components occupying the area and power consumption are the SRAM and the core. Other elements, such as the DMA engine, shifters, and control networks, contribute negligibly to the overhead.

