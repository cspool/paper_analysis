## ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

- baseline方法是什么？
  Baseline是现有lossless compression方法（DFloat11/Huffman、DietGPU/rANS、nvCOMP/rANS）集成到decoupled inference pipeline中。全栈执行例子：算法层使用Huffman或ANS entropy coding对BF16 weight exponent field进行变长编码→编译框架层无特殊修改，使用标准NVCC编译→kernel调度层decompression kernel作为独立pre-processing stage执行，先完整解压权重到global memory buffer（Bitstream Partitioning→Symbol Extraction→Pointer Advancement三阶段），再启动标准cuBLAS_TC GEMM kernel读取解压后的权重→硬件架构层NVIDIA RTX4090/L40S GPU，解压阶段受限于variable-length bitstream的串行解码，SIMT warp内线程diverge，仅达到43.7%（DietGPU/ANS）至76.5%（DFloat11/Huffman）peak memory bandwidth。Baseline缺陷：(1) kernel-level mismatch：变长编码的data-dependent decoding与GPU SIMT lockstep执行模型冲突，导致control-flow divergence和compute underutilization；(2) system-level mismatch：decoupled pipeline的intermediate global memory buffer导致redundant memory traffic，decode阶段compute intensity相比标准GEMM下降62.0%（batch size 32, M=K=4096），使decode在memory-bound regime下性能严重退化；(3) 综合效果：DietGPU/nvCOMP/DFloat11仅达到cuBLAS的0.17×/0.19×/0.28×（RTX4090），压缩带来的memory saving被解压开销完全抵消。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出ZipServ，第一个hardware-aware lossless compression framework实现LLM推理加速。两个核心创新：(1) TCA-TBE fixed-length bitmap编码替代变长entropy coding——利用BF16 exponent的contiguous top-7分布（>95%权重覆盖，99.6%矩阵exponent连续），将每个8×8 tile编码为三个64-bit bitmap（每bit-plane独立），解压只需bitwise OR + POPC + integer ADD，完全消除分支和控制流divergence；(2) ZipGEMM fused decompression-GEMM kernel——将解压和GEMM融合为单一CUDA kernel，权重在register file内解压后直接送入Tensor Core mma指令，消除intermediate global memory buffer。

  全栈执行例子：
  - 算法层：TCA-TBE offline compressor分析每层weight exponent histogram→选择top-7连续exponent→按8×8 FragTile编码为3×64-bit bitmap + PackedSignMantissa (8-bit sign+mantissa) + FullValue fallback buffer。3-bit codeword (001-111) 通过base_exp + codeword算术implicit lookup恢复exponent。平均11.3 bits/element，~1.41×压缩率。
  - Serving框架层：集成vLLM (PyBind11)，stage-aware strategy：decode用fused ZipGEMM（load-compressed, compute-decompressed），prefill用decoupled decompression+cuBLAS_TC（摊销<4% overhead）。压缩权重减少的GPU memory自动分配给KV cache（LLaMA3.1-8B上KV cache从5.07GB→8.60GB, 1.70×），提升batch size和context length。
  - 编译框架层：论文未修改编译框架。ZipGEMM通过nvcc编译为.so，使用内联PTX指令（mma.m16n8k16、LDGSTS.128、LDSM.M88、cp.async）实现低层级硬件控制。
  - kernel调度层：ZipGEMM采用split-K tiling + 两级software pipeline。Coarse-level：tile double buffering，cp.async + __syncthreads() barrier重叠global→shared memory传输与计算。Fine-level：slice-wise interleaving，Tensor Core执行slice i时ALU并发load+decompress slice i+1到register。TCA-TBE的3层tiling（8×8 FT → 16×16 TT → 64×64 BT）直接对齐Tensor Core operand register layout (Ra0–Ra3)，消除runtime坐标变换。Decompressor的spatial bitmap indicator (bitwise OR) + dynamic addressing (POPC prefix sum) + arithmetic exponent reassembly (base_exp + codeword) 全程使用GPU native integer/popcount/shuffle指令，无分支、无shared memory table lookup、无bank conflict（仅~4.7K vs DietGPU百万级）。
  - 硬件架构层：NVIDIA RTX4090/L40S/RTX5090 GPU。用29.3% DRAM read reduction换取ALU增加（LOP3/IADD/POPC），但两级pipeline隐藏decode latency，Tensor Core利用率保持cuBLAS的71.6%。在RTX4090上peak 1.71×、L40S上peak 2.21× kernel speedup over cuBLAS。端到端平均1.22× over vLLM，首次证明lossless compression可以同时提供storage savings和LLM inference acceleration。

  解决Baseline缺陷的映射：
  (1) Kernel-level mismatch → TCA-TBE的fixed-length triple bitmap layout + warp-synchronous bitwise decoding消除divergence，Decompressor全程register-resident操作。
  (2) System-level mismatch → ZipGEMM的fused design将Compute Intensity从decoupled的严重退化提升至约50%高于标准GEMM，将bandwidth saving转化为wall-clock speedup。
  (3) 整体效果 → ZipGEMM是唯一超越cuBLAS_TC的压缩kernel（DietGPU/nvCOMP/DFloat11在RTX4090上仅0.17–0.28×）。stage-aware strategy根据prefill/decode自动切换执行模式保证全阶段性能最优。
