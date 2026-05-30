## QMoE Sub-1-Bit Compression of Trillion-Parameter Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QMoE 设计了一个自定义 CUDA kernel（Sub1MatVec）用于 fused decompression + matrix-vector product，将压缩存储的三元权重以字典解码方式 on-the-fly 转换为可计算值。核心设计要点：(1) **Warp-per-Row 并行**——每个 warp (32 threads) 处理权重矩阵的一行，每行独立编码，使用 28/32 threads 进行解码和乘加累加；(2) **Dictionary-Based Decoding**——2^16 个 UINT16 codewords 映射到最多 14 对三元权重（28 weights），字典 512KB 存储于 GPU L2 cache，高频 codeword 通过概率排序实现 L1 cache prefetch；(3) **Shared Memory Dequant Table**——三元值 {0, 1, 2} 通过复制 32× 的 shared memory lookup table deq[3][32×num_warps] 转换为 {0, w_min, w_max}，避免 bank conflict；(4) **Coalesced Memory Access**——每次取 32 个 UINT16 codewords 到 shared memory（单次 coalesced transaction），输入向量 x 预加载到 shared memory 实现快速连续读取；(5) **Ternary 解码优化**——每权重 2-bit 存储于 UINT32 中，通过 shift + mask 提取（无 modulo/division 等慢速操作），线程 0-13 处理前半权重、14-27 处理后半。
  - 实验比较：(1) Per-layer kernel 性能——Sub1MatVec vs PyTorch bfloat16 cuBLAS GEMV（各 MoE 层矩阵形状），A6000 和 RTX 3090 上 compressed kernel 在全部情况下比 uncompressed baseline 更快（最高 35% speedup）；(2) End-to-end 推理——压缩后的 c2048 在 4×A6000 和 8×3090 上的 HuggingFace 全流程 runtime，与理想化 uncompressed baseline（同一专家数据复用，估计值）对比，<5% 额外延迟开销。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A6000 (48GB) 和 NVIDIA RTX 3090 (24GB)。Per-layer kernel 评估：单 GPU 上各类 MoE 矩阵形状。Compressed 推理：4×A6000 或 8×3090（单服务器）。c2048 模型若不压缩需 >65 A6000 / >130 3090 GPU，因此 uncompressed baseline 用时通过"所有 expert 指向同一权重数据"的估计方式获得（下界估计，实际需更多 GPU 及通信开销）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 PyTorch + HuggingFace Transformers 的自研 CUDA kernel（Listing 1: Sub1MatVec）。修改内容：(1) 新增 CUDA kernel 实现 on-the-fly 字典解码 + dequant + matvec 融合计算；(2) One threadblock per SM，每 warp 处理一行，超过 32 行时 warp 串行处理多行；(3) HuggingFace 推理框架的 MoE 层调用被替换为压缩 kernel；(4) HuggingFace 中空 CUDA kernel launch 的 bugfix（跳过无 token 分配的 expert 调用，>10× 加速大模型推理）。评估指标：per-layer latency (ms)、end-to-end latency per token (ms)、speedup ratio。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/ISTDASLab/qmoe。CUDA kernel 源码（含完整边界条件处理）见官方仓库。
  - Kernel 评估原理与执行流程：
    1. **Kernel Launch Configuration**：每个 threadblock 处理一个 weight matrix block（含多行），每 warp 处理一行。`num_warps = min(rows_in_block, 32)`，若 rows > 32 则部分 warp 串行处理多行。1 threadblock per SM 避免 wave quantization 效应。
    2. **Input Preparation**：压缩权重 w_comp (UINT16 array, 每行独立编码的 codeword 序列)、row_off (每行 codeword 偏移索引)、dec (UINT32[2^16 * 2] 字典表，512KB)、ter_minmax (每行的 {w_min, w_max} dequant 参数)、x (bfloat16 input vector)。
    3. **Shared Memory 初始化** (lines 7-17)：
       - 所有 warp 协作：将 x 向量加载到 `x_shared[w_width]`（bf16→float 转换）。
       - 每 warp 独立：构建 dequant lookup table `deq[3][32*num_warps]`。`deq[0]=0, deq[1]=w_min, deq[2]=w_max`，复制 32× 在列方向避免 bank conflict。
    4. **Per-Row Decoding Loop** (lines 22-33)：
       - (a) Coalesced load: `w_comp_block[warp][lane] = w_comp[i + lane]` — 32 threads 联合加载 32 个 UINT16 codewords。
       - (b) 仅 lanes 0-27 (28 threads) 参与解码：遍历 32 个 codewords，每个 codeword 去字典查表 `dec[2*enc + (lane/14)]` → 线程 0-13 取第一个 UINT32、线程 14-27 取第二个。
       - (c) Ternary 提取：`ter = (wx14 >> (4 + 2*(lane%14))) & 0x3` — 每 weight 仅需 shift + mask（硬件友好），无 modulo/division。
       - (d) Dequant + FMA：`res += deq[ter][thread] * x_shared[idx + lane]` — 连续 shared memory 读（无 bank conflict）。
       - (e) 偏移更新：`idx += 2 * (wx14 & 0xf)` — pair_count 存于低 4 bits。
    5. **Warp Reduction** (lines 37-38)：对 28 threads 的部分积进行 warp shuffle 求和。
    6. **Output**：`y[row] = float2bfloat16(res)` 写入全局内存。
    7. **性能原理**：字典 512KB 适合 GPU L2 cache（A6000 L2=6MB, 3090 L2=6MB），高频 codeword 因概率排序被 L1 自动 prefetch。Compressed kernel 读取更少的 global memory（<1 bit/param vs 16 bits/param），虽增加 bit unpacking 计算，但 global memory latency（~200 cycles）远大于 bit ops（~1 cycle），净效果为加速。
