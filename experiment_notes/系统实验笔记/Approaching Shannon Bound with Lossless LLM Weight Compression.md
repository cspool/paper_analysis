## Approaching Shannon Bound with Lossless LLM Weight Compression

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：修改开源 serving 框架 SGLang，把其 dense 矩阵乘（投影层）默认的 CUTLASS GEMM kernel 替换为本文 ANS-enabled GEMM backend（plugin 式投影算子，离线压缩权重直接作为 drop-in 权重加载），使权重静态内存大幅缩小，在同一 GPU 显存预算内腾出容量给 KV-cache，从而支撑更大 batch、提升吞吐；Mixtral-176B 以专家并行（EP）部署于 4×A100。论文不改调度算法本身，而是通过压缩权重 footprint 扩大调度器可用的内存预算（batch 上限由显存决定）。
  - 实验比较：SGLang 默认（未压缩权重 + CUTLASS GEMM）vs 本文（压缩权重 + ANS GEMM backend），固定显存预算（Qwen-14B：80 GB 单卡；Mixtral-176B：320 GB 四卡），序列长度 1024/2048。指标：权重/KV/总内存分解、最大可行 batch size、吞吐 tokens/s、median TPOT。
- 硬件平台是什么，配置是什么。
  - 8× NVIDIA A100 80 GB（HBM2e 2 TB/s）服务器（SGLang 端到端）；NVIDIA Hopper H200 用于 kernel 级对比。PyTorch 2.5.1、CUDA 12.1，吞吐为 SGLang batching scheduler 下实测执行时间。
- 开源Serving框架是什么。修改了什么。
  - SGLang（论文引用 [50]，https://github.com/sgl-project/sglang）。修改点：dense 矩阵乘（W_Q/W_K/W_V/FFN 等投影算子）的默认 CUTLASS GEMM 替换为 fused rANS 解压 + GEMM 后端；权重以"tile 压缩 bitstream + 4B/tile offset 表 + 每层共享 codebook"形式加载。多 GPU：Mixtral-176B 用 EP 跨 4 张 A100。调度算法本身论文未说明有修改。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文自身 SGLang 集成代码未找到公开仓库（arXiv 2606.15789，无官方 repo）；SGLang/CUTLASS/DietGPU 均开源（链接见上）。
  - 全过程（一个 decode 请求）：SGLang 前端接收请求 → RadixAttention/调度器按剩余显存决定可加入的 batch 上限并组批（Qwen-14B 权重 27.5→18.1 GB，释放约 9.4 GB 显存给 KV-cache，batch 上限随之提高）→ 执行投影层时调用 ANS-enabled GEMM kernel（不再先整层解压）→ kernel 内 warp 0 从全局内存按 offset 表取压缩 tile、rANS 解码进 shared memory，其余 warp 用 tensor core 与激活 tile 做 GEMM，双缓冲流水重叠 → 输出激活、KV-cache 增长由调度器记账 → 返回 token。效果：Qwen-14B（seq 1024）最大 batch 47→60（Table II，1.3×；摘要与 C 节正文写作 47→75）、吞吐 1131→1217 tokens/s（1.1×）；seq 2048：23→30（1.3×）、548→651 tokens/s（1.2×）。Mixtral-176B（4×A100，seq 1024）batch 20→95（4.8×）、吞吐 241→391 tokens/s（1.6×）；seq 2048：10→47（4.7×）、190→257 tokens/s（1.4×）。median TPOT 因解压开销略有上升（Qwen-14B 71→81 ms、112→125 ms），以轻微延迟换吞吐。kernel 级对比（H200）：vs NeuZip 吞吐最高 ~10×、vs DFloat11 ~6–7×（摘要称最高 11×）。
