## DietGPU（GPU ANS 熵编解码库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Meta（Jeff Johnson，FAISS 作者）开源的 GPU 端 ANS（rANS）熵编解码库（MIT license，github.com/facebookresearch/dietgpu）：byte-oriented rANS codec + float 编解码扩展（float16/bfloat16 无损），A100 上 ANS 吞吐约 250–410 GB/s、float codec 250–600 GB/s；以 4 KiB segment 为压缩单元、batch-oriented API（C++ device pointer + PyTorch tensor）。设计目标：在 PCIe/NVLink/网络传输前压缩数据（牺牲一点压缩率换速度），用于分布式 collective（all-to-all/all-gather/reduce-scatter/all-reduce）加速——首个公开 GPU ANS 实现，GPU 版 FSE。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文的用法：把 DietGPU 的 ANS kernel 作为解压后端，扩展为 tile 粒度解码 + tensor-core GEMM 集成。DietGPU 原设计面向数据搬运（segment 批量压缩/解压，无 tile 随机访问、无 GEMM 融合）；论文保留其 warp-cooperative 解码结构（每 lane 一条 substream、coalesced 重归一化），在其上加：tile 对齐 substream 划分、shared memory 直写、与 GEMM 生产者-消费者融合（naive 基线即"DietGPU ANS decode + CUTLASS GEMM 分离两段"）。第三方评测：H100 上 DietGPU ANS 解码约 592 GB/s，接近论文目标的权重输入速率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：rANS 状态 + per-symbol 频率表 + segment 并行（batch 处理 4 KiB 段）；C++/PyTorch 双 API。使用：GPU 集群数据搬运压缩（训练 all-gather 前压缩可提速 ~10% wall-clock）、LLM 权重无损压缩的基座（本论文、ZipServ 等）。局限：无随机访问/tile 语义——需上层扩展（本论文的 offset 表 + tile substream）。

ENEC 补充视角（Ascend NPU 侧对比）：ENEC 论文把 DietGPU 作为 GPU 侧主要 baseline（Diet_ANS 与 Diet_Float 两种模式）。ENEC 对 DietGPU 的评价：其 Diet_Float 模式虽然做了指数-尾数分离（只压指数），但指数压缩仍依赖 ANS 变长编码——在 Ascend NPU 上因不规则访存与控制流而效率极低（Ascend AIV 无条件分支、无 gather）。ENEC 用"分支无关整数变换 + 定长位打包"替代 DietGPU 的变长 ANS 编码，结果是：压缩吞吐 3.43× 高于 DietGPU、压缩比 1.12× 优于 nvCOMP；Diet_Float 在 BF16 上的压缩比（1.47-1.48）仍略高于 ENEC（1.35-1.37），ENEC 以少量压缩率换取 NPU 端 2 个数量级吞吐（BF16 压缩 372 GB/s vs ZipNN 0.4 GB/s 级）。跨平台对照（Table VII）：ENEC-GPU-V1（CUB 前缀和 + warp 内建）在 A800 达 419.2/421.0 GB/s，接近 Diet-Float 的 271.9/271.3 GB/s 两倍。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
