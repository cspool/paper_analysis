## Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是 RESONATOR，一个面向 MLLM 推理服务的共生式 Serving 运行时（构建于 SGLang 之上），沿两条轴做细粒度运行时调度：(1) **Intra-GPU Sharing Engine**——在单 GPU 上管理视觉编码器与 LLM 之间的 SM/HBM 共享，双模式调度：complementary 场景（LLM chunk 为 memory-bound 的 decode-heavy 阶段，Tag(c)=mem 且 decode token 占比 ρ≥ρ0）用显式 SM 分区（给 decode 保留 SM_dec=⌈SM_total·SM_dec_min(c)⌉ 的 SM 切片，其余给 encoder）；contending 场景（compute-bound 的 prefill-heavy chunk）用 per-kernel stream binding（Alg.1：把 compute-bound kernel 路由到可占全部 SM 的 wide stream、memory-bound/低占用 kernel 路由到窄 SM 子集的 narrow stream，wide/narrow 流由 green-ctx 或 libsmctrl 绑定 SM 子集，运行期只查 kernel profile 表选流）。两场景带迟滞切换。(2) **Inter-GPU Parallelism Engine**——跨 GPU 动态选择 encoder 的 DP/TP 并行方案：PRISM 调度算法把请求队列建模为 Multiple-Choice Knapsack Problem（最大化 ∑1/T(R_i,k)，DP 递推 dp[i][j]=max(dp[i-1][j], max_k{dp[i-1][j-k]+1/T(R_i,k)})，回溯出最优 batch 与各请求 TP 度），配合 **logical sharding** 实现近零开销并行切换（启动时每 GPU 预载完整未分片的 encoder 权重，运行时只改 cuBLAS/CUTLASS 的 leading dimension(ld) 参数做 strided GEMM 逻辑分片，TP 切换从数据面 weight 重分布降为控制面元数据更新）。
  - 实验比较：①端到端对比 vLLM、SGLang（text-only 强 baseline）与 EPD-Serve（MLLM 专用 encoder-prefill-decode 分池系统），三个模型、递增请求率（RPS）下比吞吐、mean/P99 TTFT、mean TPOT、mean E2E latency；②并行策略 landscape 分析（8 GPU、8DP/8TP/4DP-2TP/2DP-4TP 四固定策略 × 3 分辨率 336/1024/2048 × 3 RPS 1/4/8，验证无单一静态策略最优）；③Intra-GPU 消融（SM Partitioning Only vs Stream-based Sharing Only vs 完整引擎，归一化到 Partitioning；另测 encoder-decode 共存下 TPOT P99 窗口内 SLO 违例率）；④高异构 batch 案例研究（20 请求、4 种分辨率混合，动态调度 vs 静态策略 vs Oracle）；⑤系统消融（Static Baseline / +Intra-GPU Sharing / 完整 RESONATOR）；⑥与 EPD-Serve 对比（RESONATOR 4×A100 vs EPD-Serve 6×A100，省 33% GPU）；⑦logical sharding 计算效率微基准（3 类 encoder GEMM：QKV/FFN-up/FFN-down，L_seq∈{1k,4k,8k,16k}、TP∈{1,2,4}，contiguous shard vs strided logical shard 比延迟与 MFU，中位差 0.7%、91% 配置 <2%）。
  - 主要结果：相对 SGLang/vLLM，mean TTFT 最高 5.1×、TPOT 最高 3.0×、mean E2E 最高 4.9×、吞吐最高 3.4×（如 Qwen2-VL-7B 吞吐 876 vs 462 vs 257 tokens/s；Kimi-VL-16B@10RPS mean TTFT 11.6s vs 43.5/59.7s）；相对 EPD-Serve 用少 33% GPU 仍提升 TTFT 2.31×、E2E 1.58×、TPOT 1.75×；消融下 Intra-GPU 引擎把 TPOT@2RPS 从 155ms 降到 60ms、完整系统到 42.7ms，Inter-GPU 引擎带来 13.7× TTFT@4RPS 提升；共存场景 TPOT P99 SLO 违例率从 stream sharing 的 20%（21/103 窗口、峰值 28s）降到 5%（5/100 窗口、峰值 479ms）。
- 硬件平台是什么，配置是什么。
  - 单服务器 8× NVIDIA A100 SXM 80GB，GPU 间 NVLink 互联，CPU 为 Intel Xeon Gold 6430。LLM backbone 用 TP=4 on 4 A100（Qwen2-VL-7B、Kimi-VL-16B）、TP=8 on 8 A100（Qwen2-VL-72B），RESONATOR 与 baseline 用相同 LLM 并行度，仅 encoder 并行度运行时动态调整。微基准中 A100 FP16 峰值按 312 TFLOPS 归一化 MFU。
- 开源Serving框架是什么。修改了什么。
  - 基于开源 Serving 框架 SGLang-0.4.7（论文明确给出版本号）。修改/新增：①新增强化 chunked-prefill 的 LLM chunk 运行路径（chunk 特征向量 c=(n_p,n_d,L_c)，L_c 为平均 KV cache 深度 bucket）；②新增 Performance Atlas 离线 profiler 与在线查询接口（encoder 多项式模型 T_enc(r,k,SM_enc) 以 L_seq=⌈H(r)W(r)/P²⌉ 为唯一复杂度参数、LLM 随机森林模型 T_llm(c,SM_llm)，存储合法 TP 集 K(r)、decode SM 最小配额 SM_dec_min、memory/compute 标签与 kernel profile 表 P）；③Intra-GPU Sharing Engine：CUDA 流级 SM 配额控制（wide/narrow stream + SMCTRL.SetQuota），SM 分区在 chunk 边界切换（兼容 CUDA Graph 重放的 decode 路径），contending 路径用 eager 执行逐 kernel 选流；④Inter-GPU Parallelism Engine：encoder 请求批量形成 + PRISM DP 调度器 + logical sharding（strided GEMM 按 ld 参数分片，权重全量预载每 GPU）实现零开销 DP/TP 切换。调度查询只读 Performance Atlas（offline 一次 profiling 约 6 小时，Qwen2-VL-7B 上 profiled 范围平均预测误差 4.7%、外推 8.1%）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确说明 RESONATOR 是否开源或给出仓库链接（正文只引用 SGLang https://github.com/sgl-project/sglang 与 libsmctrl [26] 论文）。可复用组件：SGLang-0.4.7（开源 Serving 框架）、libsmctrl（开源，论文 [26] Bakita & Anderson "Hardware compute partitioning on NVIDIA GPUs" 提供 CUDA 流→SM 子集绑定）、green-ctx（论文提到的另一 SM 绑定机制）、cuBLAS/CUTLASS strided GEMM（ld 参数，CUTLASS 仓库 https://github.com/NVIDIA/cutlass）。SGLang 的 LLM serving 使用方式：requests 经 RadixAttention/Scheduler 进入 chunked-prefill 执行，prefill/decode 以 chunk 形式调度到 GPU；RESONATOR 在其上叠加 encoder 与 LLM 的共享与 encoder 的动态并行。
  - 使用例子（一个高分辨率图像请求到达 8×A100 集群，SGLang/RESONATOR 框架输入到硬件执行全过程）：
    ```
    # 输入：图像请求 R（如 2048×2048）+ 文本 prompt，Poisson 到达，batch 排队
    # 1) Preprocessor（CPU）：resize/归一化/分 tile → 生成 tile 序列 → 序列长 L_seq=⌈H·W/P²⌉
    # 2) Inter-GPU Parallelism Engine（控制面）：从队列 Q 取 m 个请求，
    #    对每个 R_i 查 Atlas Λ：合法 TP 集 K_i（按显存容量过滤）+ 各 k 的延迟 T(R_i,k)，
    #    PRISM 跑 DP dp[i][j]=max(skip, max_{k≤j}{dp[i-1][j-k]+1/T(R_i,k)})，
    #    回溯得到最优 batch 及每请求 TP 度（高分辨率→TP 大、低分辨率→DP 多）
    # 3) Logical Sharding（零开销切换）：每个 GPU 已预载完整 encoder 权重；
    #    对 TP=k 的请求，worker 用 cuBLAS/CUTLASS strided GEMM（改 ld 参数）只算自己的 1/k 逻辑分片，
    #    控制面更新 launch 参数即可，无需搬运/重分片权重
    # 4) Intra-GPU Sharing Engine（数据面，单 GPU 内）：encoder kernel 与 LLM chunk 共跑——
    #    若 LLM chunk 为 decode-heavy/memory-bound（Tag=mem 且 ρ≥ρ0）：SM 分区，
    #    decode kernel 固定跑 SM_dec 切片（TPOT 保护），encoder 用其余 SM；
    #    若为 prefill-heavy/compute-bound：每 kernel 查 profile 表 P 选流，
    #    compute-bound kernel → wide stream（全部 SM），memory-bound/低占用 kernel → narrow stream（q_narrow·SM）
    # 5) GPU 执行：A100 上 encoder 的 ViT 自注意力/FFN GEMM（quadratic 计算、低 HBM）与
    #    LLM prefill（chunked-prefill，FlashAttention 类 kernel）填满对方 SM/HBM 空洞；
    #    decode 阶段在 SM_dec 切片上连续跑，TP 通信等待间隙的 SM 被回收给 co-located 任务
    # 6) 输出：逐请求 TTFT/TPOT 与系统吞吐、E2E latency 日志（MMMUPro/TextVQA trace）
    ```
    作用：把 MLLM 的 vision encoder 从干扰源变成 LLM 的合作者——单 GPU 内按阶段互补性与 kernel 级空洞做 SM/HBM 细粒度共享，跨 GPU 按请求分辨率与并发度动态选 DP/TP，同时解决 encoder 进入 prefill 关键路径造成的 SLO 违例与静态并行度带来的过/欠供给，用相同 GPU 预算获得最高 5.1× TTFT、3.0× TPOT、4.9× E2E、3.4× 吞吐提升。
