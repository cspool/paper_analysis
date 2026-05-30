## Sparsing Law Towards Large Language Models with Greater Activation Sparsity

- baseline方法是什么？
  Baseline 是主流 SiLU 激活的密集 LLM（如 LLaMA 结构 + gated FFN）：使用 SiLU 激活函数、无激活稀疏度感知的架构设计、依赖固定宽深比的 Transformer 结构。在稀疏度度量方面，baseline 方法包括：(1) Straightforward ReLU——用零阈值判断弱贡献神经元（仅适用于 ReLU，无法泛化到 SiLU）；(2) Top-k——强制每层保持固定 k 个激活神经元（MoE 常用，牺牲灵活性和性能）；(3) FAT-ε——全局统一阈值忽略绝对激活值低于 ε 的神经元（无法层间自适应）。
  全栈执行例子（Baseline: SiLU LLaMA-like 0.8B 模型，GPU 推理）：
  - **算法pipeline层**：输入 token x ∈ R^{d_h} → gated FFN：s = SiLU(W^{gate}x)，FFN(x) = W^{out}[s ⊙ (W^{in}x)]。SiLU 激活函数产生大量非零但幅度可忽略的负输出值，而这些值在零阈值下不被视为"弱贡献"，导致稀疏度被严重低估。训练过程中，SiLU 模型激活比满足 A_SiLU(D) = -c/D^α + A_0（递增幂律），意味着更多训练数据反而降低稀疏度（激活比收敛到 ~40%）。以 0.1B SiLU 模型为例，极限激活比 A_0=40.9%，即最多只有 ~59.1% 的稀疏度。
  - **系统框架层**：使用 PowerInfer 或 llama.cpp（https://github.com/ggerganov/llama.cpp）进行推理。llama.cpp 进行密集 FFN 计算，即对每一层所有 d_f 个神经元计算完整的 W^{in}x、SiLU、W^{out} 输出求和，无法跳过弱贡献神经元。每个 token 经过所有层所有神经元，每 token 的 FLOPS 为常量。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：密集 FFN 计算使用标准 GEMM kernel（如 cuBLAS），无稀疏度感知的 kernel 优化。PowerInfer 虽有离线 profiler 和在线预测器来跳过弱贡献神经元，但 baseline 的 SiLU 模型稀疏度低（~60%），加速效果有限。
  - **硬件架构层**：NVIDIA A800 GPU（80GB），密集计算模式，无稀疏加速硬件支持。
  Baseline 的核心缺陷：(a) SiLU 激活函数的负输出使简单的零阈值度量失效，缺乏通用且精确的激活稀疏度评估指标；(b) 主流 SiLU LLM 的激活稀疏度随训练数据增加而降低（递增幂律），与高效推理的目标背道而驰；(c) 缺乏宽深比、参数规模等架构因素如何影响稀疏度的定量理解，无法指导稀疏 LLM 的架构设计。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法通过三个层面解决 baseline 缺陷：(1) **CETT-PPL-1% 通用稀疏度度量**——使用 CETT（累积尾部截断误差）自适应搜索每层阈值，识别弱贡献神经元集合 D = {i | ||n_i||_2 < ε}，通过控制 L2 范数相对误差而非原始激活值来泛化到任意激活函数（包括 SiLU）；引入 PPL 增加容忍度 1%，二分搜索 CETT 超参数，确保稀疏化后性能退化可忽略（表1验证 CETT-PPL-1% 平均性能退化仅 -0.16% C.R.、-0.30% R.C.）；(2) **激活稀疏度定量标度律**——发现 ReLU 模型满足递减对数空间幂律 A_ReLU(D) = exp(-cD^α + b) + A_0，即更多数据可降低激活比（提高稀疏度）；而 SiLU 模型满足递增幂律，更多数据反而损害稀疏度；(3) **架构指导原则**——确定宽深比的瓶颈效应（0.1B 模型~114）：低于瓶颈时激活比与宽深比线性正相关；建议选择确保训练稳定性的最小宽深比。发现极限激活比与参数规模弱相关（规模不敏感性）。
  全栈执行例子（论文方法：ReLU 2.4B μP Transformer，NVIDIA A800 GPU，800B tokens 训练）：
  - **算法pipeline层**：输入 token x ∈ R^{d_h} → gated FFN：s = ReLU(W^{gate}x)，FFN(x) = W^{out}[s ⊙ (W^{in}x)]。ReLU 激活函数天然产生大量零值（s_i=0 的神经元输出为零向量，对最终输出无贡献）。训练中使用 CETT-PPL-1% 度量监控稀疏度演化：对最后 5 个 checkpoint，二分搜索 [0,1] 内的 CETT 超参数，计算平均 PPL 比率 = exp(loss_sparse - loss_dense) 是否达到 1.01，确定统一 CETT 阈值后应用于全过程。ReLU 模型服从递减对数空间幂律，800B tokens 训练后极限激活比 A_0=6.48%（稀疏度 93.52%）。宽深比选择接近 0.1B-1.2B 实验模型的值（~48-56），确保在训练稳定性区间内。
  - **系统框架层**：推理时使用 PowerInfer（https://github.com/SJTU-IPADS/PowerInfer）——其离线 profiler 统计每层每个神经元的历史激活频率，在线预测器根据当前输入预测哪些神经元可能被激活。由于 ReLU 模型的高稀疏度（93.52%），对任意给定输入，仅约 6.48% 的神经元被激活（s_i > 0），PowerInfer 的预测器能高置信度跳过其余 93.52% 弱贡献神经元的计算，相比之下 SiLU 模型只能跳过 ~60%。llama.cpp 作为密集 baseline 无法利用此稀疏度。
  - **kernel调度层**：PowerInfer 根据预测的激活模式选择性地执行 GEMM：仅加载和计算被预测为"活跃"的神经元对应的 W_{i,:}^{gate}、W_{i,:}^{in}、W_{:,i}^{out} 行/列，跳过弱贡献神经元。对于 ReLU 模型密集版本（所有 d_f 个神经元都参与），每层 FLOPS 固定；稀疏版本仅计算 ~6.48% 的神经元，FLOPS 按比例减少。实测 2.4B 模型解码速度 41.79 tok/s（PowerInfer）vs 10.23 tok/s（llama.cpp 密集），4.1× 加速。
  - **编译框架层**：论文未明确说明。
  - **硬件架构层**：NVIDIA A800 GPU（80GB），104 CPUs。PowerInfer 编译为 CUDA 版本。CETT 二分搜索中，dense 和 sparse forward pass 均在 GPU 上执行，loss_dense 和 loss_sparse 在验证集 VS 上计算（每个 checkpoint 的每个 batch 执行一次 dense forward 和一次 sparse forward）。
  方法 vs Baseline 对比核心差异：(a) ReLU 替代 SiLU → 天然更高的稀疏度且数据越多越稀疏（递减对数空间幂律 vs 递增幂律）；(b) CETT-PPL-1% 替代零阈值/全局阈值 → 通用、精确、可泛化的稀疏度度量；(c) 小宽深比 + 大规模数据 → 有理论指导的稀疏 LLM 训练策略；(d) 规模不敏感性 → 小模型上的发现可直接推广到大模型。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Sparse Upcycling 通过将 dense checkpoint 的 MLP 层扩展为 MoE 层（每个 expert 初始化为原 MLP 的拷贝，router 随机初始化），以微小的额外 compute budget（~50% 原训练成本）将 dense 模型升级为容量大得多（参数增加 8-10×）但推理 FLOPs 相近（因为稀疏激活）的 MoE 模型。
  全栈执行例子（Sparse Upcycling: T5 Base → MoE，TPU v4，64 chips）：
  - **算法pipeline层**：Input token X → Self-Attention（复制自 dense）→ 若该层为 MoE 层（每间隔一层）：Router 计算 softmax(X @ W_r)，Expert Choice routing 让每个 expert 独立选择 C·T/E 个 token → 32 个 expert 各自执行 FFN（参数复制自原 dense MLP）→ Combine（加权合并 expert 输出）→ LayerNorm。非 MoE 层保持原 dense FFN。训练继续使用原 inverse sqrt LR schedule。
  - **系统框架层**：TPU v4 上使用 T5X 框架（https://github.com/google-research/t5x/tree/main/t5x/contrib/moe），数据并行 + expert 并行（将 32 experts 分布到多个 chip）+ XL 模型额外使用 model 并行（4 partitions）。Router 计算通信量小，expert 之间的 all-to-all token dispatch 是主要通信开销。
  - **编译框架/Kernel调度层**：论文未明确说明。
  - **硬件架构层**：TPU v4，使用 Adafactor optimizer，mixed precision 训练不变。
  Baseline 缺陷的对应解决：(a) **容量瓶颈**——Dense continuation 参数容量固定，Sparse Upcycling 通过 MoE 将参数量从 248M 扩展到 2.00B（T5 Base），expert 专业化学习使模型容量大幅提升；(b) **训练成本浪费**——MoE from scratch 浪费 dense checkpoint 训练投入，Sparse Upcycling 复用所有已有参数和 optimizer state（vision），仅需 ~50% 原训练成本即可超越 dense continuation；(c) **初始性能下降**——通过 Expert Choice routing + router weight normalization（vision）减小 model surgery 带来的初始性能损失，确保被至少一个 expert 选中的 token 保持与原始 dense 相同的输出；(d) **路由效率**——Expert Choice routing (C=2) 相比 Top-K routing 更快且 per-compute-time 性能更好，避免 token dropping 和 expert 负载不均衡。

- baseline方法是什么？
  Baseline 方法包括现有开源 MoE kernel 实现：(1) **ScatterMoE**（Triton, Tan et al. 2024）：forward gather fusion 仅实现 varlen-M（非 varlen-K），backward 需单独 gather kernel，dS=⟨dO,Y⟩ 需缓存 Y，无 MMA/IO 重叠，无 TMA 支持；(2) **MoMoE**（Triton, Costin et al. 2025）：类似 ScatterMoE，dS 虽融合于 up-proj act grad 但仍用 ⟨dO,Y⟩ 路径，scatter 操作较慢；(3) **MegaBlocks**（Gale et al. 2023）：block-sparse GEMM 方式，需单独 gather+pad+scatter kernel，总 IO 达 8TKd bytes；(4) **Megatron-LM GroupedMLP**：使用 CUTLASS Grouped GEMM 但无 gather fusion，假设输入已 contiguous-packed；(5) **DeepGEMM**（Zhao et al. 2025b）：高度优化的 SM90/SM100 BF16 Grouped GEMM，但仅支持 contiguous-packed 输入，无 gather fusion、无 epilogue fusion、无 MMA/IO 重叠。所有 baseline 的共同痛点：(a) 在细粒度 MoE（高 G=d/n）下 activation memory 随 G 线性增长，(b) IO cost 随 G 线性增长导致 memory-bound，(c) 稀疏 MoE 下 Grouped GEMM tile padding 浪费大量 FLOPs。
  全栈执行例子（Baseline: ScatterMoE，H100，7B MoE, n=256, K=8, E=64）：
  - **算法Pipeline层**：Forward 执行 up-proj group GEMM (gather X, GEMM, SwiGLU) → down-proj group GEMM (GEMM, scatter Y)。Backward 需要缓存 X, X_e, H, Y, S, π。dS = ⟨dO, Y⟩ 需要加载 dO 和 Y (2TKd bytes HBM 访问)，dH 通过 dSwiGLU(dY W_2^T, H) 从 H 重算。总 activation memory = 2Td + 4TKn + 2TKd ≈ 2Td + 4TKn + 2T×8×(d/256)×d bytes，随 G 线性增长。
  - **系统框架层**：PyTorch autograd 引擎管理前向/反向，Triton kernel 编译为 CUDA。FSDP-2 + ZeRO-3 分布式训练，使用 lm-engine 代码库。
  - **编译框架层**：Triton → MLIR → PTX → SASS。Triton 无法直接控制 TMA 异步操作和 warp-specialized 调度，限制了 Ping-Pong scheduling 和异步 IO 重叠的能力。
  - **Kernel调度层**：ScatterMoE forward up-proj gather+GEMM (Triton kernel, ~600 TFLOPS on H100) → forward down-proj GEMM+scatter (Triton kernel, st.global store, ~550 TFLOPS) → backward dH (单独 Triton kernel, ~300 TFLOPS) → backward dS kernel (单独 launch, 读取 dO+Y) → backward dW2 (Triton kernel, 无 gather fusion 需单独 gather kernel) → backward dX~ (Triton kernel) → backward dW1 (Triton kernel, 无 gather fusion) → backward dX aggregation。Triton kernel 无法异步 overlap MMA 与 IO，kernel 间有 CUDA stream bubble。
  - **硬件架构层**：H100 GPU SM (132 SMs)，每个 SM 有 4 warp schedulers、Tensor Core (WGMMA)、TMA engine、256KB SMEM。Triton 使用 TMA（ScatterMoE 基于旧版 Triton 不支持），cp.async 仅用于 gather fusion in forward。st.global 同步 store 阻塞下一 tile MMA 执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **SonicMoE**，一种硬件-模型架构协同设计方案，通过三个核心创新解决 baseline 痛点：
  **(1) Memory-efficient backward 算法**：重新设计计算图——dS=⟨dA',A⟩ 替代 dS=⟨dO,Y⟩（节省 2TKd bytes activation 和 2TKd bytes HBM 访问），dH 从 dA 和 H 通过 dSwiGLU 重算。Gather fusion 消除 X_e 和 dO_e 的 HBM 物化。结果：activation memory 恒定于 2Td+4TKn bytes，不随 G 增长。对 7B MoE (n=256) 减少 45% activation memory vs ScatterMoE。
  **(2) IO-aware kernel 设计**：三个层面减少/隐藏 IO——(a) Gather fusion with cp.async 在 varlen-M 和 varlen-K Grouped GEMM 中均实现（baseline 仅 forward varlen-M），在 Blackwell 上通过 relay warp + mbarrier cluster-scope 解决 2-CTA cluster 的 gather 同步问题；(b) Heavy epilogue fusion：SwiGLU/dSwiGLU/dS/A' 全部融合在 GEMM epilogue 中，用一个 kernel 产出多个输出；(c) MMA/IO 重叠：Hopper 上 Ping-Pong scheduling（2 consumer warpgroups 交替 MMA 和 epilogue），Blackwell 上利用 UMMA 单线程异步 + TMEM 2-stage 实现 MMA warp 与 epilogue warps 并发。异步 TMA store（非 st.global scatter）避免阻塞 MMA。
  **(3) Token Rounding (TR) 路由**：消除 Grouped GEMM tile padding 浪费——将 per-expert token 数舍入到 M_tile=128 倍数，每个 expert 最大偏离 TC 结果 1 tile。TR 在极稀疏 MoE (K/E≤1/64) 下额外提升 16% kernel TFLOPS 且不损失下游任务质量。
  全栈执行例子（SonicMoE，H100，7B MoE, n=256, K=8, E=64）：
  - **算法Pipeline层**：Forward 与 baseline 语义等价。Backward 重设计：dH kernel 同时产出 dH（by dSwiGLU(dA, H)）、dS（by ⟨dA', A⟩ reduce over n）、A'（by Broadcast(s)·A）。无 Y 缓存，activation memory 恒定为 2Td+4TKn bytes。TR 路由：top-K TC → 计算 f_e → nearest-round to 128 → EC 排序选择 padded/discarded tokens → 修改 π 和 S → 输入 MoE compute kernel（与路由算法解耦）。
  - **系统框架层**：CuTe-DSL (C++) 编写 kernel，PyTorch nn.Module 封装为 drop-in MoE 层。lm-engine 代码库管理 FSDP-2 训练循环。64 H100s 实现 213B tokens/day（与 ScatterMoE 96 H100s 的 225B 可比）。
  - **编译框架层**：CuTe-DSL → C++ → NVCC → PTX → SASS。CuTe-DSL 允许直接控制 TMA、cp.async、WGMMA/UMMA、warpgroup scheduling 的底层指令，实现 TileLang/Triton 无法表达的异步 IO/MMA 重叠。
  - **Kernel调度层**：8 个 CuTe kernel 流水线执行：(A kernel: cp.async gather + WGMMA + SwiGLU epilogue, ~650 TFLOPS) → (Y kernel: WGMMA + TMA store + Ping-Pong epilogue, ~600 TFLOPS) → (O kernel: TMA gather-and-sum, ~2.5 TB/s) → backward: (dH kernel: cp.async gather + WGMMA + heavy epilogue 含 dSwiGLU/dS/A' + 异步 TMA load H, Ping-Pong scheduling, ~450 TFLOPS) → (dW2 kernel: cp.async gather + WGMMA, ~500 TFLOPS) → (dX~ kernel: WGMMA + TMA store, ~580 TFLOPS) → (dW1 kernel: cp.async gather + WGMMA, ~520 TFLOPS) → (dX kernel: TMA gather-and-sum, ~2.5 TB/s)。Ping-Pong 重叠 MMA 与 IO 使得 dH kernel 的 heavy epilogue 不显著拖慢吞吐。
  - **硬件架构层**：H100 的 TMA 异步 copy（GMEM↔SMEM）与 Tensor Core WGMMA 通过 CUDA pipeline 异步并发。Ping-Pong: consumer warpgroup 0 执行 WGMMA 时，consumer warpgroup 1 执行上一 tile 的 epilogue（SwiGLU/TMA store），下一 tile 角色互换。Blackwell B300 的 TMEM (256KB/SM, 2-stage=2×128cols) + UMMA（单线程异步，无 RF 压力）让 epilogue warps 直接从 TMEM stage 读取 MMA 结果并执行 epilogue，与 MMA warp 写入另一 TMEM stage 完全并发。所有 scatter 操作替换为 gather-and-sum aggregation，使用 TMA gather 保持高 bandwidth。
