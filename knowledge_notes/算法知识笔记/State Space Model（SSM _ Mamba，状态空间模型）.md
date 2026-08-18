## State Space Model（SSM / Mamba，状态空间模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
State Space Model（状态空间模型）是用线性递归状态 $h_t = \bar{A} h_{t-1} + \bar{B} x_t$、$y_t = C h_t$ 编码序列历史的模型族。S4（Gu et al., 2022）引入结构化参数化（HiPPO 初始化 + 对角 A 矩阵），实现 O(N) 复杂度长序列建模；Mamba（Gu & Dao, 2023）进一步做选择性 SSM——离散化步长 Δ 与输入投影 B、C 由输入动态生成（$B_t=W_B x_t$, $C_t=W_C x_t$, $\Delta_t=\mathrm{Softplus}(W_\Delta x_t)$），用 selective scan 替代卷积，保留 RNN 式递推但具备输入依赖的"选择/遗忘"能力；Mamba-2（Dao & Gu, 2024）用 SSD（Structured State Space Duality）统一 SSM 与线性注意力，支持 chunk 并行。与 Transformer attention（O(N²) 计算、O(N) KV cache）相比，SSM 是 O(N) 计算、O(1) 固定大小 state（per-layer hidden state），推理时无需 KV cache，内存占用与序列长度无关。本论文（Rearchitecting the Datacenter Lifecycle for AI）用 Mamba-2.8B vs Llama3-3B 的跨代 GPU 实验论证"模型架构决定硬件兼容性"：2K 序列 TP1 下 Llama3 在 V100 上比 H200 慢 7.7×，而 Mamba 仅慢 3.6×——state-space 架构与旧/弱 GPU 更兼容，延长旧硬件生命周期。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba 单层的推理 pipeline（对照 attention）：
```
输入 x_t ∈ R^d（第 t 个 token 激活）
# 1. 输入投影
Δ_t = Softplus(W_Δ x_t);  B_t = W_B x_t;  C_t = W_C x_t
# 2. 选择性离散化（ZOH）
Ā_t = exp(Δ_t A);  B̄_t = (Δ_t A)^{-1}(exp(Δ_t A) − I)·Δ_t B_t
# 3. 递归状态更新（decode 逐 token）
h_t = Ā_t h_{t-1} + B̄_t x_t          # O(d_state × d) 矩阵-向量，state 固定大小
# 4. 输出
y_t = C_t h_t                          # + output projection 到词表
训练/长 prefill 用 selective scan 并行；推理 decode 只做 3 步的矩阵-向量
→ 无 KV cache 增长、无 attention 的 O(N) 内存；每步工作量恒定
```
论文的硬件观察：decode 本就是 memory-bound 低算术强度，SSM 因省去 KV cache 的加载与 attention 计算，在内存带宽更弱的旧 GPU（V100）上退化远小于 transformer——这正是"架构选择影响硬件寿命"的证据：若 fleet 以 SSM 为主，旧 GPU 的可服务年限显著延长，刷新节奏可放缓。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：Mamba 官方代码（https://github.com/state-spaces/mamba）、Mamba-2 与 Jamba（AI21，Mamba+Transformer 混合）、NVIDIA Mamba-2-Hybrid 系列（4 attention + 24 SSM + 28 MLP 层）；推理运行时（vLLM 等）已支持 Mamba 系模型，decode 用逐 token 递推（state 常驻寄存器/片上）、prefill 用并行 scan 或 chunk-scan kernel（TileLang 对比 Triton 的 chunk-scan 平均 1.77×、chunk-state 2.10× 加速）。论文用 Mamba-2.8B 在 T4/V100/A100/H100/H200 上跑 vLLM 测 TTFT/TBT（2K 序列、TP1、batch 8，按 H200 归一化），结论并入 TCO 框架的 workload 模型（模型架构参数决定 roofline 的算术强度与内存占用），供刷新策略判断"未来模型若转向 SSM，旧硬件仍具竞争力"。

涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI
