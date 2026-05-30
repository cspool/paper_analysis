## MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-GPS 修改了 MoE 多 GPU 推理 serving 流程，在每层 Attention 之前插入 predictor，根据预测结果动态调整 expert 在 GPU 间的分布（dynamic expert duplication）以均衡负载。整体 serving 流程修改如下：
    1. **Predictor 插入点**：在每层 Transformer block 的 Attention 计算之前，predictor 接收当前 batch 的 hidden states，输出 expert 分布预测。Predictor 可为 Distribution-Only（offline 估计）或 Token-to-Expert（在线推理）。
    2. **Expert Duplication**：基于预测的 token-to-expert 分布，使用 Algorithm 1（贪心算法）在 GPU 间动态复制热门 expert。迭代将 overloaded GPU（token > 1/G）上的 expert 复制到 underloaded GPU 直至均衡。
    3. **Token Dispatch**：Distribution-Only 使用 All-to-All Scatter（随机分发，通信开销不变）；Token-to-Expert 使用 Direct Routing（按预测结果直接路由到目标 GPU，跳过 Scatter 阶段，节省通信开销）。
    4. **Expert 移动开销隐藏**：Expert duplication 的 weight 传输（Mixtral 8×7B 单 expert ~47MB FP16）通过 NVLink（~0.1ms）可与 Attention 计算重叠，因此在 moderate batch size 下 latency 可隐藏。
  - 实验比较：(1) Baseline（无 prediction，EP-only FFN + TP Attention）vs Distribution-Only Prediction vs Token-to-Expert Prediction（多 accuracy 点）；(2) NVLink（2TB/s）vs PCIe（32GB/s）下的 prefill latency 对比；(3) 不同 skewness（1.2-2.0）下的 prediction strategy 最优选择；(4) 不同 interconnect bandwidth（600GB/s, 200GB/s, 64GB/s, 32GB/s）下的 savings difference（Distribution-Only minus Token-to-Expert）；(5) Mixtral / LLaMA-MoE / Switch Transformer 横评。

- 硬件平台是什么，配置是什么。
  - GPU：4× NVIDIA A100，fully connected
  - Interconnect：NVLink 3.0（2 TB/s, 600 GB/s per link）或 PCIe 4.0（32 GB/s）
  - 模拟器：LLMCompass [36]（block-level simulator, ISCA 2024），增强支持 MoE + EP + Mixtral 架构
  - 配置：batch size=1, sequence length=512, FP16

- 开源Serving框架是什么。修改了什么。
  - 基座模拟器：LLMCompass [36]（https://github.com/PrincetonUniversity/LLMCompass, ISCA 2024）
  - 修改/增强内容：
    1. **MoE + Expert Parallelism 模块**：新增 EP-specific communication（All-to-All scatter/gather）和 FFN workload 建模。
    2. **Mixtral 架构支持**：实现 Grouped Query Attention (GQA)、SwiGLU activation、Sliding Window attention (4K window)。
    3. **Prediction Strategy 建模**：新增 Distribution-Only 和 Token-to-Expert 两种策略的 runtime 建模，支持可调 accuracy 和 overhead。
    4. **Prediction 性能模拟**：使用 exponential 函数拟合 accuracy-overhead 曲线，polynomial 函数拟合 accuracy-performance 曲线。
  - 论文自身 MoE-GPS 框架代码未开源。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：MoE-GPS 自身代码未开源。LLMCompass 模拟器开源（https://github.com/PrincetonUniversity/LLMCompass）。
  - 框架输入到硬件执行全过程（以 Mixtral 8×7B, 4×A100 NVLink, Distribution-Only Prediction, batch=1, seq_len=512, skewness=1.4 为例）：
    1. **输入**：batch token 序列（1×512 tokens），模型权重分布在 4 个 A100 GPU 上（EP for FFN: 每 GPU 2 experts；TP for Attention）。
    2. **Predictor 阶段**：每层 Attention 前，Distribution-Only predictor 读取 offline MLE 估计的 expert 概率 $\hat{p}_i^l$ → 计算每个 GPU 的目标 token 数 → 触发 Expert Duplication（若某 GPU 负载超阈值则复制热门 expert 到其他 GPU）→ Expert 权重通过 NVLink 传输（~0.1ms, 可与 Attention 重叠）。
    3. **Attention 层（TP）**：输入经 Ring All-Reduce → QKV projection（TP sharded）→ Sliding Window Attention (4K) + GQA → Output projection → Ring All-Reduce → hidden states。Attention 延迟 ~12ms（LLMCompass 保守估计，未使用 FlashAttention）。
    4. **FFN 层（EP）**：Gating network → Top-2 expert selection → All-to-All Scatter（通信，将 token 发送到 hosting GPU）→ expert SwiGLU MLP 计算（因 Distribution-Only 均衡，各 GPU compute 时间相近）→ All-to-All Gather（通信，收集结果）。
    5. **性能**：Distribution-Only Prediction 无 predictor overhead，FFN compute 均衡 → 相比 baseline（无 prediction）节省 FFN compute delay → 相比 Token-to-Expert Prediction（最佳配置）提升 23%。
    6. **输出**：first token 生成（prefill latency），decode 阶段后续自回归生成（论文聚焦 prefill）。
