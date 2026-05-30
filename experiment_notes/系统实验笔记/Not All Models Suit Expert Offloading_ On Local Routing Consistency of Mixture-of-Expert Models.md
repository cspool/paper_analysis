## Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：本文本身**没有实现或修改任何开源 Serving 框架**，而是从**模型侧**分析 MoE 模型对不同 expert offloading 策略的友好程度。核心是提出 SRP 和 SCH 两个指标量化 MoE 模型的"局部路由一致性"（即连续 token 倾向于激活相同 experts 的程度），该属性直接影响 expert offloading/caching 系统（如 LRU/LFU cache）的 hit rate。论文通过理论证明 SCH 可近似 Belady 最优缓存的命中率，并实验验证 SCH 与实际 cache 算法（LRU、LFU）hit rate 高度正相关（m=16 时 Pearson r > 0.88），证明 SRP/SCH 可作为选取/设计 expert offloading 系统的模型级参考指标。此外，论文实现了一个简单的 LRU-based expert offloading 系统（naive 版本，on-demand loading）在 TOY 模型上测量 throughput，验证局部路由一致性与 decoding 阶段 overhead 负相关（r ≈ −0.3）。
  - 实验比较：(1) 20 个 REAL 模型在不同 cache ratio ρ 和 segment 长度 m 下的 SCH，分析 SCH 随 ρ 增长的曲线形状（group 1 模型在 ρ=2 处出现拐点）；(2) SCH 与实际 cache 算法 hit rate 的 Pearson 相关性（m=4/16/64/256, ρ=0.5-3.0）；(3) SCH 与 Belady 最优 hit rate 的相对比较（Baseline TOY 模型），展示 LRU/LFU/SCH 在不同 ρ 下相对于 clairvoyant replacement 的归一化 hit rate；(4) TOY 模型在 LRU-based expert offloading 下的 decoding/prefilling throughput 与 full GPU inference 的相对 overhead。

- 硬件平台是什么，配置是什么。
  - REAL 模型路由决策收集：NVIDIA A100 PCIe 80GB GPU。TOY 模型 expert offloading throughput benchmark：单 GPU（A100）内存不足以容纳完整模型，模拟 memory-constrained 边缘设备场景（Appendix F）。具体配置：GPU 内存足够容纳 activated parameters + 计算，但不足容纳全部 experts；CPU 端有充足 flash memory。

- 开源Serving框架是什么。修改了什么。
  - 论文未基于任何开源 Serving 框架实现（如 vLLM、SGLang 等）。论文关注的是通用 expert offloading 场景下的**模型路由行为分析**，而非特定系统的实现或修改。论文的 naive LRU expert offloading 实现为自行编写，未说明基于何种框架。论文提到主流 expert offloading 系统包括 SwapMoE (Kong et al., 2024)、MoE-Infinity (Xue et al., 2024b)、EdgeMoE (Yi et al., 2025)、AdaPMoE (Zhong et al., 2025) 等作为背景参考。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源代码：https://github.com/ljcleo/moe-lrc
  - 本文分析结论如何指导专家 offloading/caching 系统设计：
    ```
    1. 模型选择（离线/部署前）：
       输入：候选 MoE 模型列表
       过程：用本文代码对每个模型在通用语料上计算 SCH(m=16, ρ=2)
       输出：选择 SCH 最高的模型部署到 memory-constrained 设备
       依据：SCH > 70%（如 LLaMA-MoE-v2）的模型可预期 LRU/LFU cache 获得 > 65% hit rate

    2. 缓存大小选择：
       输入：目标 MoE 模型的 SCH 随 ρ 曲线
       过程：找到 SCH 增长从"快速"变为"平缓"的拐点 ρ
       输出：缓存大小 = ρ × k × expert_size（k = 每 token 激活 expert 数）
       依据：大多数模型 ρ=2 时在 cache 效率（hit rate）与开销（GPU 内存）间取得最佳平衡

    3. 运行时 expert cache 策略（以 LRU 为例）：
       GPU 内存布局：
         - pinned experts: k 个（必须保留，对应每 token 激活数）
         - cached experts: (ρ-1)×k 个（LRU cache pool）
         - 其余 experts 驻留 CPU 内存

       解码阶段每步执行流程：
         a. 当前 token x 输入 router → 得到 top-k experts
         b. 如果 top-k 全部在 cached experts ∪ pinned experts 中 → GPU 直接执行
         c. 否则 → 从 CPU 加载缺失 expert 到 GPU（on-demand load），
            驱逐 LRU pool 中最久未用的 cached expert
         d. GPU 执行 expert FFN 计算
         e. 更新 LRU 访问记录

       论文核心贡献：上述系统的 cache hit rate 上限由 SCH 决定；
       高 SRP 模型（group 1，如 LLaMA-MoE-v2、OLMoE）即使最简单的 LRU
       也能获得高 hit rate，无需复杂的 prediction-based prefetching。
    ```
