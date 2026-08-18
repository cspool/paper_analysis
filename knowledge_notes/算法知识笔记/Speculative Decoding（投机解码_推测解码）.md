## Speculative Decoding（投机解码/推测解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
投机解码是加速 LLM 自回归 decode 的无损推理范式：用轻量草稿模型（draft model）快速自回归预测 γ 个候选 token，再由目标模型（target model，即原始 LLM）对这 γ 个 token 做一次 batched 前向并行验证，按接受规则保留正确前缀。因为 γ 不大且 decode 阶段本就 memory-bound，γ-token 验证前向耗时近似单 token 前向：总时间 ≈ γ·t_draft + t_target，t_draft ≪ t_target 时收益显著。接受率（acceptance rate）= 草稿平均每轮被接受的 token 数，是加速比的决定因素；接受判定贪婪解码下要求草稿与 target 输出完全一致，采样解码下用概率接受（见 Speculative Sampling）。别名：speculative execution for LLM、draft-then-verify。代表工作：Leviathan et al.（arXiv:2211.17192）、Chen et al.（arXiv:2302.01318）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
x = prompt
while 未生成 EOS:
    drafts = []
    for i in 1..γ:                          # 草稿阶段：轻量、自回归
        q_i = draft(x + drafts)
        drafts.append(sample(q_i))
    p_1..p_γ+1 = target(x + drafts)         # 验证阶段：一次 batched 前向
    n = 接受判定(p, q)                       # greedy: 首个不一致前；采样: rejection sampling
    x += drafts[:n] + target_sample(x + drafts[:n])
```
本文（Cassandra）的用法：草稿不是独立小模型，而是目标模型权重/KV 的严格比特子集（speculation data，zero-padding 重建后标准 FP GEMM）；验证加载 speculation+verification 全量数据完全重建原始模型。γ 在 3–5 内取最优（Cassandra-1 γ=5、Cassandra-2 γ=3）。对比基线：EAGLE-3（训练型草稿，4×A100 约两天）、Draft&Verify（层跳过）、MagicDec（KV 稀疏检索）、Lookahead Decoding（n-gram）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：vLLM/SGLang/TensorRT-LLM 内置（支持 Medusa、EAGLE、ngram、独立草稿模型等），llama.cpp 支持草稿模型，HuggingFace transformers 的 assistant model 模式。硬件系统方向：PIM 加速草稿模型、FPGA（DFVG）。代价与适用边界：训练型草稿有训练成本与额外显存；低 batch 下部分方法（MagicDec 的 KV 剪枝）甚至慢于 baseline；训练无关变体见 Self-Speculative Decoding。

Raptor 补充视角（ISCA'26）：投机解码被定位为 Raptor 与 HBM-class GPU 的异构配对（与 AFD 相反）——draft 阶段是 K 步背靠背自回归、内存受限，放 Raptor（~100TB/s 压低串行 draft 关键路径延迟）；verify 阶段把 target 权重摊到 K 个 token 上、计算受限，放 GPU tensor core。Raptor 前代 Corsair 已在该配对下有规模化端到端加速的生产部署先例（Gimlet Labs 博客），NVIDIA Vera Rubin 平台（Rubin GPU + Groq 3 LPX）也把投机解码列为目标用例。这体现"draft=内存受限、verify=计算受限"的不对称性可被高带宽内存基板放大的系统级收益：draft 模型跑 Raptor 时每次推测步的 KV/权重带宽成本远低于 HBM 基板。

HybridSpec 补充视角（ISCA'26，SD 的"内存需求极化"与硬件设计杠杆）：SD 不止是延迟优化算法，其 draft/target 分裂天然极化内存需求——draft（体积 <1/10 target）逐 token 自回归、算术强度低、内存受限 → 需高带宽但仅小容量；target 一次验证多个 draft token、算术强度高 → 容忍低带宽但需大容量（权重 + 增长中的 KV cache）。据此 HybridSpec 把 draft 放 HB 栈（4TB/s 高带宽）、target 放 XPU+LPDDR5X（512GB 大容量），模型级映射使通信只在 draft-verification 边界。论文覆盖 SD 全谱系：chain 式（draft 逐 token 自回归）、tree 式（多候选 + masked attention 并行验证，见 Tree-based Speculation 条目）、hidden-state 式（在 target 内嵌投机头，用最后 token + 末层表征预测，图 3(c)）；成熟实现（vLLM/SGLang/llama.cpp 内置）接受率可达 ~80%[39]。

从算法pipeline角度拆解（极化后的一次 draft-verify 周期）：draft 在 HB 栈按当前 tree width 生成候选树（memory-bound 迭代）→ 达 draft budget 后 token 列表传 XPU → target 一次 batched 前向并行验证（rejection sampling）→ accepted 前缀回传、清误推测 KV。budget/tree width 由 Utilization-aware Speculation 按两侧 roofline 动态调（见系统架构层条目）。

实现与使用：算法侧与通用 SD 相同（draft-then-verify + 拒绝采样保证 lossless）；系统侧的新意是"为 draft/target 各自选择内存基板"——这是 SD 从纯软件优化升级为"软硬件协同异构设计"的用法，与 SpecPIM（PIM 侧 DSE）等并存。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving

IroKnight 补充视角（ISCA'26，Fully-State Encrypted 投机解码）：IroKnight 论证投机解码不改变 LLM 的细粒度 tiled/vectored 执行——draft 与 target 模型本身及其算子不变，只新增"候选 token 验证"的比较操作，而该比较等价于对 token 索引向量的规则等值检查（仿射访问模式），因此 pad 预计算与同周期 in-ALU 加解密照常成立，可实现"全状态加密"的投机解码（模型参数、用户 query、draft token 等在所有存储中保持密文，明文仅瞬态于 ALU）。评估：draft Llama3-1B + target Llama3-70B（conventional）与 draft Llama3-8B + target Llama4-Scout（mixed-vocabulary）两组，drafter 3/5 token、接受率 alpha 0-100% 扫描：加密变体延迟开销 0.1%-0.5%（低接受率略高，因 rollback 需重灌 PadGen 流水线），认证变体 3.1%-3.3%；能量加密 9.4%-29.3%、认证 13.6%-33.9%（接受率越高 target 生成 token 越少、HBM 流量越低，能量开销反而升高）。结论：SD 这类"粗粒度动态性"与 IroKnight 的"细粒度规则加密"正交兼容。
