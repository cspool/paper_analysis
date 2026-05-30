## S-NIAH (Single Needle In A Haystack)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
S-NIAH 是 RULER benchmark (Hsieh et al., 2024) 的合成检索评测套件，包含三个难度递进任务：S-NIAH-1 (passkey retrieval)：合成上下文中放置 key-value "needle"，测试纯长期记忆保持（上下文不含其他有意义信息）；S-NIAH-2 (number in haystack)：真实文章上下文中放置数值 needle，测试选择性记忆+噪音过滤；S-NIAH-3 (UUID in haystack)：needle 的 value 是 UUID（复杂模式），测试复杂模式记忆。评测不同序列长度（1K-8K+）下的检索准确率。核心诊断能力：S-NIAH-1 测"记忆保持"，S-NIAH-2/3 测"选择性记忆+过滤"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
S-NIAH-1 格式: "The special magic number is 12345. [repeat filler × N] What is the magic number?" → Target: "12345"

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RULER 开源：https://github.com/hsiehjackson/RULER。Gated DeltaNet 论文利用该 benchmark 验证 gating 与 delta rule 互补性：DeltaNet S-NIAH-1 完美但 S-NIAH-2/3 崩溃（缺遗忘），Mamba2 S-NIAH-2/3 较好但 S-NIAH-1 长序列崩溃（过度遗忘），Gated DeltaNet 在所有任务上最佳平衡。适用于评测新线性 RNN 架构的记忆保持与选择性记忆能力。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---
