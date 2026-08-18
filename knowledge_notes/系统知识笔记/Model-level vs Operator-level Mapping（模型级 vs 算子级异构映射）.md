## Model-level vs Operator-level Mapping（模型级 vs 算子级异构映射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
异构加速系统中"把什么粒度的工作放到哪个处理单元"的两种映射策略。Operator-level：按算子算术强度把低强度算子（如 attention）卸载到高带宽内存设备，高强度算子留 XPU（NeuPIM/IANUS/SpecPIM 等 PIM 工作采用）；缺陷是每次 forward 都要在单元间搬运逐层激活与部分和、通信量几个数量级增大，且 HB/PIM 容量小装不下随请求增长的 KV cache（HB-ATTEN baseline 实证）。Model-level：把整个模型（draft 或 target）作为一个整体放到一个单元——HybridSpec 把 draft 模型整体放 HB 栈、target 模型整体放 XPU+LPDDR5X，通信只发生在 draft-verification 边界（draft KV cache 几 MB + token 列表几百 B、hidden-state SD 几十 KB），XPU-HB 链路用低成本 wire bonding 即够（数据移动 <1% 执行时间）。前提是 SD 把模型内存需求极化（draft 带宽型、target 容量型），使模型级切分天然匹配两个物理基板。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
对比例（同一 serving 负载，图 22）：operator-level（PIM 式）需要两个 XPU（draft/target 各一）+ PIM 栈，每个 token 的 QKV/attention 中间激活每层跨设备搬运；model-level（HybridSpec）单 XPU + HB 栈，每请求只在"prefill 后移交 HB 栈"与"draft 达 budget 回传验证"两个边界交换小数据。执行流见 XPU/HB Stack 条目。更一般的发现：仅按算术强度映射而忽略容量约束，在线 serving 中（KV 持续增长）会限制并发、抬高 TTFT——容量应作为与算术强度并列的映射判据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：model-level 要求目标单元能装下整个模型（HB 栈装小 draft、XPU 大内存装 target），且单元具备完整逻辑 die（HB 栈集成逻辑 tier 而非 PIM 的简单 MAC）；operator-level 是 PIM 制造限制（DRAM 工艺只容简单 MAC）下的被迫选择。使用要点：异构设计先选映射粒度（模型级需模型内存需求极化、算子级适合局部卸载），再据此定互连带宽/封装成本。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
