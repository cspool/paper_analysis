## 浮点位翻转敏感性分析（sign/mantissa/exponent 位翻转误差模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
这是 RangeGuard 的动机分析（§III-B/C）：系统刻画 DRAM 存储错误（bit flip）如何映射为数值错误、再传播为 DNN 端到端准确率损失的数值模型。对 BF16（1 符号位 s + 8 指数位 e + 7 尾数位 m，bias=127），x = (−1)^s × 2^(e−bias) × (1.m)：(1) 符号位翻转 → x'=−x，误差 = 2|x|，有界且正比于原值；(2) 尾数位翻转 → |x'−x| ≤ ½|x|（k=0 最重），全部尾数位翻转累积仍 <|x|，指数衰减；(3) 指数位翻转 → x' = 2^(±2^p)·x，scale factor SF=2^(±2^p) 随位指数 p 双重指数增长（p=7 时 0→1 翻转 ×2^128≈3×10^38，1→0 翻转 ×2^-128 趋零）。核心结论：exponent 位（尤其高位）是唯一能制造"超出原值量级"的灾难性错误的位，且 0→1（放大）比 1→0（衰减）危险得多；尾数/符号位错误相对良性。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
端到端验证流程（§III-C，PyTorch 推理 + 100 次 Monte Carlo trials）：对每个目标 BER 在权重与中间激活上均匀注入随机 bit flips → 跑完整推理 → 统计准确率。位级结果（Llama-3.2-1B）：sign/mantissa 位错误到 BER=10^-5 才明显，exponent 位错误在 BER=10^-12 就足以让 1/100 trial 崩溃（平均仅 0.15 flipped bit/trial）；e[7] 位在 BER=10^-11（约 1.5 flipped bit/trial）时 10/100 trial 跌向随机。模型对比：ResNet-50 在 BER=10^-8 开始退化、10^-7 崩溃；Llama-3.1-8B 在 10^-10 退化、10^-9 崩溃——LLM 比 CNN 脆弱 2~3 个数量级，源于 transformer 的 attention 放大 + residual 跨层保留 + LayerNorm 稳定化使单个 outlier 长期存在并级联。这就是"保护 exponent 类范围变化"（RID 语义保护）的直接依据。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 Hugging Face 模型（PyTorch）+ lm-evaluation-harness（ARC-Easy、ImageNet-1k）的软件错误注入框架，方法学与 PyTorchFI/MRFI/ReaLM/FIdelity 等 fault-injection 框架一致；论文未开源注入代码。使用价值：① 评估任何 ECC/容错方案前先定位"哪些位真正致命"（本文证明 exponent 高位主导、低阶位浪费预算）；② 为 RID/range 类语义保护提供"哪些范围变化需要保护"的量化依据；③ 灵敏度模型（SF=2^(±2^p)）可直接用于推导误差上界与 RangeMap 构造。
涉及论文标题：
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs
