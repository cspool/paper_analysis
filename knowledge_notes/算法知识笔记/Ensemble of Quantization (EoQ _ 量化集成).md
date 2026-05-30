## Ensemble of Quantization (EoQ / 量化集成)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ensemble of Quantization（EoQ）是 QT-DoG 利用量化模型体型优势的高效集成策略。训练 E 个独立随机初始化的 QT-DoG 量化模型（7-bit），推理时 bagging：ŷ = argmax_k Softmax((1/E) Σ f(x; w_q^i))。核心优势：5 个 7-bit 模型总存储仅 1.1× 全精度单模型，却在 DomainBed 上达 68.4% 平均准确率（超过 DiWA 68.0%/EoA 68.0%）。相比之下，DiWA 需 60 个全精度模型（60× 体积，12× 训练开销），EoA 需 6 个全精度模型（6× 体积）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
EoQ 训练：独立训练 E=5 个 QT-DoG 模型（仅 random seed 不同），每模型在各自验证集上选最优 checkpoint。推理：probs = [softmax(m(x)) for m in models], ŷ = argmax(mean(probs))。EoQ 总内存 = E × 0.22× = 1.1× 全精度；串行推理延迟 = E × 操作时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
EoQ 局限：(1) 训练计算量为单模型 E 倍；(2) 串行推理延迟为 E 倍（可并行化改善）。EoQ 在 TerraIncognita 提升最显著：ERM 47.2% → QT-DoG 50.8% → EoQ 53.2%（+6.0% vs ERM）。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization
