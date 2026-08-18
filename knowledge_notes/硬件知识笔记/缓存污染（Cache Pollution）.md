## 缓存污染（Cache Pollution）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 缓存污染指预取/填充把将来有用（demand 会访问）的数据从缓存逐出，或把无用数据占住容量，导致命中率下降的性能损失。预取器越激进越容易污染（低准确率→大量无用填充）。Moirai 专门把防污染作为一等设计目标：其 Adaptive Control Unit 用相位切换（训练期暂停预取）+ 置信度分级（低置信发更多覆盖但限定在推理期）双重机制从源头遏制污染，而先前方案（Hermes、TLP）依赖下游感知机过滤或跨层同步。论文量化：Moirai 的 L1D 预取引发 eviction 绝大多数 harmless（与 IPCP 相当），无用预取大部分在片内被吸收（L2/LLC hit），DRAM 访问率低于 IPCP。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 污染路径与遏制（Moirai）：预取填充 L1D → 若填的是"永不访问"的行并逐出活跃工作集行 → 污染。Moirai 的防污染闭环：模型推理期 → 预测准确率下降 → L_avg 上升 → 先升激进性（1→5→9 预取，短时最大化覆盖）→ 越过 Ltrain 阈值 → 切训练相位、暂停全部 CaPNet 预取（主节流机制）→ 辅助 stride 预取器保持基础空间局部性 → 收敛后再回推理。多核场景：线程争用使预测降准 → loss 上升 → 自动节流，防污染的同时等效于显式带宽感知（Pythia）效果（多核 7.8% vs Pythia+Hermes 8.3%）。
- 对比：BTCP 等先前 B-TCN 预取器无显式污染控制（论文 VI-C 指出的区别）；记忆式预取器单点噪声即可链断裂污染模式，TCN 的时序感受野把不规则噪声当孤立扰动平滑掉。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件状态 L_avg + 计数 + 阈值（Moirai Table II：Ltrain=0.5、Linf=0.01、Ntrain=131072、Ninf=2048），相位状态机 + 组合逻辑分档；评估中用 L1D eviction 分类（harmless/harmful）+ DRAM 访问率量化污染程度（Figure 14）。使用场景：任何带预取的缓存层次设计权衡；Moirai 的"源头发污染控制"（tightly-coupled sequence generation + loss-based throttling，780 Bytes 内自包含）替代复杂跨级同步（Hermes/TLP 的 downstream filter）。

R-Max 补充视角（ISCA'26，Oracle 预取下的污染界）：R-Max 把"预取污染"作为设计约束显式建模——其预取必须与 demand 一样经 MSHR/带宽/容量/延迟流入缓存，填充时产生替换，因此不能违反容量约束。核心发现：即便地址预测完美（oracle），"盲预取 + LRU 替换"也会因预取流跑得过快、过早逐出有用块造成污染（LRU+Omniscient Prefetching 例子中 C 在第 8 步被预取、第 12 步被 F 的预取逐出、第 15 步 miss）；因此 R-Max 把 Bélády's MIN 扩展为"预取时机决策"——只在被替换块 dead counter 归零（最后一次使用之后）才发预取，预取窗口=被逐块最后使用之后、当前块 miss 之前。R-Max 自身因 oracle 预测准确几乎无污染（DRAM utilization 反而降 47.93%），而 SPP 因错误预取污染使 DRAM utilization 升 21.43%。
STEP 补充视角（ISCA'26，触发时机与污染）：STEP 把污染控制内建于触发时机——早触发（FOE）激进下发易污染，故用 Prefetch-Confidence Evaluator 要求匹配足迹收敛（Jaccard >0.75）才下发、下发交集而非并集（优先精度）；证据不足则推迟 SOE/TOE 消歧（mcf-484 案例：若 SOE 时误选候选 C1/C2 将引入 30/12 条无用预取并漏取 11 条 demand）。污染压力测试（Fig.20）：受限 way 预取（预取行限 1 way）下 STEP 仍 1.263× 领先 eBingo 1.245×，证明 STEP 收益不依赖低污染——分阶段触发同时改善污染行为与"早机会-晚消歧"平衡。消融 STEP-D3（禁 TOE）精度全套件下降，说明 TOE 主要恢复精度/降污染；STEP-D2（禁 SOE）平均最优。


涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
- R-Max: Extending Bélády's MIN with Prefetching to Bound Realistic Cache Performance
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
