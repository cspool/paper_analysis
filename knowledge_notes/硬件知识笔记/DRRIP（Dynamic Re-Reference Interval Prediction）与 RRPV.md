## DRRIP（Dynamic Re-Reference Interval Prediction）与 RRPV

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DRRIP（Jaleel et al., ISCA'10）基于 RRIP（Re-Reference Interval Prediction）：每个 cache 行维护 Re-Reference Prediction Value（RRPV），表示预测的再引用距离——0 = 近期会被再次引用（最高优先级），3 = 最不可能被引用（最低优先级、可优先淘汰的 vulnerable 位置）。DRRIP 在 RRIP 上增加 set dueling，让 SRRIP（静态）与 BRRIP（双峰插入，用概率选择 2/3）在少量采样 set 上动态竞争，选择胜出的插入策略。Bumper 基线 L2C = 6MB 12-way DRRIP，且按行类型（代码/数据/预取/MMU）优化插入与提升；Bumper 只修改代码行的插入/提升规则（Table I：IFU miss → RRPV=3 插入；IFU hit → 仅当 RRPV<3 时提升到 0；Hint 请求 hit → 提升到 0；Hint miss → 无动作；其他请求沿用 baseline），因此可叠加在任何 RRPV 类策略上，对数据/MMU 行无影响。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程：行插入时写 RRPV（Bumper 代码行固定写 3）；victim 选择时从 RRPV=3 的行中选；全局 aging 计数器每到期一轮把所有 RRPV 递增（向 3 漂移）；命中把 RRPV 降到 0。Bumper 的关键修改是"RRPV=3 的 IFU hit 不提升、等待 commit hint"——因为 RRPV=3 的代码行可能是 (i) 新插入但有用的行（尚未提交）、(ii) useless 行（永不提升）、(iii) 曾提升后老化回 3 的行；对三者而言即时提升都可能浪费容量，只有 commit hint 能鉴别。论文未发现"区分新插入与老化到 3 的行"的替代策略有收益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每行 2-bit RRPV + 全局 aging 计数器（周期翻转）；DRRIP 的 set dueling 用约 32 个采样 set。Jiménez MICRO'46（Bumper 引文 [37]）为 tree-pLRU 提出 Insertion/Promotion 变体，与 RRPV 思路互通。Bumper 的存储开销（422B）远小于任何按 L2C 规模扩展的预测型替换策略，且其提升规则与 DRRIP 本体解耦。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
