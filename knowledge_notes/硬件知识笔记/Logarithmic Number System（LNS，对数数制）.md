## Logarithmic Number System（LNS，对数数制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LNS 是把数值表示为 2 的对数（log2(x)）的数制：乘法变加法（log2(x·y)=log2(x)+log2(y)）、除法变减法（log2(x/y)=log2(x)−log2(y)）、幂变乘法（log2(x^y)=y·log2(x)）、对数变乘常数（log_b(x)=(1/log2(b))·log2(x)），从而用廉价的加法器/乘法器替代昂贵的乘法器/幂运算。代价是 LNS 中加/减复杂（需查表算 2^a±2^b），且对 0 与负数未定义（LoRA 用绝对值变换、符号与零编码在额外 bit）。经典基础：Mitchell 1962 提出的二进制对数近似（log2(1+f) 分段线性），LoRA 的 LOG/ALOG 转换器即改进 Mitchell 方案。LoRA 采用混合数制：加/减回普通算术、其余在 LNS。
- 在 LoRA 中的作用：把高次多项式 c_ix^(k_i) 化为 2^(log2(c_i)+k_i·log2(x))——幂运算变成"一次 log 变换 + 一次乘（k_i 与 log2(x)）+ 一次反log"，5 项多项式只需一个可编程 30b 乘法器（分解 5×30b×6b）而非 5 个高精度乘法器；同时让 XCore 天然支持 x^y、log_b(x) 等复杂运算。误差分析（III-A）：ε(x)=|f(x)−f̃_HW(x)| ≤ model error（分段逼近）+ implementation error（log/antilog 转换误差 δ、β），c_ix^(k_i)·2^(k_i·δ)≈c_ix^(k_i)(1+ln2·k_i·δ)+β，故转换器精度直接决定实现误差。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 XCore 中的运转流程（多项式模式）：定点/FP32 输入 x → LOG 级对数转换器：定点用 LOD 检测最高位 m 得 |x|=2^m(1+f)、浮点直接取 E−127+log2(1+f)，APP 单元 PWL 近似 log2(1+f)，拼接成 LNS 格式 → LNS 级 30b×6b 乘法算 log2(c_i)+k_i·log2(x)（5 项并行）→ ALOG 级反log 转换器：2^(m+f)=2^m·2^f（2^f 用 PWL+自定义 fine-tuning），CPA 算 log2(x)±log2(y)、SAT 饱和检测 → 普通算术加法树求和。
- 举例：计算 x^9（普通算术需多个乘法器）在 LNS 中为 2^(9·log2(x))——log2(x) 一次变换、9·log2(x) 一次乘、反log 一次变换，硬件开销大幅降低。LoRA 用 6 项多项式时 30-bit LNS 操作数分解为 5 个 6-bit（k_i 只占 6 bit）；4/5 项时 LNS 宽度降到 18/24-bit 避免浪费，但窄 LNS 下达到高转换器精度更难（Table V：转换器精度随项数/位宽权衡）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：LOG/ALOG 转换器是关键模块——log2(1+f) 与 2^f 均只在小输入域 [0,1) 定义，用 PWL 分段线性近似 + 约束驱动参数探索框架 [82] 自动生成 APP 单元（目标精度下最小硬件开销）；LNS 数据 32-bit（含 2-bit 符号/零标志）。相关硬件工作：huicore [10]、低功耗统一算术单元 [48] 也采用 LNS 逼近非线性函数，但 [48] 只支持定点且只用 Taylor、缺高效逼近算法。
- 使用：XCore 四种模式均通过配置切换 LNS 用法——乘/除（LOG 变换 y + ALOG 加减）、幂/对数（LNS 乘法器算 y·log2(x) 或 (1/log2(b))·log2(x)）、多项式（5×30b×6b 乘加）。用 LNS 的前提是目标格式与转换精度满足应用误差容忍度（LoRA 的 XCore 逼近误差与软件逼近 LoRA-SW 同数量级，sin 例外受转换器精度上限 ~1e-5 影响）；适用场景是误差容忍的 AI/DSP 应用（激活函数、三角函数、对数/幂）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
