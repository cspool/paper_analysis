## BQU（Binary-coding Quantization Unit，二值编码量化单元，含 TSE 与 BEA）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BQU 是 Omni-LUT（ISCA 2026，NYCU）加速器中把浮点 Key/Value 激活在线转换为 BCQ（binary-coding quantization）表示的轻量硬件引擎，输出格式直接被 LUT-based PE 消费。设计动机：KV 激活是推理时在线生成的，量化开销必须极小；且 Key 与 Value 分布不同（Key 有稳定 per-channel outlier 结构，Value 是高度 token 相关的动态分布），需要两条路径：(1) Key Path——per-channel 量化，缩放因子离线校准、BCQ bit-planes 在线计算；(2) Value Path——per-token uniform 量化，缩放因子与 bit-planes 都在线计算。BQU 内部两个子块：Token-Scale Estimator（TSE）与 BCQ Encoder Array（BEA）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - TSE：pipelined reduction tree，并行处理一个 incoming value token 的全部元素求 x_min/x_max，由极值算两个 uniform 量化参数 zp_v=(x_max+x_min)/2、δ_v=(x_max−x_min)/(2^b−1)（b 为目标位宽）；BCQ 缩放因子 {α_i} 直接由 uniform step 导出——δ_v 乘以预定义 power-of-2 base scales（如 q=4 bit 时 {4,2,1,0.5}），连同 zp_v 送往 BEA。BEA：把全精度 Key/Value 向量转成二值编码表示——输入向量 x、zero-point zp、缩放因子 {α_i}，输出 B∈{-1,+1}^{q×d} 满足 x≈zp+Σ_{i=1}^q α_i⊙B_i（d 为向量长、B_i 为第 i 个 bit-plane）。BEA 实现贪心残差算法：r^(0)=x−zp；对 i=1..q：B_i=sign(r^(i-1))，r^(i)=r^(i-1)−B_i·α_i——每步决定加/减 α_i 把残差拉向零，剩余由后续更小 α_i 处理。两条路径共用同一贪心算法，区别仅在输入来源：Key Path 取离线校准的 per-channel zp_{k,c}/α_{i,c}，Value Path 取 TSE 的 token-wise zp_v/α_i。
  - 运转流程（一个 decode token）：新 token 的 K/V 激活从 unified buffer 读出 → Value 走 TSE 求 per-token 参数 + BEA 编码；Key 用离线参数走 BEA 编码 → 量化后的 bit-plane 写回 buffer/量化 KV cache → 供 LUT PE 查表执行 QK^T 与 Attn×V。BQU 宽度 128，匹配评估模型（OPT/LLaMA2/LLaMA3/Mistral/Mixtral/Qwen3）的最大 head dimension，因此一个 head 向量一次量化完成、不拆分多 pass，在线量化不增加 cycle。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：RTL 综合（TSMC 7nm @500MHz），面积 0.014 mm²（占 compute core 2.63%）、能量 0.16 J（占 0.25%，Table VII）——与 control logic 合计的在线量化/控制开销约 5% 面积、<1% 能量，相对 compute core 很小。使用：作为 LUT-based GEMM 加速器的在线量化前端，与 LGU、PE array 流水协同；对应算法侧是 BCQ/BC-UQ（见算法pipeline 层条目）。BEA 的贪心残差编码即 Algorithm 1 在线侧的单向量执行形式（离线交替优化只用于 Key 的缩放因子校准）。论文未明确说明 BQU 的门级实现细节（如编码器并行度、流水级数）。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization
