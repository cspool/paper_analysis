## In-NAND Computing（in-NAND 存内计算：3D NAND 阵列内模拟域 VMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
In-NAND computing 指把向量-矩阵乘法（VMM）直接在 3D NAND 闪存阵列内部以模拟域完成：权重编程进 NAND 单元（SLC 单层存储模式），输入向量经电压施加在选通管上，各串（string）汇聚电流沿位线（BL）相加，经 ADC 量化输出数字 MAC 结果。DIAMoND 采用电流域方案 [27][51]：特定层经字线（WL）激活（该层存权重），顶选通管（TSG）馈入输入向量——输入 1 加导通电压、输入 0 加截止电压，BL 汇总聚合电流；该过程与 NAND 常规读操作一致，外围电路改动最小。为何用 NAND：容量密度 20.27~29.18 Gb/mm²、单 die 1Tb，能装下 Mixtral-8x7B INT8 47GB 级权重且非易失；in-NAND 并行度达每 die 279~1118 GOPS（@8bit，输入并行度 512/1024/2048），远超 near-NAND 计算的 1.6~38 GOPS/die。限制：模拟域噪声（D2D 变差、Vth 漂移、ADC 量化）使输入维受限于平面 TSG 数（~2048，实际约一半参与）；NAND 耐久 ~10^3 P/E，不适合 self-attention 等动态写 KV 的操作——故 DIAMoND 只把 FFN 专家矩阵放在 in-NAND，attention 交给 DRAM 侧。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
DIAMoND 的 die 级设计（NAND die 采用 YMTC 232L QLC 工艺配置，外围加 ADC/Shifter/Adder 堆叠在 die 上方）：INT8 权重按 2's complement 存入 SLC cell；激活位并行——8 个 die 各负责激活的 1 bit（权重矩阵跨 die 复刻），单次 read cycle 聚合出 8-bit MAC，由 Control Die 的加法树求和。芯片级数据路径（一次 VMM）：选 OU（512 TSG × 若干 BL 的矩形区域）→ 权重层与 mask 层 WL 同加 Vread（饱和区 + Current Clamping ~30nA 抑制 D2D 变差）→ TSG 按激活 bit 施加导通/截止电压 → BL 汇流模拟电流（所有 BL 通电）→ 7-bit ADC（@40dB SDNR）量化 → Shifter/Adder 位权重合成 → Control die 加法树跨 die 聚合。可靠性设计链：D2D 变差（把 Vread 从线性区①抬到饱和区②，读电流对 Vth 不敏感，σ 从 0.15 降到 0.02）→ Current Clamping（③，电流限 ~30nA，兼顾功耗与精度）→ Vth 漂移（降 Vpass + 计算中 on-the-fly 校准，>10G 可靠读；每权重每 token 至多读一次 → 80+ 天连续运行，超限 block 刷新，1K P/E 下 10+ 年寿命）。多 WL 激活可行性：导通后电流由 string 电阻（~1MΩ）而非选中 cell（~10kΩ）主导，激活多 WL 不改变计算结果（Flash-Cosmos 多 WL 传感、TCAM-SSD 多 WL 检索已商业验证）；外围改动类比 SanDisk/SK-Hynix HBF 与 Xtacking 的外围增强。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基线：die 配置 256Gb SLC、4 plane、页 16kB、读延迟 tR=30µs、单 die 面积 50.51mm²；协议 ONFI 6.0/JESD230G（每通道 4.8GB/s、2 通道 9.6GB/s），16 die 分 2 通道、每通道 8 die 经 TSV 堆叠 + Control Die 加法树。评估方式：基于 SSDsim（开源，https://github.com/jiangyu718/ssdsim）构建 cycle-accurate 模拟器并注入噪声模型；ADC 采 28nm 7-bit 设计 [31]、softmax/SiLU 单元采 [46][62]、数字部分 Synopsys DC 28nm 综合 + CACTI 建 SRAM。使用方式：只承载静态权重、每 token 只读的 VMM（FFN 专家、QKV 投影中的 K/V）；动态写密集型操作（KV cache、self-attention）留给 near-DRAM——按存储介质特性分工是 DIAMoND 的核心原则。相关体系：nvCIM [16][35][36]、3D-NAND CIM 架构 [50][51][64]、Flash-Cosmos（bitwise）、TCAM-SSD（search）；对比 RRAM 类 IMC，NAND 胜在容量密度（RRAM 仅 Mb 级）。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
