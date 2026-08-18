## XtraMAC: An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA

- baseline方法是什么？
  现有 FPGA 混合精度 MAC 方案分三类，共同痛点是 DSP48E2（27×18 位乘法器，W_mul=45）的位空间利用率极低：
  ① **操作数 upcasting**（AMD Xilinx Floating-Point Operator IP [1]）：低精度操作数 padding/提升到固定高精度格式，在固定高精度 datapath 上执行——例如 INT4×BF16 把 INT4 提升成与 BF16 对齐的宽格式，低精度 workload 下平均 DSP 位利用仅 32.4%；② **空间复制**（spatial replication，[1]/Tensor Slices [5]）：为每个数据类型实例化独立 datapath + MUX 运行时选择——例如 INT8/BF16 配置同时放 INT8 与 BF16 两套 MAC，同一时刻只有一套活跃，平均 DSP 利用 26.7%，资源随格式数线性翻倍；③ **时间共享**（TATAA [38]）：把 BF16 MAC 分解为 4 个顺序 INT8 微操作复用一个 INT8 datapath，避免重复逻辑，但每个 BF16 操作独占 4 个 PE/流水级，BF16 峰值吞吐只有 INT8 的 1/4，BF16 MAC 有效 DSP 利用仅 8.9%。根因是"低精度 MAC 的位级处理模式"与"FPGA DSP slice 的固定资源粒度"不匹配：DSP 被当作不透明单 lane 原语，数据类型切换被当作整个 datapath 的粗粒度控制问题，而非 DSP 位空间内的细粒度工作量分配。
  - baseline 全栈执行例子（以 Qwen3-8B-AWQ 一个 decode 步、一个 1×4096 tile GEMV、INT4×BF16 投影层 MAC 为例）：
    ```
    算法pipeline层：AWQ 权重仅量化——权重 INT4、激活/累加 BF16/FP32，混合精度由量化方案定义；
    系统框架层：论文未明确说明（无 serving 框架；GEMV 由 FPGA 流水线执行，权重 HBM 直读、激活片上缓冲）；
    编译框架层：论文未明确说明（无编译器；Vivado 综合时按固定数据类型生成 datapath）；
    kernel调度层：GEMV kernel 的 PE 内标量 MAC——upcasting 把 INT4 权重提升为宽格式后经 vendor FP IP 单 lane 执行
               （32.4% DSP 位利用）；空间复制用 MUX 选 INT4 或 BF16 datapath（另一套闲置，26.7%）；时间共享把 BF16
               分解为 4 个 INT8 微操作顺序执行（BF16 仅 8.9% 利用）；
    硬件架构层：单个 DSP48E2 被当作不透明 45-bit 单 lane 乘法器，低精度操作数只占低位、高位闲置；混合精度与运行时
               切换要靠 datapath 级 upcasting/复制/串行化，无法跨数据类型共享乘法器。
    ```
- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 XtraMAC 的统一处理模式公式化：把 INT/FP/INT×FP 乘法全部分解为"整数尾数乘积 + 轻量符号/指数处理"（Eq.1/Eq.4），使 DSP 只做纯整数乘法；把 DSP48E2 当作可分区的位空间，将 P 个低精度 lane 的尾数/幅值按非重叠位偏移打包进 27/18-bit 输入口（Eq.9-12），一次宽乘后 shift-and-mask 提取各 lane 乘积（stride S≥W_lane+G 防跨 lane 干扰）；四阶段固定流水线（解释/位映射→DSP 乘法+逐 lane 后处理→分离式 INT/FP 累加→输出选择），datatype-select 信号经匹配延迟切片贯穿，运行时不重配置、纯 MUX 切换，全数据类型恒定 4 cycle 时延 + II=1；累加按数据类型分离（整数走进位链、浮点走 barrel shifter），避免统一 adder 让整数路径白付 shifter 面积。对应解决 baseline 缺陷：upcasting 的位空间浪费→位级打包让低精度 lane 占满 DSP 位空间（FP4/FP8 达 4 lane、BF16 2 lane，DSP 利用接近 100%）；空间复制/时间共享的"一套活跃其余闲置"→单一 datapath 内所有数据类型共享同一乘法器-加法器流水线，运行时切换只是输入位映射与输出选择的 MUX 切换，无空闲硬件、无 pipeline 冲刷。
  - 论文方法全栈执行例子（同一 Qwen3-8B-AWQ decode 步、同一 1×4096 tile GEMV、INT4×BF16→BF16 投影层 MAC，随后同一硬件切到 BF16×BF16 注意力层）：
    ```
    算法pipeline层：论文未提出新量化算法——消费 AWQ/GPTQ/SmoothQuant/Atom/GPT-oss 已有的混合精度方案，硬件按
               per-tile datatype 信号适配（INT4×BF16 投影层占 decode 期 MAC 的 68%，注意力层 BF16×BF16）；
    系统框架层：论文未明确说明（无 serving 框架；GEMV 流水线：权重 HBM 直读、激活片上缓冲、HBM 512-bit 接口字按
               per-lane 拆分、per-tile datatype 控制信号与操作数同步传播）；
    编译框架层：论文未明确说明（无编译器；Vivado 2022.2 综合 RTL，N（数据类型数）/P（最大并行度）参数综合期选定，
               N_MAC=512/(4×2)=64 个 XtraMAC 级联 per channel，共 1920 实例/30 channel）；
    kernel调度层：INT4 权重段 + BF16 激活进入 XtraMAC——Stage1 按 datatype 位映射打包（INT4 幅值 4-bit 与 BF16 尾数
               8-bit 各按 stride 打包进 DSP 口），Stage2 DSP48E2 一次整数乘法出 2 lane 乘积，Stage3 BF16 累加，Stage4
               打包输出；切到注意力层时仅 datatype 信号变 BF16×BF16，同一硬件同一流水线继续，无 bubble/重配置；
    硬件架构层：1 个 DSP48E2 每 cycle 做 2 个 INT4×BF16 或 2 个 BF16×BF16 MAC（4 cycle 时延、II=1），1920 个
               XtraMAC 在 U55c 上跑 300 MHz（1920 实例 250–270 MHz），GEMV 时延 0.0246 ms（vs CUTLASS H100
               0.0294 ms）、功耗 85 W（vs 135 W）、能量效率 1.9×；相对 vendor FP IP 平均降 LUT 30.0%/FF 47.9%/
               DSP 50.0%、计算密度 1.4–2.0×，相对 TATAA 每 BF16 操作降 LUT 59.7%/FF 72.5%/DSP 93.8%，
               batch=32 端到端 LLM 推理（Alveo V80 analytical 仿真）降时延 1.5–1.8×。
    ```
