## 混合精度 MAC 与运行时数据类型切换（Mixed-Precision MAC & Runtime Datatype Switching）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 混合精度量化带来的两类硬件需求（XtraMAC 定义）：① 混合精度 MAC（Mixed-Precision MAC）——两个乘数 A、B 用异构数值格式/位宽（如 INT4×BF16），P=A×B+C；② 运行时数据类型切换（Runtime Datatype Switching，RDS）——同一硬件单元在执行中按模型组件交替使用不同 MAC 数据类型，如单次前向从投影层 INT4×BF16 切到注意力层 BF16×BF16。量化方案产生多样的 MAC 组合（Table I）：权重仅量化（AWQ/GPTQ/SpQR）投影层 INT×FP+FP→FP、注意力 FP×FP+FP→FP；权重-激活量化（SmoothQuant/Atom）投影层 INT×INT+INT→INT；原生 LLM（GPT-oss-20b/120b）MoE 块 MXFP4、其余 BF16。Qwen-3-8B-AWQ decode 期 68% 的 MAC 是 INT4×BF16（投影层），注意力层保持 BF16×BF16——两类需求在一个模型内共存。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
XtraMAC 在单一 datapath 内支持两种模式：N 个数据类型映射子模块静态实例化、datatype-select 信号注册在流水线入口并经匹配延迟切片贯穿四阶段（Stage1 与 Stage4 都消费它），运行时切换完全靠输入 datatype 控制的 MUX、无任何重配置、无 pipeline 冲刷。运转例子：GEMV kernel 中 per-tile datatype 控制信号存于 HBM、与权重 tile 一起读出并随操作数同步传播到 PE 内所有 XtraMAC——投影层 tile 用 INT4×BF16+BF16（2 lane），注意力层 tile 用 BF16×BF16（2 lane），同一硬件同一流水线连续执行。资源共享收益（Table III/Fig.7）：Config I（INT4×BF16+BF16 与 BF16×BF16）复用算术核与加法单元、加法逻辑 LUT/FF 降 ~50%；Config IV（FP4×BF16 与 BF16×BF16）因 FP4 零填充扩到 BF16 无需指数对齐/舍入调整，映射/算术/加法近完全复用、资源最低；DSP 在所有配置恒为 1、时延恒 4 cycle、II 恒 1。对照 baseline：空间复制（vendor IP 双 datapath）每操作资源 220.0 LUT/310.5 FF/DSP 1（BF16），TATAA 352.0/467.0/4，XtraMAC 142.0/128.3/0.25。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：datatype-select 输入端口 + N 个静态映射子模块 + 匹配延迟切片对齐控制与数据 + 输出组合选择；OpenXtraMAC 仓库 runtime_reconfig/ 提供 4 个双模 MAC（a_bf16_int4_shared、b_int8_bf16_dual、c_fp8_bf16_dual、d_bf16_fp4_dual，mode 位切换，Fmax 267–460 MHz）。使用：混合精度 LLM GEMV 推理——Qwen3-8B-AWQ、Llama-3.1-8B-W8A8、Qwen3-8B-FP8、Llama-3.1-8B-FP8、GPT-oss-20B 的 decode 期 GEMV 由两类 MAC 模式主导（INT4×BF16→BF16 与 FP4×BF16→BF16）；因恒定时延与 II=1，XtraMAC 可作 drop-in 替换现有 GEMV/GEMM 流水线的标量 MAC，无需改流水深度/调度/接口时序。

涉及论文标题：
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA
