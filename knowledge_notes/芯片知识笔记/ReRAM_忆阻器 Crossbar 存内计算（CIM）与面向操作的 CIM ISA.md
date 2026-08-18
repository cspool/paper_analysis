## ReRAM/忆阻器 Crossbar 存内计算（CIM）与面向操作的 CIM ISA

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
基于忆阻器/ReRAM 交叉开关阵列（crossbar）的存内计算（Computing-In-Memory）：交叉点阵列用欧姆定律（i=v·g）与基尔霍夫电流定律在模拟域原生完成矩阵向量乘（MVM）——输入向量以电压施加到字线（WL），每条位线（BL）上汇聚的电流即输入电压与阵列电导的点积，理想情况把 MVM 从 O(n²) 降到 O(1) 并消除数据搬运。在此计算模型上，一部分工作进一步提出面向操作的 ISA，在指令级显式暴露算术 kernel（如 MVM、卷积）直接在交叉阵列上执行：代表包括 ISAAC（ISCA'16，卷积加速 + 原位模拟算术）、PUMA（ASPLOS'19，流水线取指/译码/执行 + 新增存内计算指令）、PRIME（可配置存储/计算分区）、ReVAMP（VLIW）、MNEMOSENE（tile 级 ISA：RS/WD/FS/DoA/DoS/CS/DoR 等行列选择与功能指令 + 编译器把高层 kernel 翻译为存内指令）；清华专利 CN113010213B 亦属此类。DS-ISA 论文引用该路线作为对照：这类 ISA 是 operation-centric（操作中心，显式算术运算），而 DS-ISA 是 dynamics-centric（动力学中心，配置拓扑/边界/时长后计算从集体物理演化中涌现）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
芯片级运转流程（以 ReRAM crossbar 加速神经网络层为例）：权重矩阵 W 逐元素编程进交叉点电导（Write 操作，受限于器件写噪声/耐久）→ 激活向量经 DAC 转模拟电压施加到 WL → 每列 BL 电流 = Σ g_ij·v_j 即 Wx 的每个分量（乘在器件上、加在 BL 汇流上完成）→ 经采样保持 + ADC 转回数字 → 后续数字单元做激活/累加。芯片设计关注点：ADC/DAC 转换开销（面积/能耗大头）、器件非理想性（电导漂移、IR-drop、编程变差扭曲 MVM 精度）、阵列尺寸与精度/并行度权衡、交叉阵列与其他数字逻辑（如 PUMA 的取指译码流水）的片上集成与封装。DS-ISA 论文的对照视角：CIM 芯片把"算术操作"搬到阵列里执行（显式 MVM kernel），而 DSU 芯片把"演化计算"整体交给模拟动力学（节点电压 + 可编程电导网络按哈密顿量梯度弛豫），ISA 层因此分别是操作序列抽象 vs 配置-触发抽象。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：器件用忆阻器（HfOx/TaOx 等）或 ReRAM 交叉阵列；系统演示如 Wan et al., Nature 608 (2022) 的 ReRAM CIM 芯片；模拟/数字混合接口（DAC/ADC）与 weight-stationary 数据流为主（权重驻留阵列、激活流经）。使用方式：矩阵乘密集的 DNN 层（FC/卷积）加速；编程上依赖带存内指令的 ISA（PUMA/MNEMOSENE）或编译器自动映射 kernel。与 SRAM-PIM 的关系：同属 CIM 家族但器件/精度/容量不同（SRAM 宏容量 KB 级、数字域精度高，见硬件架构库"SRAM-PIM"条目）；DS-ISA 论文将其与神经形态（事件驱动 spike ISA）并列，论证"配置型且动力学中心"的 ISA 是第三种范式。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
