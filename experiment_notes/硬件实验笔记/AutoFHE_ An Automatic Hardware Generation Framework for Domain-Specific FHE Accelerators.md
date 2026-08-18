## AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现为 FHE 加速器的 CPE RTL 模板库 + CPE-virtualization 生成架构。CPE 模板库四类：(1) ciphertext arithmetic 模板——并行乘法器/加法器阵列（参数 PoC），做密文加/乘/逻辑；(2) bootstrapping 模板——多项式缩放模块（PoV 向量宽度）、external product 模块（PoE 个 MAC）、FFT/IFFT 模块（BFU/IBFU 个 butterfly 单元）、分解模块（PoD 并行 lane），并内置 bootstrapping unrolling 优化（展开阵列 2^r−1，r 为展开因子：迭代深度 n→n/r、FFT 次数 n→n/r，代价是并发访存增加带宽压力）；(3) key-switching 模板——向量单元 + 累加器做密文×key-switching key 的向量-矩阵乘（参数 PoK）；(4) HMUX 模板——以 bootstrapping 模板实例化（HMUX(a,b,s)=a·s+b·(1−s)，支持密文索引选择）。生成架构 = K 个物理 CPE 池（arithmetic+bootstrapping+key-switching 视为统一 CPE lane，K 可为向量）+ 硬件调度器（离线 GA 调度策略固化、在线确定性编排）。
  - 实验比较：vs Strix（eNPU：8 个 1D CPE、无 unrolling、28nm 1.2 GHz、141.37 mm²；AutoFHE 同预算：16 CPE、unrolling r=2、28nm 1 GHz、139.1 mm²）、MATCHA（37 mm² 14nm、640 GB/s、手工 r=3）、PPGNN（57 mm² 14nm、512 GB/s）；CPU Intel Xeon 6148 @2.5GHz（TFHE 库）、GPU NVIDIA A100（cuFHE、nuFHE）。指标：DeepCNN 20/50/100 层端到端延迟（vs CPU 86.8×、vs GPU 33.2×、vs Strix 2.6×）、加密 XOR 延迟/吞吐（vs CPU 719.2×/362.7×、vs GPU 7.8×/32.5×、vs MATCHA 1.4×/1.7×）、Cora/Citeseer/Pubmed 图聚合查找（vs PPGNN 平均 1.6×）、面积（DC 28nm 综合）、EDP（vs Strix 2.9×）、可扩展性（8→48 CPE）、GA vs round-robin 调度（+12.9%–31.6%）、unrolling 消融（禁用后性能降 39.5%）。
- 硬件平台是什么，配置是什么。
  - RTL 由 Chisel 生成，Synopsys Design Compiler + TSMC 28nm 综合、1 GHz。案例约束：eNPU 面积预算 142 mm²、带宽 300 GB/s；LRA 37 mm²（14nm）、640 GB/s；查找引擎 57 mm²（14nm）、512 GB/s；加密 ALU 200 mm²、640 GB/s。TFHE 参数：n/N/L/k = 500/1024/2/1、630/1024/3/1、592/2048/3/1（80/110/128-bit）。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 自研 cycle-accurate simulator，按 F1 [10]（arXiv 2109.05371）的方法建模微架构行为并捕获计算与数据搬移周期；论文未给模拟器名称与公开链接（AutoFHE 整体未见开源，IEEE Xplore 11617778 仅论文）。验证方式：以 MATCHA 为建模对象校准，仿真结果与其报告数据吻合。
- 模拟器模拟什么的性能，修改了什么。
  - 模拟 TFHE 加速器（CPE 池 + 硬件调度器）在密文负载下的逐周期执行：CPE 内 bootstrapping（多项式缩放→FFT→external product→IFFT→分解）、key-switching、算术与 HMUX 的计算周期 + off-chip 密文/密钥搬移周期，latency = max(计算, 访存)。配套解析模型：面积 = DC 预表征原语（MAC/butterfly）按参数线性组合；能量 = off-chip 访存 + on-chip 访存 + 计算单元 + 片上通信四源聚合。模拟器是否基于开源模拟器修改：论文未明确说明（自研 Python 模拟器）。
- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 模拟器与 AutoFHE 工具链未开源。开源可获取的是对比软件与负载依赖：TFHE 库（CPU baseline，https://github.com/tfhe/tfhe）、cuFHE（https://github.com/vernamlab/cuFHE）、nuFHE（https://github.com/nucypher/nufhe）、Chisel/FIRRTL 工具链（chisel-lang.org）。
  - 模拟原理与全过程：输入 = DSE 输出的硬件配置（CPEmicro：PoV/PoD/BFU/IBFU/PoE/PoC/PoK，物理 CPE 数 K）+ GA 调度策略 + 工作负载（DeepCNN 20/50/100、加密 XOR、Cora/Citeseer/Pubmed 图查找、加密减法）。模拟器按调度策略逐 time step 执行 PE DAG：每个 PE 在分配到的物理 CPE 上逐周期跑模板微架构（bootstrapping：多项式缩放→FFT→external product→IFFT→分解；算术：并行乘加阵列；HMUX：一次 bootstrapping 迭代），并统计 off-chip 密钥/密文搬移周期，端到端 latency = max(计算周期, 访存周期) × Tmax；输出 = 延迟/吞吐/EDP，反馈 DSE 内层与外层更新全局最优；面积由 Design Compiler 28nm 综合给出（原语预表征 + 线性组合），能量按四源聚合。
