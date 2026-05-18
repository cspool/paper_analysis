## TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

- baseline方法是什么？
  Baseline有三个层次：(1) REE-LLM-Memory：未修改llama.cpp在REE中运行，全部参数预加载在内存中，代表无保护、内存低效但接近理论最优性能的baseline；(2) REE-LLM-Flash：未修改llama.cpp在REE中运行，inference start时用pipelined restoration加载参数（buddy system allocation，不解密），代表实用但无保护的REE baseline；(3) Strawman：每次请求在TEE中冷启动、动态扩展secure memory、加载/解密参数、CPU-only计算，提供安全性和内存效率但缺少pipeline和NPU支持。Strawman是论文最直接对比的baseline。

  全栈执行例子（Llama-3-8B 512-token prompt, Strawman TEE cold start on RK3588）：
  - 算法层：llama.cpp decoder-only transformer（RMSNorm→Grouped Query Attention→FFN with SiLU gate），8-bit quantized weights，CPU-only inference。
  - 系统框架/Serving层：每次请求→启动llama.cpp→初始化metadata和tokenizer→CMA分配7.9GB连续物理内存（~4.182s）→从NVMe SSD加载加密模型参数（~4.054s）→OpenSSL AES解密（~0.892s）→CPU prefill 512 tokens（~164s量级）→decode。总TTFT包含约11.6s纯restoration overhead。参数在TEE secure memory中，但动态CMA allocation完全暴露在critical path上。
  - 编译框架层：论文未明确说明（llama.cpp使用默认C++编译路径，无定制编译器）。
  - kernel调度层：无pipeline——串行执行allocation→loading→decryption→computation，无operator重叠，无preemptive scheduling。CPU仅做computation无NPU参与。TZASC/TZPC在启动时一次性配置，无动态world switch。
  - 硬件架构层：RK3588 (4×A76 @2.4GHz + 4×A55 @1.8GHz, 16GB LPDDR4X, 3-core NPU ~6 TOPS, NVMe SSD)。Strawman中NPU闲置（CPU-only），TEE和REE无法共享NPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **方法概述**：TZ-LLM提出两个核心机制：(1) Pipelined Parameter Restoration：将LLM computation graph扩展为computation operator前插入allocation/loading/decryption三类restoration operator的DAG pipeline，配合priority-based greedy + preemptive micro-operator scheduling和partial parameter caching，将restoration latency隐藏到computation latency下；(2) Co-driver NPU：将完整NPU driver拆为REE control plane（scheduling/power/frequency）和TEE data plane（secure job context/MMIO launch/interrupt completion），通过shadow job+TZPC/TZASC/GIC完成安全NPU time-sharing。

  **缺陷→方法映射**：

  **缺陷1：Strawman TEE cold start中CMA allocation（~4.182s）+ flash I/O（~4.054s）+ decryption（~0.892s）完全串行暴露在TTFT critical path上，使TTFT增加~11.6s**
  → 方法：Pipelined Restoration将restoration与computation重叠。按computation graph拓扑顺序，为最早operator先分配/加载/解密参数→立即开始prefill→后续operator的restoration在后台推进。实验：vs strawman TTFT降低77.1%-91.1%（固定prompt length），real-world benchmark上降低76.1%-90.9%。Pipeline scheduling距critical path lower bound仅0.01%-9.9% overhead。

  **缺陷2：Strawman CPU-only prefill极慢（512-token Llama-3-8B ~164s）且NPU闲置**
  → 方法：Co-driver NPU使TEE安全使用NPU加速。REE掌握control plane（统一调度、电源/频率管理），TEE仅执行最小data plane（secure context验证、MMIO launch、interrupt处理）。Shadow job使secure job融入REE调度队列。Decoding阶段vs strawman提升0.9%-23.2%（NPU acceleration），vs REE-LLM仅1.3%-4.9%额外overhead（来自TEE/REE driver通信）。

  **缺陷3：Static secure memory partitioning（warm start方案）需长期预留约8GB参数内存，在24GB以内移动设备上内存效率差**
  → 方法：Partial Parameter Caching仅在REE memory pressure允许时缓存早期prefill operator参数（而非全部模型），inference后按reverse topological order释放。Pipeline-aware extend/shrink接口利用LLM参数first-in-last-out模式保证TZASC连续secure memory。实验：在达到cache比例阈值前TTFT近似线性下降；不使用时自动释放，不长期锁住完整模型内存。

  **缺陷4：NPU baseline中TEE和REE各放完整NPU driver会导致Rockchip NPU detach-attach ~32ms world-switch overhead，且将约60K LoC驱动代码纳入TEE TCB**
  → 方法：Co-driver将data plane缩小为TEE user-mode driver（仅secure job context/MMIO launch/interrupt completion），control plane留在REE。TZPC/TZASC/GIC硬件配置完成NPU secure mode切换（无driver detach-attach）。NPU time-sharing对REE NN应用额外slowdown最高3.8%，对LLM额外slowdown最高3.0%。TEE TCB仅增加约112 LoC（CMA mapping + TZASC/TZPC配置），不引入完整NPU driver（~60K LoC）。

  **论文方法全栈执行例子（Llama-3-8B 512-token prompt, TZ-LLM on RK3588）**：
  - 算法层：llama.cpp decoder-only transformer（同baseline），8-bit quantized weights，OpenSSL AES参数解密。计算本身无算法修改（不涉及稀疏/量化/蒸馏等算法pipeline优化）。
  - 系统框架/Serving层：TZ-LLM TA在TEE中运行→提取computation graph→按拓扑顺序调度restoration+computation operators→REE file system异步I/O直接写入allocated未保护内存→TEE OS动态扩展TZASC region映射入TA→OpenSSL解密参数→computation ready即开始CPU/NPU prefill→后续operator restoration在后台重叠执行。Partial parameter caching缓存早期prefill参数（内存压力允许时），用完按reverse order释放。
  - 编译框架层：论文未明确说明（llama.cpp使用默认C++编译路径，无定制编译器）。
  - kernel调度层：Priority-based greedy scheduler维护restoration operator队列→computation优先→否则执行earliest-computation关联restoration→alloc/decrypt切为micro-operator可抢占→computation就绪即抢占→消除pipeline bubble。Co-driver NPU路径：LLM TA准备secure execution context→提交shadow job→REE scheduler选中→smc通知TEE data plane driver→TZPC阻止REE访问NPU MMIO+GIC路由interrupt到TEE→等待non-secure job完成→TZASC允许NPU访问secure memory→sequence number验证→MMIO launch NPU job→interrupt完成→切回non-secure mode。
  - 硬件架构层：RK3588 SoC (Orange Pi 5 Plus)：4×A76负责CPU computation和scheduling，3-core NPU (~6 TOPS)通过co-driver安全执行矩阵运算。TZASC配置secure memory region（利用连续物理内存特性），TZPC配置NPU MMIO secure access，GIC路由NPU interrupt到TEE。CMA从REE Linux分配连续物理内存→TZASC保护后成为secure memory。NVMe SSD存储加密模型参数。stress-ng模拟memory pressure（Llama-3-8B对应6GB压力），stress threads与LLM threads pin到不同core。

  **关键trade-off**：pipelined restoration依赖LLM参数访问顺序足够确定（对MoE/early-exit等非确定workload可能预取未使用参数）。TZASC要求连续物理内存→仍受CMA allocation和内存碎片影响（虽然overhead变为transient可与I/O/compute重叠）。Co-driver依赖具体NPU driver可分离出小data plane（已在Rockchip实现，仅调查了Qualcomm开源NPU driver可行性，未跨平台完整验证）。安全模型不覆盖物理DRAM攻击/side channel/cryptographic attack/DoS。参数tensor size与secure NPU job execution time向REE泄漏模型结构级信息。TEE内解密会引入额外CPU overhead（vs REE-LLM-Flash在TTFT上5.2%-28.3% overhead）。
