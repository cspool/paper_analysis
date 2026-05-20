## TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现TZ-LLM，一个基于Arm TrustZone TEE保护端侧LLM模型机密性的安全推理系统。核心Serving调度实现包含：(1) Pipelined Parameter Restoration：将LLM computation graph扩展为computation operator前插入allocation/loading/decryption三类restoration operator的pipeline，按拓扑顺序调度。CPU上采用priority-based greedy policy：若CPU computation operator ready则优先执行，否则执行与最早computation operator相关的restoration operator。为减少operator时间错配引起的pipeline bubble，allocation和decryption被切成micro-operator支持preemptive scheduling。配合pipeline-aware extend/shrink接口利用LLM参数first-in-last-out模式保证TZASC连续secure memory。(2) Partial Parameter Caching：在REE memory pressure允许时缓存早期prefill operator的参数，inference后按reverse topological order逐步释放，消除pipeline起始bubble而不长期锁住完整模型内存。(3) Co-driver NPU设计：REE driver负责scheduling/power/frequency等control plane；TEE user-mode data plane driver只负责secure job context/MMIO launch/interrupt completion，通过shadow job机制使REE scheduler将secure job放入统一队列，TEE通过initialized/not-issued状态和monotonic sequence number防重放和重排序攻击。NPU world switch通过TZPC/TZASC/GIC配置实现。(4) 工程机制：TA多线程shadow thread、framework initialized-state checkpoint。实验比较：TTFT、decoding speed、pipeline critical path overhead、不同cache proportion下TTFT、NPU sharing throughput、CMA allocation对REE应用的干扰。对比baseline：REE-LLM-Memory（无保护全部参数预加载）、REE-LLM-Flash（无保护pipelined restoration不解密）、Strawman（TEE cold start无pipeline无NPU）。

- 硬件平台是什么，配置是什么。
  Orange Pi 5 Plus开发板，SoC为Rockchip RK3588，包含4×Cortex-A76 @2.4GHz + 4×Cortex-A55 @1.8GHz，16GB LPDDR4X，1TB NVMe SSD，3-core NPU（峰值~6 TOPS）。LLM TA使用4个Cortex-A76 CPU core和全部3个NPU core。为触发CMA memory migration，用stress-ng模拟REE memory pressure：四个模型对应压力分别为13GB/11GB/10GB/6GB，stress threads与LLM threads pin到不同CPU core。

- 开源Serving框架是什么。修改了什么。
  基于llama.cpp作为LLM inference backend。修改包括：(1) llama.cpp扩展约1.2K LoC实现pipelined restoration（computation graph提取、restoration operator插入、priority-based + preemptive scheduling）；(2) 扩展约1K LoC集成NPU data plane driver；(3) 使用OpenSSL做参数解密。REE侧：OpenHarmony v4.1 / Linux v5.10，扩展约364 LoC（NPU driver 167 LoC shadow job scheduling + TZ driver 197 LoC CMA allocation/deallocation）。TEE OS原本约17K LoC，扩展约62 LoC支持CMA page memory mapping，约50 LoC支持动态TZASC/TZPC配置。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：Zenodo artifact DOI 10.5281/zenodo.17213486（EuroSys '26 artifact，包含prototype system源码和复现实验脚本，MIT License），Hugging Face papers页面可交叉确认arXiv:2511.13717。论文Artifact Appendix明确写出artifact内容。以TZ-LLM serving流程（Llama-3-8B 512-token prompt）为例：
  1. 启动请求→LLM TA根据llama.cpp提取的computation graph拓扑顺序→为最早prefill operator所需参数调用extend_allocated→从REE Linux CMA获取连续物理内存。
  2. REE file system异步将加密参数读入尚未TZASC-protected的allocated memory（避免bounce buffer）→TA调用extend_protected→TEE OS扩展TZASC region→映射内存入TA→解密参数。
  3. 第一个computation operator参数ready→开始CPU/NPU prefill，同时后续operator的allocation/I/O/decryption在后台推进。
  4. CPU computation ready时→scheduler抢占长allocation/decryption micro-operator优先执行computation→减少NPU/CPU等待。
  5. Decoding阶段：TA使用co-driver NPU→REE scheduler将secure shadow job排入统一队列→选中后smc通知TEE data plane driver→TEE配置TZPC+TZASC+GIC→等待non-secure job完成→验证sequence number→MMIO launch secure NPU job→interrupt完成→切回non-secure mode→REE标记完成继续调度。
  6. 效果：vs strawman TTFT降低77.1%-91.1%；vs REE-LLM-Flash TTFT overhead 5.2%-28.3%；decoding overhead 1.3%-4.9% vs REE-LLM。NPU time-sharing对REE NN应用额外slowdown最高3.8%。CMA对并发Geekbench性能下降最高6.7%。

