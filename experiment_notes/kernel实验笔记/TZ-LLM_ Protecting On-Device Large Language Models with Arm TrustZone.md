## TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文提出两个核心运行时调度机制：(1) Pipelined Parameter Restoration调度：将LLM inference的computation graph扩展为三类restoration operator（allocation/loading/decryption）+ computation operator的DAG pipeline。CPU上实现priority-based greedy scheduler：若computation operator ready则优先执行（最高优先级），否则执行与earliest computation operator关联的restoration operator。alloc/decr operator被切分为micro-operator支持preemptive scheduling（computation就绪时可抢占长restoration micro-operator），减少NPU/CPU idle bubble。Partial parameter caching按topological order缓存早期prefill参数，按reverse order释放。(2) Co-driver NPU job执行路径：将一次secure NPU job拆为REE control plane（scheduling/power/frequency）+ TEE data plane（secure job context/MMIO launch/interrupt completion）。Shadow job机制使REE统一调度REE NN job和TEE secure job，TEE通过initialized-not-issued状态机+monotonic sequence number校验防重放/重排序。NPU world switch通过TZPC（secure MMIO access）+ TZASC（secure memory access）+ GIC（secure interrupt routing）硬件配置完成。实验比较：pipeline scheduling overhead vs critical path lower bound、preemptive scheduling对TTFT的改善、不同cache proportion下TTFT变化、NPU sharing对REE NN应用throughput影响、CMA allocation对REE Geekbench干扰。

- 后端平台是什么，配置是什么。
  Orange Pi 5 Plus (RK3588 SoC)：CPU 4×Cortex-A76 @2.4GHz + 4×Cortex-A55 @1.8GHz，NPU 3-core ~6 TOPS，16GB LPDDR4X。TEE OS基于OpenHarmony TEE系统，REE OS为OpenHarmony v4.1 / Linux v5.10。LLM TA使用4×A76 core + 3-core NPU。

- 评估性能的软件/脚本是什么。修改了什么。
  基于llama.cpp（约1.2K LoC修改实现pipelined restoration，约1K LoC集成NPU data plane driver）+ Rockchip NPU driver v0.9.8（167 LoC shadow job scheduling）。TEE OS扩展约62 LoC（CMA page mapping）+ 约50 LoC（TZASC/TZPC动态配置）。OpenSSL做参数加解密。benchmark：UltraChat/PersonaChat/DroidTask评估LLM serving性能，YOLOv5/MobileNet评估NPU time-sharing对REE NN应用影响，stress-ng模拟memory pressure，Geekbench评估CMA对REE应用干扰。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：Zenodo artifact DOI 10.5281/zenodo.17213486（EuroSys '26 artifact, MIT License, 含prototype源码和复现脚本），arXiv:2511.13717。pipeline/kernel执行流例子（Llama-3-8B 512-token prompt on RK3588）：
  1. 冷启动：LLM TA提取llama.cpp computation graph拓扑顺序→为第一个prefill operator调用extend_allocated→REE Linux CMA返回连续物理内存→REE file system DMA加密参数入此内存（未TZASC-protected, 避免bounce buffer）→TA调用extend_protected→TEE OS扩展TZASC region映射入TA→OpenSSL AES解密参数→computation operator ready→CPU/NPU prefill开始。
  2. Pipeline调度：scheduler维护computation operator和restoration operator的就绪队列→priority policy：CPU computation优先→若computation未就绪则调度与earliest computation关联的restoration→large allocation/decryption被切为micro-operator→computation ready时抢占当前restoration micro-operator。
  3. Co-driver NPU secure job：LLM TA准备secure execution context（command/register sequence, I/O page table, buffers）→向REE driver提交paired shadow job→REE NPU scheduler按统一队列调度→shadow job选中时REE driver smc通知TEE→TEE data plane driver用TZPC阻止REE访问NPU MMIO+GIC路由NPU interrupt到TEE→等待non-secure job完成→TZASC允许NPU访问secure memory→验证sequence number（防重放/重排序）→写MMIO launch NPU job→secure interrupt到达后TEE driver收尾→NPU切回non-secure mode→REE标记shadow job完成继续调度。
  4. 结果：pipeline scheduling距critical path lower bound仅0.01%-9.9% overhead；preemptive scheduling进一步降低TTFT最多16.2%；vs strawman TTFT降低77.1%-91.1%。

