## 共享内存空间的 CPU/PIM 并发执行（Concurrent CPU/PIM Execution on Shared Memory Space）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CPU 与 PIM 单元物理共享同一 DRAM 地址空间并同时工作，区别于 UPMEM 式静态分区（PIM 与 CPU 各占独立内存区域，带来内存预留开销与 CPU-PIM 数据搬运）。共享由 OS 管理的逻辑隔离实现：UM-PIM（ISCA'24）的"统一共享内存空间"（CPU 页保持交织、PIM 页非交织共存、双轨页管理、零拷贝卸载，[Web] Semantic Scholar 271646642）；后续 PIM-MMU、ComPASS 等延续。动机（移动端）：1–3B 端侧 LLM（隐私 + 毫秒响应）与后台应用并存，DRAM 成本与板级面积受限，静态分区不可接受。挑战：bank 冲突、命令总线拥塞、CPU-mediated 传输干扰——三者共同导致既有调度无法兼顾 CPU 延迟与 PIM 吞吐。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
移动端并发例子（COSM）：视频会议/浏览器等后台应用与 LLM 推理并发——LLM 权重在模型加载时预组织进 PIM bank 并静态常驻，解码阶段逐 token 执行 attention GEMV/MAC/softmax（PIM_Exec）；KV cache 增长使 CPU-mediated 传输占推理时间 50–60%，与后台应用请求同走 FR-FCFS → 传输突发饿死 CPU（降速 >80%）。COSM 的组合策略（IWE 空闲窗口插入 + 可抢占 PIM_Exec/PIM_Pause + 带宽解耦传输 + tile 重叠）使 PIM 吞吐 +2.8×（较 Chopim）且 CPU 降速 <2.0%——系统级效果 = "PIM 加速 LLM 的同时后台应用几乎无感"。热管理：Command Arbiter 可节流 PIM（Discussion）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：内存管理（OS 逻辑隔离、双轨页管理、HugeTLB 共享地址空间）+ 内存控制器 PIM 调度器（CPU 优先仲裁）+ 接口协议（扩展命令：PIM-ACT、ComPASS）。评估方式：真实手机 trace（Frida 插桩小米 Mi 11 Pro）+ 模拟器回放（Ramulator2 + zsim + 三星 LPDDR5-PIM 参数）。使用：边缘设备 LLM 与常规负载共存、数据中心 CPU-PIM 混合负载；对 GPU/NPU 等异构代理可把其访问视为标准 host 请求隔离 PIM 干扰（未来工作）。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
