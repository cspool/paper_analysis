## Runtime Packet Manager（RPM，运行时包管理器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RPM 是 MoE-Hub 放在生产者 GPU hub egress 出口的硬件模块，用于治理 MoE 路由固有的细粒度、失序、突发 token 流量，把它整形为 interconnect 友好的结构化数据流，从而提升多 GPU 互连带宽利用率。动机（Insight-2 生产者侧）：MoE 路由的随机性使每个生产者瞬间产生大量指向任意 GPU 的细粒度远程写请求，形成不规则流量——对某个消费者 GPU↔switch 链路的突发会造成拥塞、反压回生产者、拖累到其他消费者的传输，降低整体带宽利用率。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
RPM 微架构分前端与后端：(1) **前端 Remote Write Buffer Pool**——一组全相联 SRAM 分区，每个活跃 peer GPU 一个分区（peer 集合在任务初始化时固定、受物理拓扑限制，如当前系统 8-16 个 peer），实现 per-link 流量隔离。远程 st. 请求按 cache-line 对齐地址索引、st.rowsp 请求按 (RowID, RowOffset) 元组索引（把逻辑行当"地址空间"做包合并）；若找到匹配条目且优先级 flag 一致，新请求按有效性掩码合并进既有条目，生成 interconnect 友好的 128B 包，最小化协议开销。(2) **后端 Packet Scheduler**——仲裁哪些合并请求何时上线，策略多维：congestion-aware（round-robin 轮询各目标 GPU 分区，平滑流量防突发、保持链路负载均衡；优先发送满掩码条目=最大粒度合并；timer 旁路机制防止劣质合并条目无限滞留保证前向推进）+ consumer-aware（每个目标分区内两级优先级：先发高优先级 st.rowsp、再发 st.rowsp.nop；高优先级内按最小 RowID 先发，保证单个 token 整行连续到达，让消费者专家计算能对已完成行提前开始，尤其利于长序列隐藏通信延迟）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 hub egress 的 SRAM 缓冲池 + 调度器硬件（全相联分区、合并掩码、round-robin/timer 逻辑），随 MoE-Hub 整体以 TSMC 7nm 0.49 mm² 落地。对软件透明：生产者只需发 st.rowsp（可用 RowID 控制期望的传输顺序、用 .nop 标记非关键路径），包合并、排序、拥塞规避全部由 RPM 完成。消融数据（MH-PKT，routing→GEMM1 窗口）：RPM 平均加速 1.13×，收益随序列长度增长（通信量增大时无包管理会使传输更突发、放大互连排队，尤其对专家较小的模型如 Qwen-2 影响显著）。

涉及论文标题：
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
