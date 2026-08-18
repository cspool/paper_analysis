## Ring Buffer（环形缓冲区：AXLE 的 metadata/payload 双环）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ring buffer 是定长数组组成的循环 FIFO，以 head/tail 两个索引管理生产者写入与消费者读取：写满则回绕覆盖（需流控防覆盖未消费数据）。AXLE 把主机本地 DMA 区域组织成两个独立的 ring buffer——payload 环（存部分结果数据，slot 默认 32B）与 metadata 环（存每条 payload 的描述信息与 payload slot ID），主机只轮询 metadata 环的尾指针即可感知新结果到达，实现跨设备的完全异步生产者-消费者通信。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（单次背流）：设备 DMA 写 payload 环 slot i + metadata 环 slot j（栅栏保证先 payload 后 metadata tail 更新）→ 主机轮询 routine 发现 metadata tail 前进，把 head..tail-1 的 metadata 搬入 ready pool（不搬 payload，解耦消费）→ 主机调度器按 metadata 记录的 slot ID 从 payload 环读取数据执行下游任务 → 消费后主机经 CXL.mem 把两个环的新 head 索引回传设备（流控）→ 设备按 head 释放 slot。OoO 流式下 payload 环 gap-aware 消费：非连续 slot 可先消费，head 只推进到最大连续已消费前缀。索引回绕（wraparound）与单调递增是必须维护的不变量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：主机内存预 pin 区域（cache-bypass 防缓存陈旧）+ head/tail 索引 + 内存栅栏；参考 DPDK rte_ring（lib/ring/rte_ring.h）等生产实现。使用方式：跨设备异步通信、DMA 结果缓冲、生产者-消费者解耦；AXLE 用它替代远程 mailbox 轮询（轮询点从远端移到本地单地址）。工程要点：固定大小必须配流控（AXLE 用 CXL.mem store 回传 head）；slot 大小与 DMA slot/SF 对齐（32B/64B）。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
