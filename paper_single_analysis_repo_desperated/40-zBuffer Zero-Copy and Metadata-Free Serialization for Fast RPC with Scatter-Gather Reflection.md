论文标题：zBuffer: Zero-Copy and Metadata-Free Serialization for Fast RPC with Scatter-Gather Reflection

远程例程调用RPC优化。

本地条目说明：
    - 本地编号：paper_2026 第 40 篇
    - 本地 PDF：paper_2026/40-zBuffer Zero-Copy and Metadata-Free Serialization for Fast RPC with Scatter-Gather Reflection.pdf
    - 发表信息：PPoPP 2026，DOI：https://doi.org/10.1145/3774934.3786426

原文与开源仓库确认：
    - 原文状态：已找到本地 PDF 全文，共 13 页，正文包含 Abstract、Introduction、Background and Motivation、zBuffer Design、zRPC、Evaluation、Discussion and Related Work、Conclusion 等完整章节。
    - 开源状态：已找到明确开源仓库路径
    - 链接：https://github.com/alibaba/PhotonLibOS/tree/main/rpc
    - 说明：论文 Introduction 明确写道作者已开源 TCP edition 的 zBuffer/zRPC，路径为 Alibaba PhotonLibOS 的 rpc 目录；论文同时声明 zBuffer/zRPC 已在 Alibaba 内部生产系统部署。需要注意的是，论文评测中的 zRPC 使用 RoCE/RDMA send/recv、OFED libibverbs 5.4 和 Mellanox ConnectX-5 NIC，而公开说明是 TCP edition，因此可确认存在官方公开实现入口，但不能仅凭论文文本确认公开仓库完整覆盖论文中所有 RDMA/RoCE 评测路径。

1、论文工作：
    - 论文要解决的核心问题：
      zBuffer 解决的是现代高性能 RPC 中 serialization / deserialization 和网络发送路径的 CPU memory-copy tax。传统 RPC 会在发送端构造对象 metadata、把对象中分散字段 copy/coalesce 到连续用户态 buffer，再经过网络栈发送；接收端再把连续 payload 解析、copy 回目标对象结构。即使 DPDK/RDMA 这类 kernel-bypass dataplane 能去掉用户态/内核态之间的拷贝，传统序列化库仍然要做对象字段 coalescing、metadata 生成与解析，以及接收端对象重建。随着数据中心 RTT 降到微秒级、NIC 带宽提升到 800Gbps 并继续增长，这些 CPU 拷贝和 metadata 处理不再是小开销，而成为 RPC 延迟和吞吐瓶颈。
    - 瓶颈来源：
      主要瓶颈来自内存访问和数据搬运，其次来自 metadata construction / parsing 的 CPU 开销。论文把典型 send-receive transaction 分解为 sender 侧 metadata 构造、data coalescing、user-to-kernel copy，receiver 侧 kernel-to-user copy 和 copy to target data structure；一个 RPC 至少包含 request 和 response 两个 transaction，因此 copy 次数会被放大。即使使用 NIC scatter-gather 能让硬件从多个非连续 buffer 聚合 packet，receiver 仍然需要知道对象字段布局；如果像 Cornflakes 一样仍需携带 object header metadata，复杂对象的 metadata 构造和解析仍会消耗 CPU 和网络带宽。
    - 论文的主要贡献：
      第一，提出 zBuffer，一个 zero-copy、metadata-free 的 C++ serialization library，核心机制是 scatter-gather reflection：用 C++ compile-time static reflection 获得对象字段顺序、类型和边界，用 NIC scatter-gather I/O 负责把非连续字段聚合到网络包中。第二，设计 sg_array 抽象，把每个字段表示成 pointer + length 的 sg descriptor，serialization 只构造 descriptor table，不把字段 copy 到连续 buffer。第三，提出 zRPC，把 zBuffer 集成进 RPC stack，并通过 message organization、sub-packet indexing 和 header-data separation 消除网络传输路径上的额外 copy。第四，在 serialization microbenchmark、eRPC/Cornflakes 对比、Raft 和 Masstree KV 等应用中评估，展示 zBuffer/zRPC 对复杂对象、较大 RPC payload 和多包传输更有优势。
    - 论文所处背景：
      背景是 cloud microservices、HPC、distributed data stores、network file systems、LLM serving 等场景中广泛使用 RPC，而网络不再一定是主要瓶颈。论文引用 Google Cloud 的 RPC fleet CPU cycle 占比约 7.1%、Google Protobuf serialization 占 datacenter CPU usage 约 5%、Meta microservices serialization 占 CPU cycles 约 6.7% 作为动机。目标硬件是 commodity Ethernet / RDMA NIC，而不是新增专用 serialization accelerator；目标软件语言主要是 C++，因为 C++ 在云和 HPC 中使用广、支持模板元编程和编译期优化。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：
      Protobuf、FlatBuffers、Thrift、Cap'n Proto 等软件序列化库虽然格式和 API 不同，但通常仍需编码/解码、metadata 或 header 处理，以及把分散对象字段组织成可传输表示。它们的开销随字段大小增长明显，因为数据 copy 量变大。Cornflakes 这类 scatter-gather coalescing 方案使用 NIC gather 减少数据拷贝，但仍要构造 object header 来描述字段 offset、size 和嵌套结构；对象越复杂，metadata 越大，serialization 和 deserialization 时读取/解析 metadata 的 CPU 开销越高。eRPC 作为高性能 RPC 系统提供 zero-copy networking，但它本身使用连续 message buffer，并不解决对象序列化，也没有为多包场景中 header 插入导致的 payload 不连续问题提供 metadata-free serialization 语义。
    - 论文的设计方法：
      zBuffer 的设计可以概括为“编译期对象结构枚举 + 运行期 sg_array descriptor + NIC scatter-gather 聚合”。开发者在 C++ struct 中用 PROCESS_FIELDS 宏注册需要序列化的字段；宏展开为 templated process_fields 方法，并通过 variadic template 递归调用 reduce。编译器在编译期得到字段列表、类型信息和处理顺序，再通过 template specialization / type traits 为 string、array、nested object 等类型生成对应 process_field 路径。运行时 Serializer 遍历对象字段，把非空字段的地址和长度 push 到 sg_array；sg_array 随后直接提交给网络栈 / NIC。Deserializer 在接收端根据相同的字段顺序从 receive buffer 头部开始推进 offset，并把对象内的 pointer 改写为 receiver 地址空间中的正确位置，从而实现 in-place deserialization。
    - 方法如何对冲 Baseline 缺陷：
      对传统 serialization baseline，zBuffer 不再把字段 copy 到连续 serialization object，也不再构造单独 object header；对象本身的字段顺序和边界由编译期生成代码隐式编码。对 Cornflakes，zBuffer 避免了复杂对象随嵌套深度增长而膨胀的 metadata：论文报告 Tree-5 对象中 Cornflakes metadata 为 1256B，而 zBuffer 只需 512B 的对象本体表示；Tree-5 下 zBuffer serialization/deserialization 比 Cornflakes 快约 7 倍，CPU cycles 仅约为 Cornflakes 的 13.8%。对 eRPC，zRPC 在 eRPC networking 基础上增加 zBuffer API、sg_array message organization 和 header-data separation，使 RPC payload 不需要因 packet header 或多包分片而重新 copy 到连续 buffer。
    - 关键 trade-off：
      第一个 trade-off 是编程模型上需要开发者用 PROCESS_FIELDS 显式注册字段，或者为其他 IDL 写转换工具；这降低了运行期 metadata 开销，但把结构描述责任推到编译期和代码生成路径。第二，zBuffer 对 C++ 模板元编程和宏依赖较强，跨语言时需要通过 C ABI 或 IDL 转换接入。第三，zero-copy 依赖对象字段在发送期间保持可访问、并且 buffer 需要适配 NIC 注册和 scatter-gather 能力；当 disjoint memory buffer 数超过 NIC 单个 work request 能力时，zRPC 仍会选择把数据 coalesce 到连续 buffer，因为一次带 copy 的 work request 可能快于多个小 work request。第四，memory alignment 优化要求 serialization 顺序对需要对齐的字段优先处理，这可能改变纯应用声明顺序，但换来 direct I/O / SIMD 等场景下避免额外 copy。

3、论文实现：
    - Baseline 如何实现：
      serialization baseline 包括 Protobuf、FlatBuffers 和 Cornflakes。Protobuf/FlatBuffers 使用 echo application：client 发送 serialized message，server deserialize 后 reserialize 并返回；single string field 用于测最小 serialization overhead。Cornflakes 对比使用 Single bytes field 和不同深度 binary tree 对象，比较 metadata size、serialization/deserialization time 和 CPU cycles。RPC baseline 包括 eRPC + FlatBuffers、eRPC + Protobuf，以及 Cornflakes RPC；eRPC baseline 在同一 RPC workload 下由 FlatBuffers / Protobuf 提供 serialization layer。真实应用 baseline 包括 LibRaft 上的 replicated in-memory KV store，以及 Masstree KV/YCSB 场景中的 eRPC + FB / eRPC + PB。
    - 新设计如何实现：
      zBuffer 实现为 C++ serialization library，核心数据结构是 sg 和 sg_array：sg 保存 sg_base 指针和 sg_len 长度，sg_array 保存多个 sg 以及 begin/end/capacity。Serializer 生成 sg_array 时先处理需要 memory alignment 的字段，再处理不需要 alignment 的字段，最后把对象本身作为一个 sg 放入数组；nested data 会在每层递归执行相同逻辑。Deserializer 从 receive buffer 尾部定位对象本体，因为对象被序列化到 payload 的末端；随后按 serialization 相同顺序遍历字段，用 field length 和当前 offset 修复 pointer。zRPC 基于 eRPC networking stack 扩展 API：init_layout 根据 sg_array 生成 message layout，enqueue_request 将 sg_array 提交给 NIC DMA engine。zRPC 使用 RDMA send/recv 做 packet I/O，基于 RoCE 和 OFED libibverbs 5.4，把 sg_array 转换为 RDMA API 使用的 struct ibv_sge* sg_list。
    - zRPC message organization 与 header-data separation：
      zRPC 让一个 sg_array 表示完整 message，其中至少包含 packet header 和 message data 两类 sg。消息大于 MTU 时，一个 message 会拆成多个 packet；为避免每个 packet 查找 relevant sg 的开销，zRPC 引入 index array，每个 entry 标记 sub-packet boundary，包括所在 sg 和 sg 内 offset。发送第 j 个 sub-packet 时读取 I_j 和 I_{j+1} 即可确定它跨越哪些 sg 以及起止 offset。header-data separation 针对多包接收路径：传统 receive ring buffer 把 header 和 data 放在一起，若一个字段跨 packet，中间插入 header 会破坏连续性，导致额外 copy。zRPC 分配 header ring buffer 和 data ring buffer，共享 index；RDMA receive 时用两个 sge 分别指向 packet header ring 和 data ring，使 payload 在 data ring 中连续落地。
    - 实验 / 实现平台：
      实验运行在 CloudLab d6515 cluster。每台服务器有两颗 32-core AMD EPYC ROME 7452 2.35GHz CPU，关闭 C-States，运行 Ubuntu 20.04 / Linux 5.04；服务器通过双端口 Mellanox ConnectX-5 100Gbps NIC 和 2x100Gbps Dell Z9264F-ON switch 连接，MTU 为 4200B。网络实现使用 RoCE，RDMA send/recv，OFED libibverbs 5.4。论文没有把评测写成专用硬件模拟，而是在 commodity RDMA-capable cluster 上测 serialization、RPC latency/throughput 和应用性能。
    - 关键实验设置与指标：
      serialization 对比中，zBuffer 对 1KB string 的 serialize+deserialize 总时间约 25ns，FlatBuffers 约 0.34us，Protobuf 约 0.57us；到 64KB string 时，FlatBuffers/Protobuf 相对 zBuffer 分别约慢 168x / 340x。与 Cornflakes 对比中，Tree-5 对象 zBuffer 比 Cornflakes 快约 7x；CPU cycles 从 Cornflakes 的 5271 cycles 降到 zBuffer 的 747 cycles。end-to-end RPC 中，小 RPC 下 zRPC 64B P99 latency 为 5.3us，2KB 为 6.5us；1MB RPC 时 zRPC 相对 eRPC+FB / eRPC+PB 的 large RPC median latency speedup 为 3.3x / 4.2x。吞吐上，512B request 下 zRPC 为 1.68Mrps，相比 eRPC+FB 的 1.45Mrps 和 eRPC+PB 的 1.21Mrps 分别提升 15.8% 和 38.8%；32KB request 下 zRPC 仍有 0.4Mrps，约为 eRPC+FB 的 2x、eRPC+PB 的 2.5x。Masstree KV 使用 YCSB-C 生成 100 万个 23-24B keys，不同 value size，server/client 各 8 threads，每个 client thread 16 concurrent requests，运行 60s；16KB value size 下 zRPC 相比 eRPC+FB / eRPC+PB 降低 P99 latency 24.2% / 34.8%，吞吐提升 61.9%。

4、pipeline/kernel 解析：
    - 新 pipeline/kernel 是什么：
      论文的核心新执行路径不是单个 kernel，而是一条 scatter-gather reflection serialization + zRPC zero-copy transmission pipeline。它由四个关键部分组成：第一，compile-time static reflection pipeline，用 PROCESS_FIELDS、variadic template recursion 和 type specialization 在编译期生成固定字段遍历路径；第二，Serializer / Deserializer pipeline，把对象转换为 sg_array，或从 receive buffer 原地恢复对象 pointer；第三，NIC scatter-gather DMA pipeline，把 sg_array 映射到 RDMA sg_list，使 NIC 从分散字段直接 gather 到网络发送路径；第四，zRPC header-data separated receive pipeline，用双 ring buffer 和 packet index 让 packet header 与 payload 分离落地，避免多包接收时字段被 header 打断后再 copy。
    - 一个 RPC request 如何流过 zBuffer/zRPC：
      假设 client 要发送一个 Msg 对象，其中有 int id、string val1、string val2，并且 PROCESS_FIELDS(val1, val2) 已在类型中注册。编译期，宏和模板已经生成 process_fields / reduce / process_field 路径；运行时，应用填好 Msg 后调用 Serializer。Serializer 先创建空 sg_array，按 alignment 策略遍历字段；val1 非空则 push 一个 sg，sg_base 指向 val1 的实际 buffer，sg_len 为 val1 长度；val2 同理。随后 Serializer 把 Msg 对象本身也作为一个 sg push 到 sg_array，因为对象本体中保存了长度、整数等可用于接收端恢复的信息。整个过程没有把 val1/val2 拷贝到新的连续 serialization buffer。
    - 发送端 pipeline：
      zRPC 的 init_layout 接收 sg_array 后生成 message layout；如果 message 超过 MTU，系统用 index array 记录每个 sub-packet 的边界。enqueue_request 把对应 sg_array 转换成 RDMA sg_list，每个 ibv_sge 指向一个 NIC-registered buffer。NIC DMA engine 根据 sg_list 从不同虚拟地址读取字段数据，并在硬件侧完成 gather，形成 wire 上的 packet。若 packet 需要 header，header sg 和 data sg 会按 zRPC message organization 一起提交；对于多包 message，第 j 个 packet 通过 I_j / I_{j+1} 快速定位应发送的 sg 区间和 sg 内 offset，避免逐字段扫描。
    - 接收端 pipeline：
      接收侧预先 post receive WQE，其中第一个 sge 指向 header ring buffer，长度为 zRPC packet header 的 72B；第二个 sge 指向 data ring buffer，长度为 MTU 减 header 长度。RDMA 到达时，header 和 payload 被 NIC 分别写入两个 ring。这样，如果某个字段跨越多个 packet，它的 payload 在 data ring 中仍然保持连续，不会被 packet header 插入打断。zRPC 收齐 message 后把连续 receive buffer 交给 zBuffer Deserializer；Deserializer 从 buffer 尾部取得对象本体指针，再按编译期相同字段顺序从 buffer 起始 offset 开始修复每个 pointer。整数等固定字段已经在对象本体中，可直接访问；string / bytes / nested field 的 pointer 被改写到 receiver buffer 内对应位置。
    - memory alignment 优化路径：
      如果某个字段后续要用于 direct I/O 或 SIMD，需要 block-size alignment。传统 serialization 即使 receive buffer 起始地址对齐，也可能因为前面字段长度不是 block-size 倍数，导致该字段在 buffer 内 offset 不对齐，只能额外 copy 到 aligned memory。zBuffer 在 serialization 时区分 aligned field 和 non-aligned field，优先把需要对齐的字段 push 到 sg_array；Deserializer 按相同顺序恢复 pointer，从而保持这些字段在 receive buffer 中也满足 alignment 要求。这一机制把潜在的后处理 copy 前移为字段排序约束，仍保持 zero-copy。
    - 与传统 pipeline/kernel 的区别：
      传统 RPC serialization pipeline 的核心动作是“构造 metadata + copy scattered object fields 到连续 buffer + 网络发送 + 解析 metadata + copy 回对象”。Cornflakes 的 pipeline 消掉部分数据 coalescing copy，但仍保留 object header metadata，并在复杂对象中付出更大的 metadata 构造/解析成本。zBuffer/zRPC 的 pipeline 则把对象布局信息隐式嵌入编译期代码，把运行期传输单位改成 sg_array，让 NIC 执行数据聚合，让接收端按固定代码路径修复 pointer。它的关键变化不是让 CPU 更快地 copy，而是尽量让 CPU 不再 copy、不再构造 runtime metadata，并且让网络接收 buffer 布局配合对象恢复语义。
    - 边界与限制：
      zBuffer 适合字段可由 pointer + length 表达、对象布局能在编译期注册、buffer 生命周期可被 RPC runtime 控制的 C++ 系统。它不自动解决所有 IDL 生态问题；论文在 Discussion 中说明，Protobuf / FlatBuffers 等 IDL 需要额外 schema parser 和类型映射工具才能转换成 zBuffer message structure。zRPC 的最完整 zero-copy 效果依赖 RDMA/NIC scatter-gather 能力、buffer registration 和 ring buffer 管理；当 scatter-gather element 数量超过 NIC 能力，或者应用无法提供可直接传输的 buffer 时，仍可能退化为 coalescing copy。公开仓库状态上，论文明确公开 TCP edition，因此公开 artifact 与论文 RDMA 评测之间还存在实现覆盖范围的不确定性。
