## FIFO 与 Ping-Pong（双缓冲）通信缓冲（HLS dataflow 通道实现）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 数据流架构中任务间通信缓冲的两种实现（Fig. 1）：FIFO-based 数据流——producer 逐元素顺序写入先进先出缓冲，consumer 一旦所需元素就绪即开始处理：性能高（消费者提前启动、间隔小）、资源省（只存 in-flight 数据），但要求两侧访问顺序与计数严格一致；ping-pong-based 数据流——数据按块交替写满 Buffer1/Buffer2，producer 写一块时 consumer 读另一块：消费者须等整块写完才能开始（latency 高）、内存开销至少两倍块大小，但块内可随机访问、灵活。
- Web 证据（Xilinx UG902/Vitis-Tutorials）补充：Vitis HLS dataflow 对数组通道默认顺序访问→深度 1 FIFO、非顺序→ping-pong RAM（PIPO），可用 config_dataflow -default_channel fifo|pipo -fifo_depth N 覆盖；FIFO 深度配错会在 cosim 中死锁。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 片上实现：FIFO 由 BRAM/分布式 RAM + 读写指针与 empty/full 握手组成；ping-pong 由两块独立 BRAM + 读写选择信号组成。CODO 的缓冲选择策略是 FIFO-first：无违例连接优先 FIFO；细粒度违例无法消除处回退 ping-pong；调度冲突不可解处（如 A-B-C-D 链上 B 与 D 并行策略冲突使 C 违例）把 C-D 间缓冲降级为 ping-pong、保住 A→C 的 FIFO 段。
- 资源对比（Tables III/IV）：ScaleHLS/HIDA 的 ping-pong 设计 BRAM 高（ResNet-18 3*224*224：HIDA 46.1%、ScaleHLS 208.7% 溢出），CODO FIFO 设计 BRAM 13.6%；Allo 极端 array partition 把缓冲拆小到 LUTRAM→BRAM 0%、LUT 升高。
- 时间线对比（Fig. 2(b)）：同一三 kernel 例子，ping-pong 派（POM/ScaleHLS/HIDA）长间隔低重叠，理想 FIFO 间隔最小；StreamHLS 因控制逻辑与循环序错位把 FIFO 写推迟到近 8/9 迭代后→大量气泡，性能只比 ping-pong 略好。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- HLS 层使用：Vitis HLS 自动选择或 config_dataflow 指定；CODO 在 MLIR 上用专用 FIFO/ping-pong 类型与操作建模，综合前即确定每个连接的缓冲类型。量化：CODO 的 FIFO 占比 Gesummv/Residual Block/MobileNet/ResNet-18 100%、Multi-Head Attention 84%、GPT-2 89%（Table VIII），是 FIFO-first 与违例消除的联合效果。

涉及论文标题：
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
