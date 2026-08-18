## Cassandra Encoder & Decoder（位级格式转换硬件单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cassandra 为消除"压缩格式 ↔ 标准浮点格式"转换开销而设计的轻量 encoder/decoder IP。Decoder 把 speculation 数据（剪枝+截断+指数压缩后的比特流）重建为标准浮点供 SM/PE 计算。Cassandra-1（unary 路径）数据流：mantissa concatenator 拼接截断尾数高低位 → sign+mantissa 送 dynamic shifter 等待指数 → 指数按 8-bit 分块送 parallel zero counter（逐位并行输出"前面连续 0 的累计数"、总 1 个数与末位值；跨 chunk 用前一 chunk 末位与末尾连续 0 计数进位修正，Algorithm 1）→ zero eliminator 队列 → LUT unary 码本查指数 → bitmap-based de-sparsification 按 bitmap 把剪枝位补 0 输出。Cassandra-2（MX 路径）：指数经 crossbar 广播到各元素 accumulator → parallel zero counter 数尾数前导 0 确定移位量与指数减量 → dynamic shifter 移尾数、accumulator 减指数 → 同样做 bitmap de-sparsification。Encoder 做在线格式化：输入数据 top-k 分组 speculation/verification 两组，verification 组连同 bitmap 直接入 buffer，speculation 组做指数压缩 + 尾数截断后入 buffer 写主存。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
集成位置：GPU——decoder 置于 L2 cache ↔ interconnect 之间、encoder 置于主存 ↔ L2 之间，每个内存分区（memory channel）独立配一组，由 memory controller 管理（类似 L2 的管理方式）；NPU——encoder/decoder 置于 DMA 内由 DMA controller 管理，物理地址区间预存于 DMA（无虚拟内存）。数据流（GPU decode 一步）：superblock 从主存整体载入 L2 → 各类型 cache block 送 decoder（decoder buffer 拼接余量）→ 解压为标准 FP → 经 interconnect 进 SM 做 GEMM；KV cache 每生成一个 token 经 encoder 在线格式化写回。成本（Synopsys Design Compiler 28nm + Samsung 28nm SRAM Compiler，64 TFLOPS NPU、40 个 decoder、9MB scratchpad@1024B/cycle）：encoder 0.08mm²/0.1%/0.017W，decoder 1.76mm²/1.9%/0.264W，合计约 2% 面积开销。设计要点：decoder 数量须匹配 L2/scratchpad 最大带宽（解码不成为读路径瓶颈），encoder 可少配（权重离线格式化，仅 KV 在线使用 encoder）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SystemVerilog 端到端 encoder/decoder + 自研 64TFLOPS NPU（systolic array、VPU、DMA、scratchpad），Synopsys Design Compiler 28nm 综合；性能由 Accel-Sim（GPU 集成）与扩展 Scale-Sim+LPU simulator（NPU 集成）周期模拟 + 实测接受率换算。同类参照：Ecco（GPU cache 压缩编解码器）、Oaken（KV cache 压缩编解码器）——区别是 Cassandra 编解码服务于投机解码（speculation/verification 双数据流）而非单纯损失压缩。开源：论文未提供 RTL 与修改版模拟器链接（无法确认）。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
