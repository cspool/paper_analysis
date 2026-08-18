## Superblock-based Data Management（超块内存映射管理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cassandra 解决变长压缩数据（变长 unary 指数、随稀疏率变化的 speculation/verification 比例、bitmap）无法对齐 cache block 的内存映射方案：以多个 cache block 组成的 superblock 作为加载/驱逐的基本单位，把 bitmap、变长指数、mantissa 等不同类型数据完全打包（fully packed）连续存放；加载时整个 superblock 一次读入 L2 再分发给 decoder。解码过程中某些类型的数据块可能未被完全消费：decoder 内部 buffer 保存该类数据的跨 block 余量并与下一 block 拼接；当 buffer 余量 >128B 时，memory controller 跳过本次该类型的 cache block，并记账跳过次数、维护每类数据下一次应送的 block 地址。目的：保证全局内存连续读取与稠密映射，避免"压缩数据不能完美装进 cache block"造成的冗余加载与性能退化。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
load superblock S（n 个 cache block，混装 bitmap | 变长指数 | mantissa）
→ 整体进 L2
→ 按类型分送 decoder：decoder_buffer[type] += 当前 block 数据
→ if len(decoder_buffer[type]) > 128B: 跳过下一块该类型 cache block; skip_count[type]++
→ memory controller 为每类数据维护"下一次应送 block 地址"（含跳过记账），按需送数
```
适用两类系统：GPU 的 cache-based 内存（superblock = 多个 cache block 的组合）；NPU 的 scratchpad 内存（任意 block 尺寸直接指定，同样避免 row buffer miss / page miss）。与 decoder buffer 的流式拼接配合，使变长编码的随机访问问题转化为顺序流式解压问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文未提供开源实现。实现要素：(1) superblock 粒度 = 多个 cache block 的打包单元；(2) memory controller 按数据类型分流的读指针 + skip 计数簿记；(3) decoder 内 128B 阈值的余量 buffer。作用类似内存侧"类型分流的块调度"，是 Cassandra 变长编码（unary/MX + bitmap）能在真实 GPU/NPU 内存系统落地并保持带宽效率的关键支撑；一般化后可用于任何变长压缩数据流的 cache/scratchpad 友好映射。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
