## Stream Unit（流式访存、父-子流依赖与 bexp）

术语解释
UTE 后端中负责地址生成的硬件单元：每个流按"元素大小 + 步长 + 起止边界"的模式从内存取数，流之间可形成父-子依赖树以支持间接寻址；边界由 bound expression（bexp）运行时计算。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
流（stream）= 从 beg 到 end、按 size×stride 步进的连续取数循环；一次"repetition"就是内层循环的完整执行，子流每随父流一次迭代开启新 repetition（可能边界不同）。每个流占一个 Stream Unit，其微架构分三块：Repetition Initializer（用 Bounds ALU 计算每次 repetition 的 beg/end，即 bexp 求值）、Mem Address Generator（内层循环增量地址）、NCA Address Generator（scratchpad 写地址，类似逻辑）。Access Queue 缓冲待发地址并合并同缓存行的连续访问。bexp 形式为 `Op1(I1, Op2(I2, I3))`，输入为：ATX 指令的运行时常量、父流取回的数据、repetition 序号；Op1/Op2 ∈ {加、乘、比较、移位}，2 字节编码——这个表达能力覆盖 CSR 行指针、指针追逐等间接模式。流抽象源自 decoupled/stream 访存研究（论文引用 Wang & Nowatzki ISCA'19 的 stream-based 访存特化）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CSR 行求和的流配置（论文图 9/10）：S1 根流取 `row_ptrs[r]` 与 `row_ptrs[r+1]`（8B 元素）；S2 子流取 `vals[e]`（4B 元素），其每次 repetition 的 beg=vals+c21×(parent 值)、end=vals+c21×(parent 下一值)，即 S2 的边界取决于 S1 刚取回的两个行指针。运行时：S1 的 repetition 先启动、取回边界数据经 Common Bus 送入 S2 Stream Unit 的 PDQ → S2 的 Repetition Initializer 用 bexp 算出本 repetition 的起止 → Mem Address Generator 逐个生成 `vals` 地址。父流领先子流的距离由 PDQ 容量（论文 1KB/Stream Unit）限定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
配置接口（论文图 12）：`UTE_cfg_stream_size(VAccId, stream_id, 8)`、`UTE_cfg_parent(VAccId, stream_id, parent_id)`（-1 为根流）、`UTE_cfg_bexp_beg/bexp_end(VAccId, stream_id, ENCODE)`，可选配置非单位 stride 与 flags；这些配置存于 VAcc→Streams 映射 CAM，任务执行时传播给分配的 Stream Unit。程序员职责：任务尺寸要保证流取数不溢出 NCA scratchpad、输出不溢出输出寄存器。适用场景：规则/半规则/间接访存模式的结构化预取与取数；比逐地址 DMA 描述符表达力强、比通用核访存省硬件。

涉及论文标题：
- ATX: Accelerator Task Extensions
