## DeviceConfig

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

DeviceConfig 是 MetaAttention 中用于抽象硬件配置的组件，与 IntermediateTensor 共同定义 attention kernel 的 scheduling space。DeviceConfig 封装两类硬件约束：(1) **BaseTileShape（basetile）**——目标硬件上矩阵乘法指令的 optimal tile shape，如 H100 wgmma MMA instruction 要求 tile 对齐到 {64,128}×{128,64}，AMD MI250 Matrix Core 要求对齐到 {64,64}；同时约束 memory transaction 的 alignment 要求（如 128B cacheline）；(2) **MemoryInfo（memoryInfo）**——各 memory tier 的容量信息，如 Register File 256KB/SM、Shared Memory 228KB/SM（H100）、Global Memory 80GB，以及各 tier 之间的 bandwidth 和 latency 层级关系。

DeviceConfig 的作用是作为 scheduling 的**硬约束**——在 Tile Config Scheduling 中约束可枚举的 tile sizes 必须对齐 basetile；在 Tile Resource Scheduling 中约束 memory placement 不能超出 MemoryInfo 的容量限制（`MeetMemoryConstraint(plan, D.memoryInfo)` 检查 Σ IntermediateTensor.tile_size 是否 ≤ SMEM capacity 和 Register budget）。

从kernel调度角度拆解：DeviceConfig 使同一套 scheduling policy 可跨 hardware 复用。例如同一 attention template 在 H100 上时，DeviceConfig 提供 H100 的 basetile (64/128) 和 memoryInfo (RF 256KB, SMEM 228KB)；在 MI250 上时，DeviceConfig 提供 MI250 的 basetile (64×64 for Matrix Core) 和 memoryInfo (调整后的 Register/SMEM 容量)。Scheduling policy 在不同 DeviceConfig 下自动生成不同的 execution plan（tile size, memory placement, pipeline stages），无需 per-hardware 手写 kernel。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention 为每个支持的 hardware backend 预定义 DeviceConfig。用户不直接操作 DeviceConfig——框架在 compiling attention template 时根据 target device 自动选择对应 DeviceConfig。扩展支持新 hardware 时，仅需添加新 device 的 basetile + memoryInfo 配置，无需修改 scheduling policy 或 attention runtime（后者需 backend-specific lowering 如 TMA/WGMMA for NVIDIA 或 Matrix Core for AMD）。当前支持设备：NVIDIA H100 (CUDA 12.4)、AMD MI250 (ROCm 6.2.4)、AMD MI300X (via TileLang backend)。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends
