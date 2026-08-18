## Sector Cache（扇区缓存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Sector cache（扇区缓存）是一种缓存组织：一个缓存行（tag）下按更小的扇区（sector）粒度管理数据有效性，典型如 64B 行含 2 个 32B 扇区或 128B 行含 4 个 32B 扇区，每个扇区有独立 valid bit。相比整行缓存，它减少 tag 开销并支持只取请求的部分数据（细粒度 fill），常用于空间局部性差、单次只读少量字节的负载。Vulkan-sim 2.0 的 GPU 模型使用 sector size 32B 的 sector cache：BVH 节点数据往往大于 32B，一次节点读请求会被拆成多个 32B sector 请求发送。TTP 论文利用这一模型：预取请求对 >32B 的节点同样按 32B 拆块，每周期发送一块，与 demand read 一致。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 Vulkan-sim/TTP 中的运转流程：①RT unit 发起一次节点内存读取（例如 64B 内部节点）→ ②按 sector 粒度（32B）拆成 2 个请求，每周期一个插入内存访问队列 → ③L1 按 tag 查行、按 sector valid bit 判定每个 32B 是否命中；未命中的 sector 下行取数 → ④fill 时只填充缺失的 sector 而非整行 → ⑤后续对同一行另一 sector 的访问命中。TTP 的预取也走同一路径：从栈取节点地址（可能指向 >32B 节点）→ 拆 32B sector 逐周期发预取 → 提前把各 sector 填入 L1/L2。原理：对射线追踪这种"每次只读节点的一部分、节点又可能被多条射线共享"的负载，sector 粒度避免整行传输浪费，也让预取可以按最小传输单位（32B）渐进填充、与 demand 请求公平竞争每周期一个请求的带宽。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：sector cache 是缓存控制器设计，每个 tag 项带 N 个 sector valid bit（Vulkan-sim 中 32B sector），fill 按 sector 粒度写；TTP 中预取器按 32B 拆块是配合该模型的实现细节。使用场景：GPU 图形/光线追踪负载（节点与几何数据零散访问）、低空间局部性负载；Vulkan-sim 配置表 III 即采用 sector cache（sector 32B）。跨论文复用：任何在 sector-cache 模拟器上做预取的方案都需按 sector 粒度拆分请求并统计 sector 级命中。

涉及论文标题：
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing
