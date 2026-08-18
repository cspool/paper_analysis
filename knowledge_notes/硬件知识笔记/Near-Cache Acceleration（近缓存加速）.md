## Near-Cache Acceleration（近缓存加速）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Near-Cache Acceleration 是把加速逻辑（计算/地址生成/数据移动优化）放置到缓存层次旁边（尤其 L1/L2 附近）的一类架构范式，属于 Near-Data Processing（NDP）家族中"靠近缓存而非靠近内存"的分支：与 Near-Memory（PNM/M²NDP，数据放到内存侧）相比，近缓存加速器离执行核心更近、访存延迟更低，能直接观察/干预缓存行为。täkō [42]（cache 计算卸载）、Xcache [43]、METAL [21]、Leviathan [41]（任务卸载分类法）是该方向代表。RoboCortex 的 RSU 即部署在 L1 缓存旁的可编程近缓存加速器，对应 Leviathan 分类中的 long-lived task（长生命周期任务）。关键价值：既硬件加速搜索计算，又把程序级物理信息（物理坐标）暴露给缓存，使缓存能做基于物理坐标的优化——这是普通缓存（只认内存地址）做不到的。
- 从硬件架构角度拆解术语，给出运转流程具体例子：RoboCortex 中一条 NNS 的近缓存执行流程 = CPU 发 VLIW 指令 `NNS Pos, RootAddr, Num, OAddr`（阻塞流水线）→ RSU（近缓存，8×8 CGRA）接收 Pos/RootAddr → Read 原语经 LSU 向 L1 发取数请求（与核内正常访存同路径）→ 数据返回前 RSU 计算阻塞 → 数据流原语逐级执行 DFS → 结果经 Write 写回 OAddr（L1 旁）→ 指令提交。缓存因 RSU 暴露的物理坐标获得额外优化机会：Path Buffer 命中判定、RSU 引导预取。对比纯 CPU：地址生成在核流水线、取数在缓存，二者解耦，缓存无程序物理信息。
- 术语一般如何实现？如何使用？：实现方式包括 (a) 缓存内嵌可重构单元（täkō 的 polymorphic cache、Xcache 的 domain-specific cache）、(b) 缓存旁挂可编程数据流阵列（RoboCortex RSU）、(c) 语义指令封装（METAL 的 index caching）。编程/使用：RoboCortex 通过配置函数（config，线程启动前）配置 RSU 计算图（编译期静态映射），运行时用 NNS 指令 offload；因点云任务长期循环运行，同一配置无需重配。相关开源/工具：CCF CGRA 编译框架、zsim 模拟器。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）
