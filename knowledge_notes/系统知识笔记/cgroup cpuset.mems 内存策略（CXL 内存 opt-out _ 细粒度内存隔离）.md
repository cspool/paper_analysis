## cgroup cpuset.mems 内存策略（CXL 内存 opt-out / 细粒度内存隔离）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
cpuset.mems 是 Linux cgroup cpuset 控制器的内存节点限制文件：把某 cgroup 内进程的物理内存分配限制在指定 NUMA 节点集合。Vistara 用它实现"软件 opt-out"——把工作负载的 cpuset.mems 设为仅含本地 NUMA 节点，即禁用它使用 CXL 内存，无需 BIOS 改动或重启即可快速切换；反向（opt-in）则允许使用 CXL。集成进 Meta 的资源编排器（Twine）：运维按服务 profile 声明式下发内存策略，作业启动时自动应用，并可在线响应迁移/扩容/健康事件动态更新，最大化服务器 fungibility（节点可在时延敏感池与容量池间灵活迁移）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：作业提交 → 编排器查服务 profile（是否容忍 CXL 时延）→ 写 cgroup 的 cpuset.mems=0（仅本地 NUMA）或 cpuset.mems=0,1（含 CXL 节点）→ 内核分配器按 cpuset 约束选择物理内存 → 作业运行期间若服务升级/迁移，编排器在线改写 cpuset.mems 切换 opt-in/out。细粒度控制：同一 host 内后台/低优先服务默认 opt-in（放 CXL），把本地 DRAM 留给主负载；也顺带保证 CXL 内存持续被使用以提供可靠性遥测。论文策略：默认所有服务用 CXL，仅时延敏感/未验证服务 opt-out。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：内核 cgroup v2 cpuset 控制器（/sys/fs/cgroup/<cgroup>/cpuset.mems），配合 NUMA 内存分配路径（mempolicy）与 ZONE_MOVABLE 约束；Meta 侧由内部资源编排器 Twine（OSDI'20）驱动。使用方式：运维/自动化声明式策略（默认 opt-in，异常 opt-out），作业粒度或容器粒度设置；用于时延敏感负载保护、多租户内存分配、服务器池间 fungibility 迁移。论文未开源编排器代码；内核机制为公开 upstream 功能。

涉及论文标题：
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment
