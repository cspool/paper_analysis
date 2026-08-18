## Unified Virtual Address（UVA，统一虚拟地址）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UVA（CUDA 4.0 引入，2011）是 NVIDIA 提供的统一虚拟地址空间机制：主机内存与所有已连接 GPU 的设备内存处于同一个虚拟地址空间中，应用可用一个指针统一引用 host 与 device 内存；配合 GPUDirect Peer-to-Peer（CUDA 4.0 的 P2P 直接访问），GPU 间可经 NVLink/PCIe 直接 load/store 对方内存，无需先拷回主机内存。MoE-Hub 论文把 UVA 定义为"address-centric"通信模型的代表：生产者用与本地访存相同的 store 指令发起远程写——本地 TLB 查址后，远程 store 请求被路由到传输单元（hub）投递到其他设备。虽然统一了内存空间，但该模型要求生产者必须知道消费者内存中的确切目标地址才能发指令——这正是 MoE 动态路由下软件地址解析复杂性的根源。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 MoE-Hub 的语境中，UVA 的运转流程（baseline dispatch）：生产者 SM 发普通远程 store（虚拟地址指向远程 GPU 内存）→ 本地 TLB 查页表 → 解析出目标 GPU 与远程地址 → 请求路由到 hub 传输单元 → NVLink 送达 → 目标 GPU 的 IOMMU 做地址翻译 → 写入 L2/内存。问题：该地址必须在发指令前已知——MoE 中 token 在 expert 输入张量里的行偏移是运行时动态的，故软件必须先做跨 GPU 地址协调（同步、shuffle、CPU/GPU 布局计算）才能通信。MoE-Hub 的 st.rowsp 与 UVA 兼容共存：st.rowsp 的事务在 TLB 用轻量指令 flag 门控逻辑解析 MallocID→GPU ID（不干扰标准地址翻译路径），地址分配转到消费者 AAU；常规远程 store 仍走 UVA 路径（绕过 AAU）。即 MoE-Hub 不废除 UVA，而是为 MoE 增加一条不经由"生产者已知地址"的旁路通信路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
UVA 是 CUDA 运行时/驱动的能力（cuPointerGetAttribute 可查询指针所属设备），P2P 需 GPU 支持（NVLink/PCIe 直连）并启用；CUDA-aware MPI、NCCL 等依赖 UVA。编程上开发者直接解引用跨设备指针（远程 load/store），或经 cudaMemcpyPeer/P2P DMA 引擎做批量拷贝。局限（网络证据）：直接 load/store 与计算争用线程资源、对合并访问敏感、GPUDirect RDMA 在 kernel 运行期间与 NIC 内存无一致性保证；在 MoE-Hub 语境下其根本局限是"address-centric"——目标地址必须静态已知，无法表达 MoE 的运行时动态目的地。MoE-Hub 的贡献正是用 destination-agnostic 范式（st.rowsp + AAU）补上这一缺口，同时复用 UVA/IOMMU 的翻译与 NVLink 传输基础设施。

涉及论文标题：
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
