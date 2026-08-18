## MGPUsim（多 GPU 模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MGPUsim 是 Akita 项目基于 Go 语言的多 GPU 架构模拟器（ISCA'19，Sun et al.），模拟 AMD GCN3 ISA，重点支持多 GPU 系统（每 GPU 独立 TLB/GMMU/DRAM + inter-GPU 网络/RDMA engine），也支持 UVM 与统一内存。开源：原 gitlab.com/akita/mgpusim（已归档），现维护于 github.com/sarchlab/mgpusim（BSD 许可）。NVIDIA GPU 模拟仍在开发、仅 AMD GCN3 稳定。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
基于 Akita 离散事件框架逐周期模拟：SM 发指令 → cache/TLB 层级 → GMMU 页走查 → far-fault/UVM 驱动逻辑 → DRAM/inter-GPU 网络；支持插入自定义组件——论文在其中新增 duplicated TLB、DDU+CDB、Access Count Monitor 并改造 far-fault 处理（远端读 → 32MB 复制/细粒度去重；CPU 页 → 常规页装载），NVLink 网络参数替换为真机实测延迟-大小曲线。输出总/kernel 执行时间、cache/TLB 命中率、DRAM 事务数等。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
samples/ 与 benchmarks/ 下内置 AMD APP SDK、DNN-Mark、HeteroMark、Polybench、Rodinia、SHOC 等套件：在各目录 go build 后运行 `./fir -timing --report-all` 输出 metrics.csv。论文用 4-GPU 配置（SM 1.0GHz×108、L2 2MB、DRAM 2GB、L1/L2/L3 TLB 16/128/1024 项）+ 13 个 benchmark 对比 GPS/GRIT/CoarseDup/CDFD，并扩展到 8/16/32 GPU。

LIBRA 补充视角（ISCA'26）：用 MGPUsim 评估 4-GPU 系统（每 GPU 独立 local page table 与 GMMU；SM 1.0GHz×108、L1 D-Cache 64KB 4-way、L1 I-Cache 32KB 4-way、L2 2MB 8-way、DRAM=应用内存足迹 70%、三级 TLB、inter-GPU 300GB/s NVLink 3.0、CPU-GPU 32GB/s PCIe-v4），23 个 benchmark（AMDAPPSDK/Hetero-Mark/SHOC/DNNMARK，表 IV，4MB–544MB/GPU，Adjacent/Scatter-Gather/Mixed/Random 模式）。修改：GPU 侧新增 MMP 硬件（Triggered Table + MMAT）、扩展 TLB-miss 元数据携带 source SM 信息、修改 CPU UVM runtime 的 fetch_access_counter_buffer_entries(.)、新增软件 PPC 模块；扩展到 1/8/16/32 GPU（按比例缩放工作负载与模拟器组件）与 multi-rack（2 rack×8 GPU，NDR 400Gb/s InfiniBand）评估。
ShadowUpdate 补充视角（ISCA'26，MGPUSim 建模访问计数迁移与 ShadowUpdate）：ShadowUpdate 用 MGPUSim 模拟 4-GPU 平台：每 GPU 16 SA × 4 CU = 64 CU；L1 TLB 32 项 32-way 1-cycle、L2 TLB 512 项 16-way 10-cycle；共享 GMMU（PWQ 64、8 PTW、PWC 128、每级 100 cycle）；L2 2MB 16-way；DRAM 4GB 1TB/s 100-cycle；CPU-GPU 32GB/s、GPU-GPU 600GB/s；access counter 阈值 256；分布式 CTA 调度。修改：① 实现 access counter-based page migration；② 实现 ShadowUpdate（invalidation 消息携带新 PA、GMMU 顺带写 PTE、IfMT 挂起在途翻译、completion 广播清 IfMT）；③ 支持 8–32 GPU 扩展、2MB large page、CTA Clustering/LADM 调度、LLM GEMM 微 kernel 评估。14 个 workload（AMD APP SDK/DNN-MARK/HETERO/PANNOTIA/POLYBENCH/SHOC）平均 1.40×。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
