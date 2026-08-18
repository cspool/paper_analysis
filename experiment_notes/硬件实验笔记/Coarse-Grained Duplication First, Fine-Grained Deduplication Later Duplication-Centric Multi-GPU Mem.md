## Coarse-Grained Duplication First, Fine-Grained Deduplication Later Duplication-Centric Multi-GPU Memory Management

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现：CDFD，面向 UVM 多 GPU 系统的 duplication-centric 内存管理硬件-运行时协同设计，在 MGPUsim 中新增三类硬件组件并扩展 UVM runtime：
    (1) Duplicated TLB（32 项，每项 36-bit VPN + 7×36-bit PFN，共 1,152 B）+ 内存中 duplicated page sharer table：页表用 1-bit 标志标记重复页；写命中重复页时 L1 TLB 转发至 duplicated TLB，miss 则对 sharer table 页走查，随后把各共享 GPU 的物理地址交 GMMU 发远程写更新（off critical path）。
    (2) Duplication and Deduplication Unit (DDU)：管理 far-fault、按成本收益分析触发复制/去重；内含 Candidate Deduplication Buffer（CDB，256 项，36-bit VPN + 12-bit 总远程更新计数 + 16×17-bit 子项，共 10,240 B），按 sub-entry 组织（32MB 页含 16 个 2MB 子项、1MB 页含 16 个 64KB 子项），每个子项含 8-bit 本地访问计数 + 8-bit 远程更新计数 + valid bit；收益 = 本地访问数 − 远程更新数，收益最低者优先去重。
    (3) Access Count Monitor（256 项，36-bit VPN + 8-bit 计数器，共 1,408 B）：跟踪远程更新最频繁的重复页的本地访问，周期与 CDB 同步（右移 1 位叠加新旧计数，融合长期/短期访问模式）。
    总计 12,800 B/GPU 片上存储 + 66.9K–114.9K NAND2 等效门。策略：远端读触发 32MB 粗粒度复制（对齐 NVIDIA TLB 单条目 32MB 覆盖范围），之后对 2MB/64KB 子页选择性细粒度去重；运行时维护 current/target duplication ratio（LRU 列表采样自适应调整 target），超 ratio 时先去重腾空间再复制/装载常规页；sys-scoped 写触发副本合并为单一权威版本（类似 cudaMemAdviseSetReadMostly 处理）。
  - 实验比较：CDFD vs GPS（订阅式复制 + 批量远程写更新）、GRIT（on-touch 迁移 + 访问计数迁移 + 复制混合）、CoarseDup（消融：仅 32MB 粗粒度复制 + 远程更新）。指标：归一化端到端性能、性能分解（复制/迁移开销 vs 远程访问开销）、迁移/复制次数、复制页大小分布（32MB/2MB/64KB）、duplication ratio、coherence 广播数（有用/无用）、功耗与面积。
- 硬件平台是什么，配置是什么。
  - 架构模拟平台：MGPUsim 4-GPU 配置——SM 1.0 GHz × 108/GPU；L1 D-cache 64 KB 4-way、L1 I-cache 32 KB 4-way；L2 2 MB 8-way；DRAM 2 GB/GPU；L1 TLB 16 项 16-way 1 cycle、L2 TLB 128 项 8-way（每项 16 sub-entries）10 cycles、L3 TLB 1024 项 8-way（每项 16 sub-entries）40 cycles，均 LRU；GMMU 8 个页表走查器、100 cycles/level；CPU-GPU 网络 128 GB/s PCIe-v5；inter-GPU 网络按实测 NVLink 3.0 各传输大小的延迟建模（敏感性研究扩展到 8/16/32 GPU）。
  - 真机特征化平台：NVIDIA DGX A100（8× A100 80GB SXM4，NVLink 3.0，驱动 570.148.08，CUDA 12.8，1,800 GB CPU 内存，20 TB SSD）与 NVIDIA DGX H100（8× H100 80GB SXM5，NVLink 4.0，驱动 570.195.03，CUDA 12.8，22 TB SSD）。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - MGPUsim（论文引用 [38]，ISCA'19，Akita 项目多 GPU 模拟器）。原 GitLab 仓库已归档：https://gitlab.com/akita/mgpusim ；现维护于 GitHub：https://github.com/sarchlab/mgpusim 。论文在其 4-GPU 配置上新增 duplicated TLB、DDU+CDB、Access Count Monitor 并扩展 UVM runtime 逻辑；NVLink 3.0/4.0 延迟-大小参数来自真机 NCCL 与 cudaMemPrefetchAsync 实测。
- 模拟器模拟什么的性能，修改了什么。
  - MGPUsim 是 Go 编写的多 GPU 架构模拟器（Akita 离散事件框架，AMD GCN3 ISA），模拟 SM、缓存、sub-entry TLB 地址翻译、GMMU 页表走查、DRAM 与 inter-GPU 网络，输出总/kernel 执行时间、cache/TLB 命中率、DRAM 事务数等。论文修改：新增上述三类硬件组件及 DDU 触发逻辑（远程读→检查 duplication ratio→CDB 取候选→去重→32MB 复制；CPU 页装载同理），far-fault 处理区分远程读与 CPU 页；NVLink 网络参数替换为实测延迟-大小曲线。
- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - CDFD 的 MGPUsim 修改代码论文未给出开源链接（无法确认）；MGPUsim 本身开源（Go，链接同上）。使用流程：clone MGPUsim（github.com/sarchlab/mgpusim）→ 在 benchmark 目录（如 samples/fir）`go build` → `./fir -timing --report-all` 输出 metrics.csv。模拟原理：逐周期离散事件仿真——kernel 指令从 SM 发出，经 L1/L2/L3 TLB 翻译（L2/L3 为 sub-entry 结构），miss 由 GMMU 页走查，失败触发 far-fault 交 UVM driver（论文扩展为 CDFD DDU 处理）执行 32MB 复制 / 2MB-64KB 去重 / 远程写更新；输入 = benchmark（13 个 AMDAPPSDK/Hetero-Mark/SHOC/DNN-MARK 应用：SC、C2D、MM、MT、FIR、ST、IM2COL、FFT、PR、BERT-M、BERT-B、GPT2-M、GPT2，默认输入集）+ 系统配置；输出 = 归一化执行时间（GPS/GRIT/CoarseDup/CDFD 对比）、复制/迁移次数、duplication ratio、coherence 广播计数；功耗用 CACTI(32nm) 估 ACM/CDB 能量（0.00998 nJ/access）+ NVLink 1.3 pJ/bit 估复制/广播能耗，除以执行时间得平均功率。
