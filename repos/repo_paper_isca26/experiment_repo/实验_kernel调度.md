# 实验_kernel调度

## Accelerating MoE with Dynamic In-Switch

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - **实现**：DySHARP 提出两大 kernel 级技术：(1) **Dynamic Multimem Addressing**——扩展 NVLink SHARP (NVLS) 的 multimem 指令，新增 `dymultimem.st`（Dispatch 多播）和 `dymultimem.ld_reduce`（Combine 归约）指令，支持动态非规整通信模式；(2) **Token-Centric Kernel Fusion**——通过 Token Tracker（TS Table、TID Table、OR Table）追踪 token 级依赖，配合 Token-Centric Scheduler（基于 megakernel + persistent thread blocks）将 Dispatch-Computation-Combine 全流水线化，使 Dispatch 和 Combine 并发执行以合并双向非对称通信模式。
  - **实验比较**：与七个 baseline 对比——DeepEP（SOTA 通信库）、NVLink SHARP (NVLS)（静态 in-switch computing）、FasterMoE、Tutel（粗粒度计算-通信重叠）、CCFuser、COMET（细粒度计算-通信重叠）、DualPipe（跨节点流水线重叠）。消融实验还包含 DySHARP-Basic（仅 dynamic multimem addressing 无重叠）、DySHARP-COMET（dynamic multimem addressing + COMET 重叠）、kernel fusion only（仅 token-centric fusion 无 dynamic multimem addressing）。

- 后端平台是什么，配置是什么。
  - **硬件平台**：模拟 NVIDIA GH200 NVL32（32-GPU 系统，通过 9 个 NVSwitch 全连接 fat-tree 拓扑互联），每 GPU 按 NVIDIA H200 规格配置。NVLink 4.0 双向带宽 900 GB/s，延迟 250ns（往返 1µs），flit 大小 16B。NVSwitch 每输入端口 16 个 256-depth 虚拟通道（8 请求 + 8 响应），端口归约 buffer 64KB。
  - **扩展评估**：DGX-H100（8 GPU）、多节点扩展（4/8×DGX-H100、2/4×NVL32，IB 互联）、64 GPU 节点（扩展 NVL32，18 NVSwitch，每 switch 64 端口）。

- 评估性能的软件/脚本是什么。修改了什么。
  - **模拟器组合**：整合 **BookSim2**（cycle-accurate 片上网络模拟器）和 **Accel-Sim**（GPU 模拟器，扩展到支持 Hopper 架构 FP8 kernel）进行 cycle-accurate 多 GPU 模拟。模拟器经验证，GEMM 和 DeepEP 通信算子平均误差在 6% 以内。
  - **修改**：Accel-Sim 扩展支持 Hopper 基本特性 + 高性能 FP8 kernel；BookSim2 集成以模拟 switch-based 多 GPU 网络；支持多 GPU 并发执行。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：论文未明确说明 DySHARP 是否开源。使用的模拟器 BookSim2（https://github.com/booksim/booksim2）和 Accel-Sim（https://github.com/accel-sim/accel-sim）为开源项目。
  - **评估原理与流程**：
    1. **输入**：模型配置（hidden size、MoE hidden size、attention heads、sequence length、expert 数量、topk），token 分布（训练用正态分布 std=0.032，推理用幂律分布 α≈1.5），硬件配置（GPU 数量、NVLink 带宽/延迟、NVSwitch 端口/VC/buffer）。
    2. **模拟执行**：Accel-Sim 模拟 GPU 端 kernel 执行（GEMM kernel、dymultimem 指令执行、token tracker 表操作），BookSim2 模拟 NVLink/NVSwitch 网络传输（dymultimem 请求包转发、多播复制、归约完成检测）。两者 cycle-accurate 协同，GPU 间通过 switch-based 网络并发通信。
    3. **性能输出**：MoE 层执行时间（Dispatch + GEMM-1 + GEMM-2 + Combine 时间分解）、端到端训练/推理加速比、带宽利用率、通信流量对比（DeepEP/NVLS/DySHARP）、payload efficiency（动态 multimem vs 显式寻址）、AL-TLB 命中率、reduction buffer 命中率。
    4. **kernel 执行全过程**：Source GPU 的 LSU 从 shared/global memory 获取 target expert list → 组装 dymultimem 请求包（单 multimem 地址 + target 扩展 flit）→ 经片上网络→NVLink→NVSwitch。Switch 的 Route 模块按 target 计算输出端口，复制并裁剪请求包，多播到各 destination GPU。Destination GPU 的 Hub 中 Hardware Memory Manager 查 AL TLB（miss 则查 DRAM 中 AL Table），完成 algebraic→layout index 映射→multimem→virtual address 转换→写入/读取数据。Combine 时 Switch Reduction Logic 追踪归约计数，计数归零后返回最终归约结果给 source GPU。
