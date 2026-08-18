## All-bank PIM（全 bank 命令式 PIM，host 控制的 SIMD 风格执行）

术语解释
一类 bank-PIM 架构：host 发出紧凑的 all-bank 命令，同时触发芯片内所有（或一组）bank 旁 PIM 单元做 SIMD 风格执行，缓解命令带宽瓶颈；控制权保留在 host，PIM/non-PIM 模式经内存映射配置寄存器切换。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Samsung（ISCA'21 工业产品论文）与 SK Hynix（GDDR6-AiM HotChips'22）提出的 bank-PIM 形态：为克服逐 bank 下发指令的命令带宽瓶颈，host 用一条紧凑命令驱动全部 PIM 单元并行执行同一操作（SIMD 风格），支持 GEMV 与其它向量 kernel 的简单命令集。其特点：控制权完全保留在 host（区别于 UPMEM 每 bank 独立编程的 DPU），all-bank 命令执行期 bank 被占用（与 FR-FCFS 调度的"All-Bank 命令"基线同源，见 知识库_kernel调度.md"FR-FCFS 与 PIM 并发调度基线"）；PIM 与非 PIM 模式通过写配置寄存器切换，ECC 逻辑可直接读 PIM/non-PIM 模式信号判定当前解码模式。论文选择 DDR5 all-bank PIM 作底座，理由：DDR 的容量/外部带宽比最高（内部 PIM 带宽收益最大），且 rank 组织天然匹配 DDR rank 结构（可靠性需要 rank 组织，见"Two-tier ECC"条目）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（论文可靠版 all-bank PIM，一次 GEMV）：host 把权重/激活按 bank 分布 → host 写配置寄存器切到 PIM 模式 → 发 all-bank PIM 命令（每 tCCDL 窗口一条）→ 各 bank 旁 PIM 单元（4 FPU）以 bank 带宽执行乘加，中间结果写 PIM 本地 SRAM → 读回结果时 host 侧做 replication/reduction。受限点：all-bank 命令要求所有 bank 的行先激活（tFAW 四激活窗口使控制器必须等全部 bank 就绪 → idle bubble，论文用奇偶组流水消除，见"Odd/Even Bank Pipelining"条目）；可靠性上 all-bank 并发激活使控制器无法直接定位出错 bank（须顺序读各 bank 重构 rank 码字，见"Two-tier ECC"条目）。Samsung 原型 [48] 的 PIM-host 模式切换约 37.5ns。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：工业原型 Samsung HBM-PIM/Aquabolt-XL、LPDDR5-PIM、SK Hynix GDDR6-AiM；论文在 Ramulator2 中建模（DDR5-6400、8 channel×4 rank、32 bank/chip、每 2 bank 一个 PIM 单元）。使用：host 驱动/内核生成 PIM 命令 trace 喂给模拟器评估执行时间；与 PIM 并发调度的关系见 kernel 调度层 All-Bank 命令基线（时间片轮转、执行期无 bank 供 CPU）。

MERIDIAN 补充视角（ISCA'26，All-Bank-Mode 的 LPDDR5X 应用）：MERIDIAN 采用同名 All-Bank-Mode：DRAM 命令广播到所有 bank 的同地址位置，最大化 bank 级并行与内部带宽，连续 column 命令的 issue 率受 t_CCD_L（同 bank group 相邻 column 命令最小间隔）约束。与 ECC 论文的 DDR5 all-bank PIM（host 控制 SIMD、TFAW 四激活窗口导致 idle bubble、需 Two-tier ECC 保证可靠性）不同，MERIDIAN 的 PU 是每 bank 独立放置（16 FP16 比较/乘法/加法器 + 4KB buffer），bank 旁就地消费 256-bit（16 FP16）数据；PU 只承担 GEMV/skinny-GEMM/分段线性，softmax 与跨 channel 归约上移到控制器侧 NMU（专用 softmax 硬件），跨设备聚合交给 BOOMv2 RISC-V 核——即"all-bank 计算 + controller 侧归约/非线性"的分层组织，避免在 bank 内实现复杂控制逻辑。

涉及论文标题：
- ECC Enabled Reliable and Performant Processing-in-Memory
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
