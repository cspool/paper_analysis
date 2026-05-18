## RoMe: Row Granularity Access Memory System for Large Language Models

- baseline方法是什么？
  Baseline是conventional HBM4-based memory system，配置为每cube 32 channels（64 pseudo channels）、8 Gbps data rate、2 TB/s bandwidth、32B access granularity、1KB row size。MC使用FR-FCFS scheduling policy + open-page policy + per-bank refresh。bank states包含Idle/Activating/Active/Reading/Writing/Precharging/Refreshing七个状态，需管理15个timing parameters，bank FSM数量等于每PC所有bank数。MC必须做bank group interleaving和PC interleaving来最大化带宽利用率。每64-bit data channel需要10 row C/A pins + 8 column C/A pins。在LLM serving场景中，decode阶段的weight/KV cache/activation以KB-MB级连续块被顺序访问，但传统HBM4将这些大块访问拆成128个32B cache-line transactions。

  全栈执行例子（DeepSeek-V3 decode阶段读取一段weight block，HBM4 baseline）：
  - 算法层：LLM decoder block执行GEMV（decode阶段），需要读取weight矩阵连续block（如12MB weight chunk），activation为单token向量。
  - 系统框架/Serving层：accelerator DMA engine发出memory read requests→MC address mapping将物理地址映射到channel/PC/bank group/bank/row/column→每个12MB block被分解为~384K个32B cache-line read requests→request queue需depth≥45才能充分look ahead做bank-level parallelism。
  - 编译框架层：论文未明确说明（使用底层硬件DMA直接管理memory transfer，无编译框架介入）。
  - kernel调度层：论文未明确说明（无kernel级优化，所有调度在硬件MC层完成）。
  - 硬件架构层（核心）：MC command scheduler对每个32B request执行：检查bank state→若row-buffer hit直接发RD；若miss则发PRE→ACT→RD→维持open-page等后续访问→page policy判断何时发PRE。同时在不同bank group间交错（tCCDS=1ns间隔）和不同PC间交错。bank FSM需跟踪128 banks/channel的状态，timing constraints包括tRCDRD/tRAS/tRP/tCCDS/tRRDS/tFAW等15个参数。每通道18 C/A pins承载RD/WR/ACT/PRE/REF/MRS命令。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出RoMe，核心思路是将HBM接口从cache-line粒度的column-level RD/WR替换为row-level RD_row/WR_row（4KB粒度），并通过VBA、command generator、simplified MC三个组件重构整个memory hierarchy。

  **缺陷1：Baseline将LLM大块连续访问拆成大量32B碎片事务，导致MC调度复杂且queue压力大**
  → RoMe将AGM_C从32B提升到4KB，一个RD_row替代128个32B RD命令。request queue从需要≥45 entries降到2 entries即可饱和带宽。DRAM访问从column-level变为row-level后，ACT/PRE/WR/RD之间的复杂时序不再暴露给MC。

  **缺陷2：Baseline的bank group和pseudo channel是为了在cache-line粒度下扩展带宽而引入的，但迫使MC做BG/PC interleaving和复杂状态追踪**
  → RoMe提出VBA，从MC-DRAM接口中移除bank group和PC概念。单个VBA由两个不同BG的bank以time-multiplexed方式组成，两个PC并发工作。一个VBA即可提供满带宽，MC不再需要跨bank group和PC搜索ready bank。bank FSM从每PC所有bank（~128个）降到仅5个。

  **缺陷3：Baseline需要管理7个bank states、15个timing parameters、open/close/adaptive page policy的复杂决策**
  → RoMe MC只发RD_row/WR_row/REF三种命令，bank states缩为4个（Idle/Writing/Reading/Refreshing），timing parameters缩为10个。Row granularity本身保证了row-buffer locality，不再需要page policy——每次row access后自动precharge。调度简化为跨VBA交错+oldest-first公平性。RoMe MC scheduling logic面积仅为conventional MC的9.1%。

  **缺陷4：Baseline每通道需18 C/A pins（10 row + 8 column），随着HBM代际演进C/A-to-DQ pin ratio持续上升**
  → RoMe移除column C/A pins（8个），减少address bits（PC bit + 1个bank bit），C/A pins从18降到5（节省72%）。省下的13 pins/channel × 32 channels = 416 pins聚合后增加4个新channel，仅需额外12 pins。HBM cube从32 channels扩到36 channels，带宽从2 TB/s提升到2.25 TB/s (+12.5%)。

  **缺陷5：Baseline在LLM decode阶段受memory bandwidth-bound限制，但HBM带宽扩展受限于DRAM core frequency和access granularity**
  → RoMe通过row granularity释放C/A pins→增加channel count→直接提升bandwidth。Command generator放置在logic die中，将row-level command静态展开为传统DRAM command序列，内部处理tRRDS/tCCDS等时序约束。decode阶段TPOT在DeepSeek-V3/Grok 1/Llama 3上分别降低10.4%/10.2%/9.0%。DRAM energy降低1.9%/0.7%/0.7%，主要来自ACT数量减少（仅需minimal ACT）和interposer command traffic减少。

  论文方法全栈执行例子（以DeepSeek-V3 decode阶段读取一段weight block，RoMe）：
  - 算法层：不修改模型或推理算法，使用相同LLM architecture（MLA+MoE）和weight format（BF16）。
  - 系统框架/Serving层：accelerator DMA engine发出4KB-granularity memory requests→RoMe MC只做address mapping（channel/SID/VBA）和oldest-first scheduling→避免连续访问同VBA以保持带宽→MC发出RD_row命令。
  - 编译框架层：论文未明确说明（无编译框架介入）。
  - kernel调度层：论文未明确说明（无软件kernel修改，所有优化在硬件路径）。
  - 硬件架构层（核心创新）：(1) RoMe MC: 接收4KB request→address mapping→选择一个空闲VBA→发RD_row命令→仅需4 bank states + 10 timing params + 5 bank FSMs + 2-entry queue。(2) Command Generator in HBM logic die: 接收RD_row→插入tRRDS−tCCDS intentional delay→对VBA内Bank A发ACT→对Bank B发ACT→按tCCDS间隔交替发RD（两个BG的数据传输错开填满通道）→发PRE→VBA回到Idle。(3) VBA: 两个不同BG的bank以time-multiplexed方式工作，两个PC并发接收数据→有效row size 4KB→36 channels/cube提供2.25 TB/s带宽。(4) Refresh优化: MC每2×tREFIpb发一次per-bank refresh→command generator对VBA内两bank间隔tRREFD发两个REFpb→每VBA stall从2×tRFCpb降到tRFCpb+tRREFD。RoMe MC scheduling logic面积仅为conventional MC的9.1%，command generator占logic die 0.003%，总chip area overhead仅0.10%。
