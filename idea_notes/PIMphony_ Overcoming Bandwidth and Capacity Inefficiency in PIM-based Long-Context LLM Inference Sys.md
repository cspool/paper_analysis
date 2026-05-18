## PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

- baseline方法是什么？
  PIM-based LLM inference系统（CENT[16]和NeuPIMs[21]）在长上下文decoding场景下存在三个系统性低效：
  (1) Head-First Partitioning (HFP)：将head-batch pair分配到PIM channel执行，默认batch/head并行足够填充所有channel。但长上下文下单个请求的KV cache足以占满一个channel容量，batch size被压低，Tensor Parallelism下不同请求token length不同造成channel执行时间不平衡（短请求channel早空闲），Pipeline Parallelism下每stage只激活与当前请求相关的少数channel（sparse channel activation）。论文在32K context CENT分析中MAC utilization下降48%。
  (2) Static PIM Command Scheduling：PIM primitive采用WR-INP→MAC→RD-OUT固定序列，传统scheduler只按保守时间间隔发射命令，不跟踪GBuf/OutReg entry级真实依赖，即使命令间无hazard也等待固定间隔。Attention的QK^T和SV因dh/dout小、数据复用低，I/O transfer频繁，静态调度导致MAC大量idle（小维度Attention MAC utilization低至14.7%）。
  (3) Static KV Cache Management：传统PIM指令的loop count和operand address在编译期固定，无法根据当前token length调整；系统必须按最大上下文Tmax为每个请求预留KV cache。真实workload请求长度差异大，静态预留导致平均容量利用率仅31.0%-40.5%。

  全栈执行例子（以CENT PIM-only系统、LLM-7B-32K decoding为例）：
  - 算法层：标准Transformer decoder，无算法改动（non-GQA，每层32 heads，dh=128，QK^T和SV均为GEMV操作）
  - 系统框架层：CENT PIM-only multi-node系统，HFP按head-batch pair将Attention GEMV分配到PIM channels。TP=2时两个module各持一半heads，PP=2时layer 1和layer 2分配到不同module。由于KV cache按Tmax预留，batch size受容量限制
  - 编译框架层：CENT使用fixed PIM instruction sequences（WR-INP→MAC→RD-OUT），loop count和operand address编译期固定为Tmax，无动态partitioning/metadata支持
  - kernel调度层：每个Attention head的QK^T固定映射到1-2个channel，其他channel idle（因无足够head-batch pair填充）。PIM controller按固定tWR-INP/tMAC/tRD-OUT间隔串行发射指令，即使命令间无hazard也等待，I/O transfer和MAC无法重叠
  - 硬件架构层：PIM module有16 channels × 16 banks，每个channel的PIM controller使用标准single-entry GBuf/output register，无dependency tracking logic，无on-module address translation

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**PIMphony**：通过TCP（Token-Centric Partitioning）、DCS（Dynamic Command Scheduling）、DPA（Dynamic PIM Access）三项协同设计，系统性地解决PIM在长上下文decoding中的channel utilization、I/O bottleneck和容量浪费。

  论文方法全栈执行例子（以PIMphony on CENT、LLM-7B-128K-GQA decoding为例）：
  - 算法层：无算法改动（同一Transformer decoder，GQA group size=8）
  - 系统框架层：TCP将Attention的token维度作为主并行维度。对QK^T，每个channel处理一段token的Key cache，与同一query做部分dot-product，score在module内拼接后Softmax。对SV，每个channel处理一段score/value的partial context，经module内reduction得完整context。不跨module同步（SV module内reduction开销<0.2% attention latency at 16K tokens）。DPA使KV cache按实际token length以1MB chunk lazy allocation，batch size不再受Tmax容量限制
  - 编译框架层：MLIR-based compiler自动识别Transformer decoder pattern→生成PIM-specific instruction sequences→embed TCP partition metadata（per-channel token segment range）、DCS dependency annotations（GBuf/OBuf entry-level hazard info）和DPA dynamic addressing（Dyn-Loop/Dyn-Modi编码）。IREE runtime HAL对接commercial PIM SDK，根据当前token length自适应dispatch
  - kernel调度层：TCP确保16-channel/16-bank配置下QK^T token length>256即可full channel activation（远超HFP的batch-dependent activation）。DCS通过D-Table/S-Table跟踪每GBuf/OBuf entry的依赖状态，命令到达时仅等待真正依赖的前序命令完成——无关WR-INP和MAC可乱序穿插、MAC和RD-OUT可在不同OBuf entry上并行。GQA下DCS利用dual-port GBuf/OBuf在MAC消费当前entry时预取下一批query/score，将row-reuse的KV reuse转化为真实吞吐。DPA的Dyn-Loop按runtime Tcur循环而非Tmax，Dyn-Modi按stride自动计算row/col，on-module dispatcher做VA→PA翻译
  - 硬件架构层：在AiMX PIM HUB侧新增dual-port OBuf（每bank面积0.47% of MAC unit），D-Table/S-Table（576B metadata）+ dependency-check unit（0.5% area/1.3% power overhead on PIM HUB control blocks），on-module dispatcher（<200KB buffer, 4% area overhead）。不改动DRAM bank array本身

  Baseline缺陷→PIMphony方案映射：
  | Baseline缺陷 | PIMphony方案 | 效果 |
  |-------------|-------------|------|
  | HFP按head/batch分配channel→长上下文下batch不足→MAC utilization降48% | TCP沿token维度partition→每个channel处理token segment→QK^T token>256即可full channel activation | PIM-only up to 11.3× speedup |
  | 固定WR-INP→MAC→RD-OUT timing→I/O和MAC无法重叠→小维度MAC util 14.7% | DCS entry-level dependency tracking→乱序issue→dual-port OBuf重叠数据搬运和计算 | MAC util从14.7%显著提升，DCS vs ping-pong up to 1.4× higher utilization |
  | 静态按Tmax预留KV cache→真实workload capacity util 31.0%-40.5% | DPA Dyn-Loop/Dyn-Modi+on-module dispatcher VA→PA翻译→1MB chunk lazy allocation | Capacity utilization提升至75.6% |
  | GQA row-reuse的KV复用被WR-INP transfer stalls抵消 | DCS dual-port GBuf/OBuf预取+MAC并行消费→隐藏input transfer overhead | GQA 128K模型收益更大（up to 11.3×） |
  | CENT在1M context退化至2% utilization（pipeline bubbles放大） | TCP+DCS+DPA协同消除三个瓶颈→Attention比FC更快→长上下文下系统utilization持续提升 | 1M context达46.6× speedup over CENT |
  | GPU A100虽有用但受HBM带宽/容量限制 | PIM内部带宽32TB/s (vs GPU ~2TB/s HBM)→PIMphony最大化此带宽利用 | PIMphony vs GPU-A100取得显著throughput优势，尤其non-GQA长上下文
