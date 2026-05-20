## PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

- 属于硬件架构的实现是什么？实验比较什么？
  在现有PIM module架构（AiMX specification）基础上引入三项硬件修改，使PIM controller支持长上下文LLM decoding的高效执行：(1) Expanded Dual-Port Output Buffer（OBuf）：将原有PIM output register扩展为multi-entry output buffer，使用dual-port memory，支持MAC消费当前GBuf entry的同时写入下一批输入或读出已完成输出。OBuf面积估计为MAC unit area的0.47%/bank（CACTI综合估计）。(2) DCS Controller Logic：在PIM HUB controller内增加Dependency Table（D-Table，记录每个GBuf/OBuf entry最近访问命令）、Status Table（S-Table，记录命令ID/完成时间/OBuf is-MAC flag）和dependency-check unit（验证per-entry hazards）。D-Table+S-Table合计576B metadata per controller。DCS control blocks在PIM HUB全部control block中带来0.5% area和1.3% power增加。(3) On-Module DPA Dispatcher：在PIM module内实现轻量级pseudo-MMU，包含instruction buffer、configuration buffer和VA2PA table（<200KB总buffer容量，远小于典型PIM HUB 512KB GPR容量），在PIM HUB内执行运行时虚拟地址到物理KV cache chunk的翻译。Dispatcher面积开销4%。这三项修改不改变DRAM bank array本身，仅在PIM HUB侧增加controller logic和buffer。实验比较CENT baseline和NeuPIMs baseline的throughput、MAC utilization、energy breakdown、capacity utilization，以及context length 4K-1M scalability下的PIMphony硬件效率提升。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  使用Ramulator-based cycle-accurate simulator（论文未给出具体链接），集成AiMX PIM architecture parameters。论文修改了CENT[16]和NeuPIMs[21]两个simulator，在其中实现PIMphony的on-module dispatch logic、I/O buffering、DCS dependency tracking和expanded output buffer。DRAM command timing和resource contention按AiMX[62] PIM specification校准。硬件overhead通过CACTI[8]估计OBuf面积，通过综合估计DCS controller和DPA dispatcher area/power。

- 模拟器模拟什么的性能，修改了什么。
  Ramulator-based simulator模拟PIM module的cycle-accurate执行，包括DRAM command时序、PIM channel内部MAC pipeline、GBuf/OBuf读写、inter-channel通信和PIM HUB调度。修改：(1) 在simulator中建模dual-port OBuf和multi-entry GBuf的I/O-aware buffering行为；(2) 加入DCS dependency tracking logic（D-Table/S-Table查询和更新、per-entry hazard check、乱序command issue）；(3) 加入DPA dispatcher的VA-to-PA address translation和1MB chunk lazy allocation逻辑；(4) 建模on-module instruction/configuration buffer和VA2PA table的访问延迟。Simulator结合AiMX的DRAM command时序约束（tWR-INP、tMAC、tRD-OUT等）和bank-level resource contention建模。系统级评估覆盖7B/72B模型、4K-1M context length、128GB-1024GB system capacity。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未开源。PIMphony硬件架构在PIM module内的作用流程：
  1. PIM HUB接收来自compiler/runtime的PIM instruction sequence（含Dyn-Loop/Dyn-Modi动态指令）→DPA dispatcher根据request ID查询VA2PA table，将virtual row/col operand翻译为物理KV cache chunk地址。
  2. CMD scheduler将翻译后的commands推入per-channel CMD queue→DCS controller在issue前查询D-Table/S-Table：检查WR-INP写入的GBuf entry是否被前序命令占用、MAC读取的GBuf entry是否已完成写入、RD-OUT读取的OBuf entry是否已完成MAC累加。无hazard则立即issue，有依赖则等待对应entry ready。
  3. WR-INP通过internal bus将32B input tile写入GBuf指定entry→MAC unit读取GBuf entry和DRAM bank row data，执行dot-product并累加到OBuf对应entry→RD-OUT从OBuf读出结果经PIM HUB/EPU返回。
  4. dual-port OBuf允许：port A被MAC写入当前OBuf entry时，port B可同时读出前序已完成的OBuf entry（RD-OUT），或反之。DCS跟踪entry级依赖避免RAW/WAR/WAW hazard。
  5. 以GQA row-reuse场景为例：DCS在MAC消费当前GBuf query entry时，通过dual-port的另一个端口预取下一批query/score到其他GBuf entry；MAC继续处理当前DRAM row上的多个query head时，RD-OUT可并行读出已完成结果。相比ping-pong buffering baseline（需等待两个region均idle才能切换）消除hand-off pipeline stalls，DCS实现up to 1.4× higher compute-unit utilization。
