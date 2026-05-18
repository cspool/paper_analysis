## Sea-of-Neurons Architecture / Sea-of-Neurons架构

术语是什么？
Sea-of-Neurons是一种Metal-Programmable Structured ASIC架构，在ASPLOS '26 "Hardwired-Neurons Language Processing Units"论文中提出。该架构预制造Hardwired-Neuron阵列（HN Array）作为同质化基底，仅通过高层金属层（M8-M11）自定义互联来"编程"权重参数。其核心思想继承自历史上多次出现的Structured ASIC概念（1970s gate array、1990s sea-of-gates、2000s Altera HardCopy、2020 Intel eASIC N5X），但将预制造单元从通用逻辑门替换为Hardwired-Neuron算术单元。60/70层光罩跨芯片同质化共享（包含所有EUV层），仅10层DUV光罩因芯片而异，使初始tapeout光罩成本从$480M降至$65M。

从芯片设计角度拆解：
芯片横截面组织（自上而下）：(1) M12+顶层金属：电源分配网络、时钟树、IO外设；(2) M8-M11（10层DUV）：Metal-Embedding层，承载权重参数，每芯片独立光罩（$2.31M/chip）；(3) M0-M7 + FEOL（40 DUV + 12 EUV）：器件层和局部互联，预制造HN Array，所有芯片共享同质化光罩（$27.69M）。分层策略利用光罩成本差异——FEOL和低层金属需要EUV（cost factor 6× vs DUV）或DUV多次曝光，而M8-M11使用DUV单次曝光。参数更新respins仅需重制M8-M11的10层DUV光罩。在5nm工艺下，该方案与标准ASIC EDA流程完全兼容：HN Array在M0-M7内完成P&R后复制填充die area，然后运行定制脚本在M8-M11连接金属嵌入线。Murphy模型（D0=0.11 def/cm²）估算yield约43%。

术语一般如何实现？如何使用？
实现流程：标准ASIC EDA完成HN Array在M0-M7内的P&R→复制阵列填充芯片面积→定制工具根据模型权重生成TCL脚本→脚本集成到P&R工具完成M8-M11金属化→DRC/LVS sign-off→tapeout。不同于传统Structured ASIC的通用门阵列，Sea-of-Neurons预制造的是专用的神经元算术单元，在LLM时代首次使hardwired实现具有经济可行性。该架构支持周期性模型更新（年度LLM参数respins），每次仅需$37M而非$480M全光罩重制。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates
