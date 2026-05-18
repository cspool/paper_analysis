## Metal-Embedding (ME) / 金属嵌入

术语是什么？
Metal-Embedding（ME）是一种将神经网络权重参数嵌入芯片金属互联导线3D拓扑结构中的方法学，在ASPLOS '26论文"Hardwired-Neurons Language Processing Units as General-Purpose Cognitive Substrates"中提出。传统方法（Cell-Embedding, CE）将权重编码在硅器件的2D网格中（不同constant-multiplier单元），而ME将Hardwired-Neuron重构为accumulate-multiply-accumulate单元，将权重参数表达为金属导线的源-目的地连接关系——每个FP4权重值由连接输入信号到对应累加器（POPCNT）区域的一根金属线表示。在5nm工艺下，ME将参数密度提升15×（面积比CE降低93.4%，仅0.95× SRAM面积），使gpt-oss 120B从200+芯片减少到16芯片；同时70层光罩中60层可跨芯片同质化共享（包括所有EUV层），将光罩成本降低112×。

从芯片设计角度拆解：
在芯片横截面中，ME利用不同金属层的光罩成本差异。M0-M7（FEOL+低层金属）需要EUV或DUV多次曝光（极为昂贵），M8-M11（∼60nm half-pitch）仅需DUV单次曝光（相对便宜），M12+用于电源/时钟/IO。ME将参数相关结构浓缩到M8-M11，使M0-M7的HN Array可以预制造并跨芯片共享光罩。具体流程：(1) 预制造HN Array模块在M0-M7内完成标准cell P&R；(2) 复制布局填充芯片面积，加上SoC外设、power grid和clock tree；(3) 导出布局到定制工具，读取权重参数生成TCL脚本指导M8-M11金属嵌入线连接；(4) 集成脚本到标准P&R EDA工具；(5) DRC/LVS验证和自动修复；(6) 寄生提取和post-layout仿真。论文在5nm完成sign-off：M8-M11布线密度<70%，寄生avg R=164Ω, C=7.8fF，热密度avg 0.3W/mm², peak 1.4W/mm²，DRC/LVS clean，1.0 GHz timing闭合（SSG/0.675V/125°C worst-case）。

术语一般如何实现？如何使用？
ME使用标准ASIC设计流程和EDA工具（Synopsys Design Compiler综合、IC Compiler P&R）实现。首先完成HN Array在M0-M7内的P&R，然后定制工具根据目标模型权重在M8-M11层指导金属嵌入连接，集成到整体布局中完成签核。初始tapeout光罩成本约$65M（$27.69M预制造HN Array + $2.31M×16芯片M8-M11金属化），参数更新respins仅需$37M（仅M8-M11金属化层）。该方案适用于需要将单一大型模型极致专用化的场景。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates
