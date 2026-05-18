## Hardwired-Neuron (HN) Architecture / 硬连线神经元架构

术语是什么？
Hardwired-Neuron (HN) 是HNLPU论文提出的基础算术单元架构，将传统神经网络的multiply-accumulate重构为accumulate-multiply-accumulate。通过三步优化实现：(1) Weight Constancy——权重固定后乘法器优化为multiply-by-constant，FP4 multiply-by-constant比GPU的FP4 multiplier小约6×；(2) Distributive Law——提取并合并公共乘法器，如a(x1+x2+...+xn)替代ax1+ax2+...+axn，节省乘法器并减少累加宽度；(3) Bit-Serialization——输入信号从LSB到MSB序列化，以CSA树替代单周期累加，以时间换面积。HN的关键结果是：将权重参数的嵌入从硅器件（Cell-Embedding）提升到金属互联（Metal-Embedding），即硅器件cell变得参数无关。

从硬件架构角度拆解：
在HN中，一个神经元处理gpt-oss 120B的2,880个输入（hidden size）。(1) 输入信号经bit-serial化（1b宽度，LSB-first）进入HN；(2) 根据每个输入的权重值，通过金属线将其路由到对应颜色编码的累加器区域（FP4共16个唯一值=16个区域，每个区域内含POPCNT进行1b累加）；(3) 区域累加结果通过16个4b乘法器乘以对应权重值（参数无关的固定乘法器）；(4) 4b×16加法树求和输出。图4对比了CE（2,880个4b乘法器+8b×2,880加法树）与ME（1b串行输入+POPCNT累加+16个4b乘法器+4b×16加法树），ME的面积显著更小。硅器件cell（POPCNT、乘法器、加法器）在所有芯片间完全相同——仅金属线连接的源-目的地因权重而异。

术语一般如何实现？如何使用？
HN以Verilog RTL实现，经Synopsys Design Compiler综合和IC Compiler布局布线在5nm工艺。HN Array由大量HN单元阵列组成，占据芯片面积的69.3%但仅消耗24.9%功耗（因MoE稀疏激活和bit-serial低活跃度）。HN是实现Hardwired LPU的基础构建块——将权重嵌入从cell提升到metal是后续Sea-of-Neurons光罩共享的前提。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

