## Weight Hardwiring / Hardwired Weights（权重固化/硬连线权重）

术语是什么？
Weight Hardwiring是将神经网络的权重参数物理固化到芯片电路中（而非存储于SRAM/DRAM并动态加载）的实现方法。其核心理念来自1980年代的VLSI神经网络实现（Graf et al., 1988）和Hinton的"mortal computing"论点。在hardwired实现中，权重不再是"数据"而是"电路的一部分"——乘法器被优化为multiply-by-constant（CMAC），权重参数在芯片制造时确定且物理不可变。HNLPU将这一概念推至新高度：通过Metal-Embedding将权重编码在金属导线的3D拓扑中，而非2D硅器件grid。

从算法pipeline角度拆解：
Hardwired weight inference与传统weight-loading inference的算法层对比：
```
// 传统推理（GPU）:
for each token in autoregressive loop:
    for layer in 1..36:
        W_qkv = HBM_read(weights_addr[layer].qkv)  // 从HBM加载权重
        Q, K, V = X × W_qkv                          // 通用GEMM
        // ... attention, FFN with repeated weight loading ...

// Hardwired weight推理（HNLPU）:
for each token:
    for layer in 1..36:  // 36层pipeline并行
        // 无weight loading步骤——权重存在于金属连线中
        // 输入信号通过金属线自动路由到对应的POPCNT累加器
        Q, K, V = HN_Array_compute(X)  // 即电路本身
        // ... attention on VEX, FFN on HN Array ...
```
关键区别：hardwired方案消除了每个decoding step的weight fetch开销。gpt-oss 120B有约120B参数，在GPU上每次decoding需反复读取这些权重（占大部分系统功耗），在HNLPU中这些权重是零开销的电路连接。代价是权重不可更新（除非重新制造芯片），但Sea-of-Neurons架构将参数更新respins的NRE从$480M降至$37M。

术语一般如何实现？如何使用？
实现方式演进：(1) 早期VLSI/光学/printed flexible电路直接hardwire小型网络；(2) CMAC方案——用constant multiplier替代通用multiplier，但仍嵌入在硅器件cell中（$6B光罩成本）；(3) HNLPU的Metal-Embedding——将权重从硅器件提升到金属互联（15×密度提升，112×光罩成本降低），使120B级模型在经济上可行。适用于长期高容量部署（年均数百万token服务），与GPU短期模型开发形成互补。论文提出LoRA for post-deployment updates（~1% field-programmable HNs at side-channel容纳动态权重）作为未来方向。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

