## Constant Multiply-and-Accumulate (CMAC) / 常数乘累加单元

术语是什么？
Constant Multiply-and-Accumulate (CMAC) 是一种针对固定权重优化的硬件乘累加单元。在hardwired神经网络中，由于权重参数物理固化在芯片中，乘法器不需要支持任意两数的乘法，而是仅需实现"乘以常数"——这比通用乘法器的布尔复杂度低数倍。在HNLPU论文中，FP4 CMAC比GPU的FP4通用乘法器小约6×；累加也可利用EDA工具的常数优化。CMAC的极致形式是HN中的Metal-Embedding——权重由金属连线表达（源-目的地路由），硅器件仅执行参数无关的POPCNT和固定乘法操作。

从kernel调度角度拆解：
CMAC与通用MAC的差异（以1×1024 input × 1024×128 FP4 weight为例）：
```
// 通用MAC Array (MA): 
// 1024个MAC单元，每个从SRAM读取weight
for i in 0..127:  // 128个输出
    for j in 0..1023:  // 1024个输入
        weight = SRAM_read(addr[i][j])  // 从SRAM取FP4权重
        acc[i] += x[j] × weight          // 通用FP4乘法器

// CMAC (Cell-Embedding):
// 权重固定→乘法器被优化为multiply-by-constant
for i in 0..127:
    for j in 0..1023:
        acc[i] += x[j] ×_constant W[i][j]  // 优化后的constant乘法器
```
论文的实验显示：相比MA（64KB SRAM + 1024 MACs），CE和ME分别将执行周期降低至MA的~1/100（全并行计算+无SRAM fetch），ME的能耗最低（消除SRAM访问+较小面积减少leakage）。

术语一般如何实现？如何使用？
CMAC通过逻辑综合工具的常数传播优化自动实现——当综合工具检测到一个乘法器的一个输入为常数时，自动简化电路。在HNLPU中，CMAC演进为更极致的HN架构——先POPCNT累加输入再乘以常数权重值（accumulate-multiply-accumulate），用16个通用4b乘法器替代2,880个CMAC。CMAC适合无法用Metal-Embedding的常规hardwired实现（如灵活的printed/flexible electronics场景）。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

