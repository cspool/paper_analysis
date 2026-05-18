## Bit-Serialization with Carry Save Adder (CSA) for Neural Computation / 神经网络计算中的位串行化与进位保存加法器

术语是什么？
Bit-Serialization with CSA是一种硬件算术优化技术，将多bit输入信号从LSB到MSB串行化（每次处理1 bit），利用Carry Save Adder（CSA）树以多时钟周期替代单周期全宽累加，从而以时间换面积。在HNLPU的Hardwired-Neuron架构中，这是第三步关键优化：gpt-oss 120B的hidden size为2,880，若单周期累加需要大规模加法器，而bit-serial化将单周期累加展开为多周期CSA树（CSA将三个输入压缩为两个输出，避免进位传播延迟），显著减小了硬件面积。CSA是计算机算术中的经典技术（Kai Hwang, 1979），其特点是在进位传播之前通过多级3:2压缩器逐步减少操作数。

从kernel调度角度拆解：
HN中的bit-serial计算过程（伪代码）：
```
// 输入: x[0..2879] 每个4-bit，串行化LSB-first
// 权重: w[0..2879] 由金属线编码（源→颜色区域的连接）
// 16个颜色区域，每个有1-bit POPCNT累加器

for bit_pos = 0 to 3:  // 4-bit精度，4个周期
    for 每个颜色区域 color in 16:
        // POPCNT: 统计路由到该区域且当前bit=1的输入数
        count[color] = POPCNT(输入x_i中满足w_i=color且x_i[bit_pos]=1者)
        // CSA树累加: 将count压缩为partial sum和carry
        (psum[color], carry[color]) = CSA_tree(count[color], psum[color], carry[color])
    
    // 每bit位置累加完成后移位
    psum[16] <<= 1
    
// 4周期后: psum[color] = Σ x_i (对于所有权重=color的输入)
// 乘法阶段:
for color in 16:
    result[color] = psum[color] × weight_value[color]  // 16个固定4b乘法器

// 最终加法树:
output = SUM(result[0..15])  // 4b×16加法树
```
关键设计：CSA树避免每步进位传播（仅最后需要），使关键路径极短适合高频；bit-serial化将2,880宽的全并行累加变成4周期流水线，面积大幅减少。

术语一般如何实现？如何使用？
HN以Verilog实现bit-serial CSA树。CSA采用Full Adder阵列级联，每级将3个输入压缩为2个输出（和+进位）。在HN的16个颜色区域中，每个区域独立配有CSA树，4个bit周期后通过移位累加完成全精度累加。该技术配合weight constancy和distributive law使HN的面积比CE降低93.4%。bit-serial以吞吐换面积——HN仍可在1.0 GHz运行，通过6级pipeline×36层=216并发batch来补偿每操作多周期。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

