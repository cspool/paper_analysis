## 时序卷积网络（Temporal Convolutional Network, TCN）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TCN 是 Bai、Kolter、Koltun（2018，arXiv:1803.01271）提出的用于序列建模的卷积网络，核心是因果卷积 + 空洞卷积 + 残差连接：每个时间步 t 的输出只依赖 t 及之前的输入（因果、无未来信息泄漏），空洞卷积让感受野随深度指数增长，残差块保证深网络可训练。与 RNN/LSTM 相比，TCN 可并行处理整个输入序列（无串行状态更新）、感受野可调、梯度稳定、训练内存低。在 Moirai 中，设计空间探索（Figure 3）显示同参数预算（≈380 参数）下 TCN 预测准确率最高：RNN/LSTM 的理论无限时序感受野在实际 L1D 场景下被串行状态更新的延迟瓶颈抵消——推理时间过长使预测"迟到"，完全抵消长程模式覆盖收益；TCN 的卷积并行 + 空洞感受野更适合低延迟硬件实现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Moirai 中 CaPNet 的 TCN 前向（3 层，通道 [8,4,2]，输入 10 个历史 delta）：
  ```
  # 输入：A_0 = [delta_{t-9}, ..., delta_t]  (10 个 delta)
  for layer i in 1..3:                      # 通道 8 → 4 → 2
      for channel k in 1..C_i:              # 每层 C_i 个并行卷积滤波器
          Ac_i^k = sign( bitcount(A_{i-1} ⊙ W_bin^k) )   # 空洞因果卷积(二值化)
      A_i = 拼接所有通道输出
  D_pred = A_3                               # 预测下一个 delta
  ```
  整个输入窗口广播到各通道，每个通道独立学习一个空间-时间模式（通道指并行滤波器，非硬件布线路径）。
- 关键点：CaPNet 是 TCN 的 BNN 化（权重/激活 1-bit），空洞感受野让 3 层网络捕获 delta 序列中的长程依赖；序列式 TCN 能在模式起始点就前瞻识别（predictive lookahead），这是其 92.37% 及时性的来源之一。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：1D 全卷积（FCN），每层保持输入长度，空洞系数随层指数增长（d=1,2,4,...），残差块含两层空洞因果卷积 + weight norm + ReLU + dropout；PyTorch 有官方 TCN 教程与社区实现。Moirai 的硬件实现是 FCC（前向卷积单元，3 输入对应 kernel size 3）+ BCC（反向，10 输入对应对整个时序序列长度，跨所有时间步累计共享权重梯度）。使用场景：语言建模、时间序列预测（客流/盾构预测等）、以及本论文的硬件预取器；BTCP 是先前把 B-TCN 用于预取（drop-in 设计，PC 流跟踪 + 无片上反向传播 + 无污染控制，4.5KB/134-cycle，只能放 L2）。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
