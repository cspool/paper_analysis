## 空洞卷积（Dilated Convolution）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空洞卷积（dilated/atrous convolution）是在卷积核元素间插入空洞（跳过步长 dilation d）的卷积，感受野从普通卷积的线性增长（每层 +k−1）变为指数增长（第 L 层感受野 ≈ 1 + Σ(k−1)·d_L，d 逐层翻倍时指数扩展），而不增加参数与计算量。源自 WaveNet（音频生成），是 TCN 的核心组件。在 Moirai 中，空洞卷积让仅 3 层、≈380 参数的浅 TCN 也能覆盖 delta 序列中的长程依赖——这是"极简硬件预算下保持泛化能力"的关键。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一维空洞因果卷积（Moirai 的 CaPNet 中）：
  ```
  # 卷积核大小 k=3，第 i 层空洞系数 d_i，输入 A（长度 10 的 delta 序列）
  for t in range(L):                                  # L=10，逐时间步
      acc = 0
      for j in range(k):                              # j=0,1,2
          idx = t - j*d_i                             # 空洞：只取过去 d_i 步前的点
          if idx >= 0:
              acc += bitcount(A_bin[idx] ⊙ W_bin[j])  # 二值化乘加
      Ac[t] = sign(acc)
  ```
  例：d=1（第 1 层）覆盖 t,t-1,t-2；d=2（第 2 层）覆盖 t,t-2,t-4；d=4（第 3 层）覆盖 t,t-4,t-8——3 层即可触及 8 步前的 delta，感受野随层指数增长，参数仍只有 3×通道数。
- 因果性：只允许看向过去（idx = t - j·d ≥ 0），保证预测只用历史，符合预取"预测未来"语义。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：深度学习框架中直接指定 dilation 参数（PyTorch `nn.Conv1d(dilation=d)`）；硬件上空洞只是改变数据访问索引/移位量，不增加乘加单元——Moirai 的 FCC 输入为 3（kernel size），硬件把空洞实现为跨周期的寄存器延迟线取数。使用场景：序列建模（TCN）、音频（WaveNet）、语义分割（DeepLab）；在 Moirai 中作为 CaPNet 的低延迟长程建模手段。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
