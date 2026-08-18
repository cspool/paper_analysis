## 量化感知训练（QAT，Quantization-Aware Training）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 量化感知训练（QAT）是在训练/微调过程中模拟量化效果、让模型参数适应低比特量化的训练方法：前向传播插入伪量化（fake quant）算子（量化到低比特再反量化），保持权重为浮点同时让损失看到量化噪声；反向传播用直通估计器（STE）把梯度穿过不可微的量化算子（vault 笔记 knowledge_notes/算法知识笔记/Quantization-Aware Training (QAT).md 与 Straight-Through Estimator (STE).md 有详细定义：量化范围内梯度=1、范围外=0 并被 clamp 截断）。与 PTQ（训练后量化，仅校准不重训）相比 QAT 精度更高但成本高。论文（ISCA 2026）用 QAT [81]（Nagel et al., "Overcoming oscillations in QAT", ICML 2022）把 ResNet50/MobileNetV2 量化到 3/4-bit（及 8-bit 对照）用于 SiPh 加速器精度评估，QAT 同时学习激活与权重的动态范围（量化器在推理时使用）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 论文的 QAT 流程（Sec-III-C/III-D）：①QAT 训练（每数据集分别训练），前向含激活/权重量化器（伪量化），学最优动态范围；②推理时按所学范围做低比特量化；③在每层注入 SiPh 非理想因素（pre-hook 加调制器非线性、post-hook 加 ISI 分布与 AWGN）评估精度。伪代码：
  ```
  # QAT 训练（每批）
  for x, y in train_loader:
      x_q = Q_act(x)          # 伪量化激活到 4-bit（scale=动态范围/15，QAT 学范围）
      w_q = Q_wt(w)           # 伪量化权重
      y_hat = layer(x_q, w_q) # 前向（含 STE 反传量化器）
      loss = CE(y_hat, y); loss.backward()   # 梯度经 STE 穿过量化器
      w -= lr * grad_w        # 权重保持浮点更新
  # 推理（SiPh 精度评估）
  for layer in model:
      y = layer(Q(x), Q(w))          # 低比特 MAC
      y += N(0, σ²_ISI) + Σ N(0, σ²_opt)   # ISI + 光噪声注入（post-hook）
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：PyTorch 中用 hook 或量化感知算子实现；常见框架 QAT 工具（torch.ao.quantization、Brevitas、Intel Neural Compressor 等）。论文用法：图像模型走 QAT（3/4-bit 精度损失小），LLM（Qwen2.5-7B）因算力限制改用 AWQ + 激活量化（post-training）而不用 QAT。使用注意：QAT 学到的最优动态范围直接决定量化器 scale；评估 SiPh 加速器时量化后的激活值在电平均匀分布（论文瞬态仿真假设的依据）。论文关键数据：3/4-bit 低比特模型对噪声更不鲁棒（MobileNetV2 3/4/8-bit 需 SNR>20/>12/>2；ResNet50 >12.5/>8.33/>2.5），说明 QAT 恢复精度后仍需信号完整性补偿。

涉及论文标题：
- Shining Light on Silicon Photonic DNN Accelerators
