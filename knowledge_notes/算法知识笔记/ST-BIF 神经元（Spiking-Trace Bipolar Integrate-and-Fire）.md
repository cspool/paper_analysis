## ST-BIF 神经元（Spiking-Trace Bipolar Integrate-and-Fire）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ST-BIF（带脉冲追踪的双极积分-激发）神经元是 SpikeZIP-TF 提出、ELSA 采纳的 SNN 神经元模型：与只发 {0,1} 二值脉冲的 IF 神经元不同，ST-BIF 发三元脉冲 {-1,0,1}（双极），并在神经元内维护一个"脉冲追踪器"（spike tracer）S_t 记录历史累计脉冲数。其关键在于：在特定条件下 ST-BIF 与量化 ReLU（Q-ReLU）数学等价，因此 ANN（量化）→SNN 转换无损，SNN 精度与 QANN 完全一致。ELSA 论文明确指出：转换损失是 IF 式 SNN 相对 ANN 精度下降的主因（conversion errors），ST-BIF 消除该损失。
- 三步动力学（ELSA 论文 Eq.1-3）：① 积分 V̂_t = V_{t-1} + Σ x_{i,t}·w_i（x 为 {-1,0,1} 预突触脉冲，w 为突触权重）；② 激发 y_t=Θ(V̂_t,V_thr,S_t)：V̂_t≥V_thr 且 S_t<S_max → +1；V̂_t<0 且 S_t>S_min → −1；否则 0；③ 更新 V_t=V̂_t−y_t·V_thr（soft reset）、S_t=S_{t-1}+y_t。S_max/S_min 是 tracer 上下界，对应 Q-ReLU 的 clip 上/下界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 作为 SNN 推理 pipeline 的最小计算单元，ST-BIF 把"乘法-激活"（QANN 的 MAC + ReLU）拆成"加法-脉冲-累积"：
```
# 一个输出神经元的单时间步（输入脉冲 x_i ∈{-1,0,1}，权重 w_i 4-bit）
acc = Σ_{i=1..N} x_i * w_i          # 硬件里 = 加法树（x=0 跳过，x=-1 权重取二补码）
V = V + acc
if V ≥ V_thr and S < S_max: y = +1
elif V < 0 and S > S_min: y = -1
else: y = 0
V = V - y*V_thr;  S = S + y          # 输出 y 成为下一层输入脉冲
```
- 例（ELSA Fig.3/Fig.10c）：输入 spike batch (0,1),(0,3) 触发读取权重矩阵第 2 行 [2,2,3,3] 与第 4 行 [1,3,1,1]，加法树累加得膜电位行 [3,5,4,4]，fire 组件结合 spike tracer 判定激发并回写膜与 tracer。
- Annotations：N 是每个输入 spine 的突触数；V_thr 阈值（如 8-bit 整数）；S_max/S_min 对应 Q-ReLU 的量化上下界；负脉冲 y=-1 使权重在硬件中按二补码取反后累加。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：SpikeZIP-TF（arXiv 2406.03470，GitHub Intelligent-Computing-Research-Group/SpikeZIP_transformer）在转换时把 QANN 的量化 ReLU 层一一替换为 ST-BIF 神经元，输出 S_T = clip(floor((Σ_t V_t^in + V_0)/V_thr), S_min, S_max)，实现 ANN↔SNN 精确等价；SpikingJelly 已集成 ann2snn.SpikeZIPTFQANNRecipe 路径。硬件侧：ELSA 把它实现为 ST-BIF 神经元电路 = 16 输入加法树 + fire 组件（读 spike tracer 与膜电位判激发）+ update 组件（回写膜与 tracer）；每 PE 128 个该电路、每周期 1024 次加法；路由器内的 SSoftmax/SLayerNorm 单元也复用少量 ST-BIF 电路。ELSA 全部 benchmark（VGG16/ResNet18-101/ViT-S/YOLOv2，4-bit 权重、T=32）精度与 QANN 一致（如 ResNet50 ImageNet 75.60%）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
