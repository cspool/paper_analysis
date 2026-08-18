## LSQ（Learned Step Size Quantization，学习步长量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LSQ（Esser 等，ICLR 2020，IBM）是一种把量化步长（step size）作为可学习参数训练的量化方法：每个量化层维护一个浮点步长 s，量化操作 v̄ = clip(round(v/s), -QN, QP)（QN/QP 为量化整数范围），通过直通估计器（STE）让梯度穿过 round/clip 回传到 s，训练中联合学习权重与步长，从而找到比固定范围（如 min/max 标定）更优的量化间隔。DESSCam 用它把 Robust ViT 量化到 INT8 后部署到 STM32N6。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 前向：v_bar = clip(round(v/s), -QN, QP)      # s 可学习
# 反向（STE）：dL/dv_bar = dL/dv_bar；v_bar 对 v 的梯度在量化区间内视为 1
# 步长梯度：dL/ds = sum(-v/s + round(v/s))     # 区间内
#                    -QN 或 QP                  # 越界部分
# 更新：s <- s - lr * dL/ds（与权重同训练）
```
相比 QAT 固定范围、LSQ 的范围端点随 s 连续可调，INT8 下精度损失更小；DESSCam 的量化 pipeline：LSQ INT8 → ONNX（QDQ 格式）→ STM32Cube.AI 部署。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 PyTorch/TensorFlow 量化训练流程（LSQ 官方提供 STE 自定义算子），也可作为 QAT 一部分插入任意网络；使用场景：边缘 NPU 仅支持 INT8 的部署（Neural-ART NPU、NPU 类硬件）。注意：量化后的卷积/线性层可映射到 NPU，而 LayerNorm/Softmax 等非线性算子通常需回落 CPU 或保持高精度，DESSCam 即采用该异构切分。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
