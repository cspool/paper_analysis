## Robust ViT（CNN-Transformer 混合眼动追踪模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Robust ViT 是 DESSCam 为高稀疏事件数据设计的轻量 CNN-Transformer 混合模型：conv stem（两层 depthwise-separable 卷积，输出 128 维，引入局部归纳偏置）+ conv enhancement（两层 3×3 卷积替代标准 ViT 的位置嵌入，在 token 序列形成前做跨 patch 交互，增强局部空间信息）+ 3 个 transformer encoder（每个含 8 head、128 维多头自注意力）+ 平均池化 + 检测头（两层全连接 + sigmoid 输出 gaze 坐标）。设计动机：标准 ViT 非重叠 tokenization 会丢失邻域信息、形成孤立 patch 表示；在高稀疏输入（50× 下采样）下，前置 CNN 层增强局部纹理并恢复被稀疏采样破坏的上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
x = conv_stem(event_frame)        # 两层 depthwise-separable conv -> 128 维
x = conv_enhancement(x)           # 两层 3×3 conv 替代位置嵌入
tokens = flatten(x)[ESS_mask]     # 应用同一 ESS 掩码生成稀疏 token
for _ in range(3):                # 3 个 transformer block
    tokens = MHA(tokens)          # 8 head、128 维
z = avg_pool(tokens)
(x_pred, y_pred) = sigmoid(fc(fc(z)))
AE = arccos(v_pred·v_gt / (|v_pred||v_gt|)),  v=(x,y,L0)
```
计算量分布：早期卷积层在全分辨率操作、承担大部分 MAC；transformer 只处理 ESS 掩码后的稀疏 token，计算量大幅下降——这一设计同时保证精度与部署友好（卷积层可上 NPU，transformer 计算量小）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 MobileViT、CvT、Conformer 同属 CNN-Transformer 混合族（卷积提供局部归纳偏置、transformer 提供全局相关性）；训练 batch 64、500 epochs（EVBEYE）。部署：LSQ 量化到 INT8、ONNX 导出、STM32Cube.AI 异构部署（卷积/线性在 Neural-ART NPU、LayerNorm/Softmax 等在 Cortex-M55）；由于 Neural-ART NPU 对 transformer 块无原生加速，计算量集中于卷积层是关键部署优势。效果：50× 压缩率 AE 0.5°，压缩率 1×–50× 全程 AE < 2°。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
