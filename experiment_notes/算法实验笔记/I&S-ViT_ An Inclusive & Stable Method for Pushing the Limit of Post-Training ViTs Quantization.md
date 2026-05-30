## I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  提出Shift-Uniform-Log2 Quantizer (SULQ) 解决log2量化器对post-Softmax激活的量化低效问题，以及三阶段Smooth Optimization Strategy (SOS) 解决不同量化粒度下loss landscape的粗糙和放大问题。实验在ImageNet上比较ViT-S/B、DeiT-T/S/B、Swin-S/B在W3A3/W4A4/W6A6下的Top-1准确率，在COCO上比较Mask R-CNN和Cascade Mask R-CNN（Swin-T/S为主干）的W4A4检测/分割AP。对比方法包括PTQ4ViT、BRECQ、QDrop、PD-Quant、RepQ-ViT、FQ-ViT、APQ-ViT、Ranking-ViT、EasyQuant、NoisyQuant、Bit-shrinking等。

- 硬件平台是什么，配置是什么。
  单张NVIDIA 3090 GPU。框架为PyTorch，预训练模型来自Timm库。训练时间约31分钟（DeiT 3-bit）。

- 模型是什么。数据集和bench分别是什么。
  模型：ViT-S、ViT-B、DeiT-T、DeiT-S、DeiT-B、Swin-S、Swin-B（ImageNet分类）；Swin-T、Swin-S作为Mask R-CNN和Cascade Mask R-CNN的主干（COCO检测/分割）。数据集：ImageNet（分类），COCO 2014（检测+分割）。从数据集中随机选取1024张图片作为校准集。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/zysxmu/IaS-ViT
  
  算法Pipeline：
  1. **量化器配置**：对所有权重和矩阵乘法的输入做均匀量化（channel-wise权重，layer-wise激活），post-Softmax激活用SULQ，LayerNorm和Softmax保持全精度。
  2. **SULQ量化器**（替换标准log2量化器）：
     ```
     输入: X (post-Softmax激活), bit-width b, shift η
     X_q = clamp(round((-log2(X + η) - min_val) / s), 0, 2^b - 1)
     反量化: X_deq = 2^{-round(s * (X_q - z))} - η
     ```
     SULQ通过添加shift bias η后接log2变换再均匀量化，使量化区间完整覆盖输入域，解决标准log2量化器的"quantization inefficiency"（大量值被clamp到远端）。
  3. **SOS三阶段优化**（Block-wise reconstruction objective L_l = ||X_l - X̄_l||_2）：
     - Stage 1：全精度权重 + post-LayerNorm激活用channel-wise量化，其他激活用layer-wise量化 → 在平滑低loss的landscape下优化
     - Stage 2：通过scale reparameterization将channel-wise量化器无缝转换为layer-wise：调整LayerNorm的affine参数 β̃=(β+s⊙r₂)/r₁, γ̃=γ/r₁ 以及下一层权重 W̃_{:,j}=r₁⊙W_{:,j}, b̃_j=b_j-(s⊙r₂)W_{:,j}
     - Stage 3：量化所有权重，在量化激活+量化权重下再微调恢复性能
  4. **训练超参**：Adam优化器，权重lr=4e-5（cosine衰减），weight decay=0，量化参数校准后固定不优化。ImageNet batch_size=64，6-bit用200 iterations，其他用1000 iterations。η通过grid search选取最小化量化误差的值。
