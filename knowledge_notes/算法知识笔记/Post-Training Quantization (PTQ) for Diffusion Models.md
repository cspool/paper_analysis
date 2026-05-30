## Post-Training Quantization (PTQ) for Diffusion Models

术语是什么？
Post-Training Quantization (PTQ) 是一种训练无关的模型量化技术，在预训练完成后直接对网络参数和/或激活值估计量化参数（scaling factor s 和 zero-point z），无需任何fine-tuning或retraining。对于扩散模型，PTQ面临的特有挑战是：(1) 激活在不同时间步（t=T→1）之间的分布范围大幅变化，静态scaling factor难以覆盖全范围；(2) 激活在每个时间步内存在显著异常值（outliers with long-tailed distributions），min-max scaling受极端值主导导致大多数正常值被过粗粒度量化；(3) 低bit（<6-bit）时clipping error和rounding error无法同时被控制。

标准PTQ量化公式：$\mathbf{x}_{\text{int}} = \text{clamp}(\lfloor \mathbf{x}/s \rceil + z, 0, 2^b - 1)$，$Q(\mathbf{x}) = s(\mathbf{x}_{\text{int}} - z)$。Q-Diffusion通过time-step-aware calibration data sampling和MSE reconstruction loss优化scaling factor；BRECQ/LCQ通过block reconstruction和per-channel量化提高精度。但现有方法在扩散模型中仅能将activation量化到8-bit，更低精度（<6-bit）时质量急剧塌陷。

从算法pipeline角度拆解术语：
```
// 扩散模型PTQ的逐层流程（以Q-Diffusion为例）
// 离线校准阶段：
calibration_data = sample_from_diffusion_steps(images, T_steps)
for layer l in 1..L:
    for time_step t in calibration_steps:
        a_t^{(l)} = forward_layer(l, x_t, t)  // 收集每层的激活值
    // MSE重建学习scaling factor
    s^{(l)} = argmin_s ||A^{(l)}(a^{(l)}) - A^{(l)}(Q_s(a^{(l)}))||²

// 在线量化推理：
for t = T..1:
    for layer l in 1..L:
        â_t^{(l)} = Q_s(a_t^{(l)})     // 使用校准的s量化激活
        ô_t^{(l)} = A^{(l)}(â_t^{(l)}) // 低精度整数矩阵乘法
```

MoDiff论文在PTQ评估中使用的方法：Q-Diffusion（time-step-aware MSE reconstruction）、LCQ（dynamic per-channel min-max quantization from BRECQ框架）、LTQ（dynamic per-tensor min-max quantization）。Weight quantization: per-channel MSE reconstruction to 4/8 bit。Activation quantization: per-channel or per-tensor dynamic scaling to 2-8 bit。效率评估使用DeepSpeed计算GBops而非实际硬件加速。

术语一般如何实现？如何使用？
Q-Diffusion开源：https://github.com/true-grub/Q-Diffusion。评估指标：IS/FID/sFID（标准生成模型指标）。主要应用于扩散模型的线性层（Conv2d、Linear, 计算瓶颈）。MoDiff证明可与任意PTQ方法正交叠加，将activation量化从8-bit推进到3-bit而无质量损失。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

---
