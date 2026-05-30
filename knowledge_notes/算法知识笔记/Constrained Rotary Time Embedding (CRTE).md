## Constrained Rotary Time Embedding (CRTE)

术语是什么？
Constrained Rotary Time Embedding (CRTE) 是 OmniVinci 提出的绝对时间戳编码方法，用于将时间信息直接注入 omni-modal 嵌入向量中。CRTE 继承 RoPE (Rotary Position Embedding) 的旋转编码思想，但将其从编码序列位置扩展为编码绝对时间戳。核心创新是引入最大时间范围 $T_{\max}$ 约束和几何级数频率设计，实现多尺度时间编码——高频维度捕获细粒度时间差（如毫秒级声音事件），低频维度编码粗粒度长时间关系（如场景切换）。

三个计算阶段：
1. **基础频率生成**: $\omega_i = 2\pi / (T_{\max} \theta^{i/C}), i=0,...,C-1$，其中 $\theta \geq 1$ 控制频率缩放，$C$ 为嵌入维度，$T_{\max}$ 定义最粗时间分辨率。小 $i$（前几个维度对）分母小 → $\omega_i$ 大 → 高频 → 对细粒度时间差敏感；大 $i$（后几个维度对）分母大 → $\omega_i$ 小 → 低频 → 编码长程时间关系。

2. **频率调制**: $\Omega_{i,j} = \omega_i \cdot t_j$，将基础频率与实际时间戳 $t_j$ 相乘。

3. **旋转变换**: $\text{CRTE}(\mathbf{x}, \Omega) = \mathbf{x} \odot \cos(\Omega) + \text{RotateHalf}(\mathbf{x}) \odot \sin(\Omega)$，其中 $\text{RotateHalf}(\mathbf{x}) = [-x_2, x_1, -x_4, x_3, ..., -x_C, x_{C-1}]$，将 $C$ 维嵌入分成 $C/2$ 个独立的 2D 旋转平面。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
CRTE 在 TEG 之后、LLM 输入之前应用，作为一个无参数的确定性变换层。伪代码：
```
输入: embedding x [C], timestamp t_j, T_max, theta

# 阶段1：基础频率 (可预计算)
omega = [2*pi / (T_max * theta^(i/C)) for i in range(C)]

# 阶段2：频率调制
Omega = [omega[i] * t_j for i in range(C)]

# 阶段3：旋转变换
x_rotated = zeros(C)
for i in range(0, C, 2):
    cos_val, sin_val = cos(Omega[i]), sin(Omega[i])
    x_rotated[i]   = x[i] * cos_val - x[i+1] * sin_val  # 2D 旋转
    x_rotated[i+1] = x[i+1] * cos_val + x[i] * sin_val

输出: x_rotated  # 时间编码后的嵌入
```

与 RoPE 的关键区别：(1) RoPE 编码序列位置 $pos$，CRTE 编码绝对时间戳 $t_j$；(2) CRTE 引入 $T_{\max}$ 约束，使频率范围可控；(3) CRTE 的频率呈几何级数分布，天然实现多尺度。消融实验中 CRTE (50.25) 优于 Learned Time Embedding (47.30) 和 RoTE (47.80)，证明约束频率+旋转编码对绝对时间的编码效果最好。

术语一般如何实现？如何使用？
基于 PyTorch 实现，作为无参数模块嵌入到模型的前向传播中。$T_{\max}$ 和 $\theta$ 为超参数，OmniVinci 论文中未明确给出具体值，通常 $T_{\max}$ 设为视频最大时长（如 120s），$\theta$ 通过消融确定。对于视觉嵌入，$t_j$ 为帧采样时间戳；对于音频嵌入，$t_j$ 为音频帧的采样时间点。CRTE 与 TEG 组合使用：TEG 提供相对时序嵌入序列的 token 排列，CRTE 在此基础上为每个 token 注入绝对时间信息。CRTE 在视频理解、audio-visual synchronization、temporal reasoning 等任务中表现突出。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM
