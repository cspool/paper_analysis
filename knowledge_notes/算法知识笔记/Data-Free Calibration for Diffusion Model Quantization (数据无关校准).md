## Data-Free Calibration for Diffusion Model Quantization (数据无关校准)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
数据无关校准（Data-Free Calibration）是一种无需真实数据即可构建量化校准集的技术。在扩散模型量化场景下，其核心思想是：利用扩散模型本身的性质——推理以随机高斯噪声 x_T ~ N(0, I) 作为输入——将高斯噪声通过全精度模型前向传播，在不同时间步 t 采样中间激活作为校准数据。QuEST 的做法：仅需推理全精度模型数次（总计 128-256 样本/时间步），即可获得足够校准样本用于微调量化模型。这与需要真实图像数据的 PTQ 方法（如 Q-Diffusion 使用 5120 张真实图像）形成对比，完全消除了对外部数据集的依赖。数据无关校准之所以可行，是因为：扩散模型全精度前向传播本身就能产出有意义的特征分布，且权重微调的目标是对齐全精度模型输出，而非拟合真实数据分布，因此纯噪声驱动的合成数据即可满足需求。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
数据无关校准在量化 pipeline 中的流程：
```
输入：全精度模型 FP_model, 总时间步 T, 每时间步样本数 N=256
输出：校准集 activations = {a_t^i : t ∈ [0,T], i ∈ [1,N]}

for t in sample_time_steps(T):      # 采样部分时间步
    for i in range(N):               # 每步 128-256 个样本
        x_T = randn(latent_shape)    # 采样标准高斯噪声
        a_t^i = FP_model.forward(x_T, timestep=t)  # 全精度前向
        store(a_t^i)                  # 存储中间激活
```
校准集随后用于 TLA 和 CMA 的 MSE 损失计算：L_TLA = ΣE_t[||FP_TE(t) - Q_TE(t)||²]，其中 FP_TE(t) 来自校准集存储的全精度激活，Q_TE(t) 来自量化模型在同一噪声输入下的前向。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 QuEST 中，数据无关校准直接嵌入微调 pipeline：首先用高斯噪声推理全精度模型获取各层激活（作为 ground truth），然后在每次微调迭代中，将相同的噪声输入量化模型，计算量化输出与预存全精度输出的 MSE。该方法也适用于 Stable Diffusion 等文本到图像模型——尽管文本条件不同，但推理仍从随机噪声开始，因此合成校准集同样有效（Stable Diffusion 使用 128 样本/时间步）。相比 EfficientDM 等方法依赖 ImageNet 等真实校准集，数据无关方式消除了数据采集和隐私顾虑。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
