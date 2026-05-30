## Modulated Diffusion (MoDiff): Accelerating Generative Modeling with Modulated Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Modulated Diffusion (MoDiff)，一个加速扩散模型采样过程的框架，包含两个核心算法：(1) Modulated Quantization——利用扩散过程相邻时间步之间激活的相似性，将每层线性算子的计算从直接量化激活重构为量化时序差分：o_t = A(Q(a_t - a_{t+1})) + o_{t+1}。时序差分 a_t - a_{t+1} 的分布范围比原始激活小10×以上，更集中且异常值更少，因此量化误差显著降低；(2) Error-Compensated Modulation——通过中间变量 â_t = Q(a_t - â_{t+1}) + â_{t+1} 跟踪量化误差 e_t = a_t - â_t，在下一时间步将误差反馈到输入中（â_t 替代 a_{t+1} 作为差分基准），实现误差补偿而非累积，理论证明误差以指数速率(2c)^{T-k-1}递减（相比标准调制的2^{T-k-1}c线性以上累积）。

  实验比较的算法baseline：(1) Q-Diffusion (Q-Diff)——基于MSE reconstruction loss的PTQ方法，使用time-step-aware校准数据采样，8/8 bit为强baseline；(2) Dynamic Channel-wise Quantization (LCQ)——基于BRECQ框架，per-channel min-max动态量化；(3) Dynamic Tensor-wise Quantization (LTQ)——per-tensor min-max动态量化，更硬件友好；(4) Full Precision Activation (32-bit) 作为上限。评估指标：IS (Inception Score)、FID (Fréchet Inception Distance)、sFID (Sliced FID)、Precision/Recall、GBops（通过DeepSpeed计算理论binary operations）。核心实验证实在W8A4及以下位宽时MoDiff的优势急剧扩大——baseline方法在W8A4时质量塌陷（FID>300），MoDiff保持FID≈4。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体GPU型号用于实验。效率评估使用DeepSpeed计算GBops（binary operations per denoising step per image），而非实际wall-clock hardware speedup。论文明确说明"Implementing acceleration on specialized hardware is beyond the scope of this work"。Weight quantization使用MSE reconstruction method（Q-Diffusion checkpoint），activation量化使用动态per-channel/tensor min-max scaling。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) DDIM on CIFAR-10（32×32, 100 denoising steps）；(2) Latent Diffusion Models (LDM-4) on LSUN-Bedrooms（256×256, 500 sampling steps）；(3) LDM-8 on LSUN-Churches（256×256, 200 steps）；(4) Stable Diffusion v1.4 with DPM-Solver on MS-COCO 2014（50 steps, 30K images generated）；(5) DiT-XL/2 on ImageNet 256×256（50 steps, 10K images）；(6) SDXL-Turbo (few-step) on MS-COCO。数据集：CIFAR-10（50K generated images for evaluation）、LSUN-Bedrooms、LSUN-Church-Outdoor、MS-COCO 2014、ImageNet 256×256。评估指标：IS/FID/sFID（基于50K generated images）、Precision/Recall。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/WeizhiGao/MoDiff

  MoDiff算法pipeline（以DDIM on CIFAR-10，T=100 denoising steps，单层linear operator A^{(l)}为例）：

  **离线阶段（Calibration for Q-Diff + MoDiff）：**
  ```
  # 重构校准数据集以捕获时序差分信息
  for each calibration sample:
      # 完整运行扩散采样过程
      for t = T, T-1, ..., 1:
          # 标准扩散步骤获取每层的激活
          a_t^{(l)} = layer_l_input(x_t, t)    # 第l层在时间步t的输入激活

  # 对每层使用MoDiff重跑校准以学习量化参数
  for layer l in 1..L:
      # 在Q-Diff框架中，用MSE reconstruction学习scaling factor s
      # 输入变为时序差分而非原始激活
      loss = MSE(A^{(l)}(a_t^{(l)}), A^{(l)}(Q(a_t^{(l)} - â_{t+1}^{(l)})) + ô_{t+1}^{(l)})
      s^{(l)} = argmin loss  # 学习per-channel/per-tensor scale
  ```

  **在线推理阶段（逐denoising step, 逐layer）：**
  ```
  # Step 1: 第一个时间步T（warm-up，使用全精度激活避免初始量化误差）
  â_T = Q(a_T)                                       # Eq.(8): 量化激活
  ô_T = A^{(l)}(â_T)                                  # Eq.(9): 全精度计算
  # 可选：重复warm-up 4-5步收敛到全精度激活

  # Step 2: 后续时间步 t = T-1, T-2, ..., 1
  for t = T-1, T-2, ..., 1:
      # Error-compensated modulated quantization (Eq.13):
      â_t = Q(a_t - â_{t+1}) + â_{t+1}                # 量化时序差分 + 补偿前一量化激活
      # 等价于: â_t = (a_t - â_{t+1} - e_t') + â_{t+1}
      #          = a_t - e_t'
      # 其中 e_t' 是当前时间步的量化误差

      # Modulated computation (Eq.14):
      ô_t = A^{(l)}(Q(a_t - â_{t+1})) + ô_{t+1}       # 计算差分输出 + 累加上一时间步的输出
      # 等效于: ô_t = A^{(l)}(â_t)   (但实际通过差分计算)

      # 误差追踪分析 (Eq.18):
      e_t = (a_t - â_{t+1}) - Q(a_t - â_{t+1})       # 当前步量化误差
          = (a_t - â_{t+1}) - (â_t - â_{t+1})
          = a_t - â_t                                  # 该误差将在t-1步被补偿
  ```

  关键设计要点：
  - Bias Removal: 应用MoDiff的层必须去除bias项，因为Eq.(13)需要对算子做线性分解（A(a+b)=A(a)+A(b)要求A为纯线性算子，无bias）
  - Warm-up: 第一步使用全精度激活，避免初始量化误差被累积。经4-5次warmup后量化误差收敛到可忽略水平
  - Calibration Dataset Reconstruction: Q-Diff+MoDiff重新构造校准数据以捕捉时序差分而非原始激活
  - Layer-wise Reconstruction: 逐层独立重建（非整块重建），性能更稳定
  - 0-bit skipping (Remark 4.1): 当时序差分幅度低于可容忍阈值时，MoDiff允许分配0-bit表示跳过计算——此时等效于caching方法的特例

  定理保证：
  - Theorem 4.3: 量化误差 ∥x-Q(x)∥²₂ ≤ (max(x)-min(x))²d/(2^b-1)²，即误差正比于输入范围的平方。时序差分的范围比原始激活小10×+，因此相同位宽下量化误差降低100×+，或可用低3-4位达到相同误差界
  - Theorem 4.4: 标准调制误差以2^{T-k-1}c速率累积（指数增长），而error-compensated调制以(2c)^{T-k-1}速率递减（c<1/2时指数衰减），而非累积
