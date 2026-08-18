## Shining Light on Silicon Photonic DNN Accelerators

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现：在 DNN 推理精度评估中，通过 PyTorch hook 工具在模型前向中注入 SiPh 加速器的模拟非理想因素（调制器 E/O 非线性、ISI 导致的时序噪声、光路损耗导致的 AWGN 噪声），量化 3/4-bit 量化模型在 SiPh 加速器上的精度/困惑度损失。实验比较：(1) 非线性（MRM/MZM 调制器传递函数，不同偏置 ER 20/15/14 dB）对 ResNet50/MobileNetV2 在 ImageNet 上 3/4-bit 精度的影响；(2) ISI（不同 TX 驱动 -3dB 带宽 5/10 GHz、不同阵列尺寸 32×32~256×256）对精度的影响；(3) 噪声（不同 SNR/bit-precision、不同阵列尺寸）对 3/4/8-bit 模型精度的影响；(4) Qwen2.5-7B-instruct-AWQ 在不同激活量化精度（int8~int4）与粒度（per-tensor/per-feature/per-block）下的 Wikitext-2 困惑度。
- 硬件平台是什么，配置是什么。
  - 本层为精度评估（不跑真实硬件）：模型精度由 PyTorch 在通用服务器上评估。SiPh 加速器参数作为输入约束：时钟频率 5/10 GHz、MAC 精度 3/4-bit、波长数 8-128、阵列尺寸 32×32~256×256、TX 驱动 -3dB 带宽 5/10/15 GHz、调制器 ER 14/15/20 dB、噪声方差由光路损耗预算推导（Pmin = 2×SNR×In,rms/R，线性 TIA 输入噪声 1.3 μA）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：ResNet50（26M 参数）与 MobileNetV2（3.4M 参数），采用量化感知训练（QAT [81]）到 3/4-bit（另有 8-bit 对照）；Qwen2.5-7B-instruct 语言模型（fp16 基线 → AWQ int4 权重 [82] → 激活量化到 int4-int8）。数据集：ImageNet（图像分类 top-1 精度）、Wikitext-2（困惑度）。bench：ImageNet top-1 分类精度、Wikitext-2 perplexity（baseline 6.79）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确说明是否开源其评估代码。所用组件开源：PyTorch（https://github.com/pytorch/pytorch，hook 工具 [77]）、AWQ（https://github.com/mit-han-lab/llm-awq，[82]）、Qwen2.5-7B-Instruct（https://huggingface.co/Qwen/Qwen2.5-7B-Instruct，[80]）、ImageNet/Wikitext-2 为标准公开数据集。
  - 算法pipeline 伪代码（每 DNN 层，噪声注入方式见图 8）：
    ```
    # 输入：量化模型（图像模型 QAT 到 3/4-bit；LLM 为 AWQ int4 权重 + 激活量化）
    for layer in model:
        # 1) activation/weight/output 量化器保证低比特（LLM 激活按 per-tensor /
        #    per-feature(每 hidden dim) / per-block(14 batch × 74 hidden dim) 的
        #    affine scale+zero-point 量化 [85]；QAT 同时学习最优动态范围）
        x_q = Q_act(x); w_q = Q_wt(w)
        # 2) forward pre-hook：注入 E/O 调制器静态非线性（取决于调制器与偏置）
        #    MRM: P_out = P_in(1 - K_ring/(1 + βV²))；MZM: P_out = P_in/2(1 + cos(πV/Vπ))
        y = MAC(x_q, w_q)              # 光域加权和（点积）
        # 3) forward post-hook：注入 ISI 推导的 post-MAC 条件输出分布（高斯拟合，
        #    来自 Cadence Spectre 瞬态仿真眼图，随 TX 驱动带宽/阵列尺寸变化）
        y += N(0, σ²_ISI)
        # 4) forward post-hook：注入 AWGN（σ² 由光损耗预算推导，Pmin=2·SNR·In,rms/R；
        #    按点积长度/通道数添加多个噪声样本 [86]）
        y += Σ_{i=1..n_samples} N(0, σ²_opt)
        # 5) 输出量化器
        y_out = Q_out(y)
    # 评估：ImageNet top-1 精度 / Wikitext-2 perplexity
    ```
  - 张量计算示例（一个卷积/线性层，dot-product 长度 d、输出通道 C）：激活张量 (N,C,H,W) 经 Q_act 量化到 4-bit（scale=range/15 均匀量化，QAT 学习的最优动态范围）→ 与 int4 权重矩阵做整数 MAC → 输出累加值按 SiPh 阵列参数叠加 n_samples≈C 个独立 AWGN 样本（模拟每通道经不同 λ/PD 的噪声累加）→ 输出量化器 Q_out 回到 4-bit → 统计 top-1 精度。关键结论：任何单一非理想因素未补偿都会使 ResNet50 4-bit ImageNet 精度相对理想 4-bit 下降 >10%；3-bit 下 MZM 余弦非线性使 ResNet50 掉 >25%、MobileNetV2 崩溃到近零；Qwen2.5-7B 激活量化到 int5/int4 时困惑度从 6.79（fp16）剧增（per-block 粒度最优仍达 182/2129，int4 时 per-tensor 达 120 万），指示 LLM 部署 SiPh 需进一步算法/器件级改进。
