## VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

- 属于算法pipeline的实现是什么？实验比较什么？
  提出VAR-Turbo软件-硬件协同加速框架，包含三项算法优化：(1) Draft-Free Parallel Decoding (PD)：利用图像token的空间冗余（entropy分析表明图像token冗余远高于语言token），每轮同时预测所有masked token，按置信度TopK选择多个token并行解码，无需draft model，将采样步数降低>80%（256×256下从256步降至8-32步）；(2) Token Aggregation (TA)：在浅层Learning Region中，将token sequence划分为non-overlapped local window，先用Small Attention将窗口内token聚合为representative token，再经Big Attention做全局建模，减少约41% attention MAC且质量下降<0.5%；(3) Dynamic Bypass (DB)：在深层Inert Region中，用轻量MLP为token打importance score，仅TopK重要token进入完整Transformer（attention+FFN），其余bypass并通过token restoration避免信息丢失，额外减少约58% MAC。实验比较生成质量（IS/FID）和计算量（TFLOPs），对比Vanilla VAR、MaskGIT、ViTCoD、AdapTiV。VAR-Turbo-Peak在256×256下20 steps、2.8 TFLOPs、FID 2.65、IS 272.4；VAR-Turbo-Balance为8 steps、1.1 TFLOPs、FID 2.67、IS 268.6；512×512下Balance为32 steps、5.7 TFLOPs、FID 3.15、IS 259.6。

- 硬件平台是什么，配置是什么。
  算法训练：4×NVIDIA V100 GPU（ImageNet, 500 epochs, batch size 256, 816 GPU hours, AdamW: lr=1e-4, weight decay=1e-5, momentum (0.9, 0.96)）。通用平台baseline：Intel Xeon Platinum 8168 CPU @2.70GHz、NVIDIA V100 GPU。GPU延迟用torch.cuda.event测量，CPU延迟用time.time；功耗分别用pynvml和s-tui获取。硬件加速器：TSMC 28nm+HPC 1P8M CMOS, TT 25C, 7.09 mm², 1.98 W + 2×64bit HBM2 @2GHz 32GB/s。

- 模型是什么。数据集和bench分别是什么。
  模型：基于DeiT的generative Transformer + VQGAN tokenizer。数据集：ImageNet（主训练集和benchmark）。泛化实验：MS-COCO、CC3M、Places2以及ViT/DeiT/LeViT/SwinT-VAR backbone。评估指标：IS (Inception Score) 和 FID (Frechet Inception Distance)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：论文未提供明确开源仓库（HPCA 2026, DOI: 10.1109/HPCA68181.2026.11408607）。VAR-Turbo算法pipeline（以256×256图像生成8步Balance模式为例）：
  1. Tokenization：VQGAN将图像编码为256个visual token（16×16 grid）
  2. 初始化：文本条件与全masked visual token拼接为V0，mask array全True
  3. 每轮迭代：generative Transformer对所有当前位置输出logits→PD使用Gumbel sampling得pred token和概率→已unmasked token置信度设为负无穷→根据schedule r(t)计算K=N*(1-r(t))个释放位置→Radix Sort Core对confidence array执行TopK→TopK token替换为pred token并mask=False
  4. 每轮Transformer invocation内：浅层0-15层(Learning Region)执行TA——token分local window（低分辨率size=2，高分辨率size=2/4混合）→Small Attention聚合representative token→Big Attention全局建模；深层(Inert Region)执行DB——轻量MLP打分→TopK重要token进入attention+FFN→bypass token通过Token_i × JudgeWeight_i + Token_i恢复信息回下一层
  5. 所有visual token解码完成→VQGAN decoder还原像素图像
  6. PD、TA、DB协同：跨迭代减少Transformer调用次数（PD），每次调用内减少attention MAC（TA）和FFN MAC（DB）。关键trade-off：PD需PD-aware training选择sampling temperature/masking ratio/guidance scale；TA的local window size≥8时质量明显下降；DB需schedule function控制逐层skip rate（α=0.3, β=-0.4, max skip threshold=0.55）
