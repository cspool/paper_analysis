## GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

- baseline方法是什么？
  Baseline是现有将rotation与group quantization简单组合的量化方案，以及使用浮点dequantization的硬件加速器。

  全栈执行例子（以LLaMA-3-8B W4A4推理为例）：
  - 算法层：现有rotation-based quantization（Quarot/SpinQuant）执行全局Hadamard rotation flatten activation/weight分布→per-channel或per-token symmetric/asymmetric quantization→GPTQ weight quantization。或LightRot执行local rotation (R=G=128) + asymmetric group quantization (FP16 SF/ZP)。缺陷：全局rotation将outlier分散到所有channel，与fine-grained group quantization的localized scaling冲突——rotation期望全局平滑，group quantization依赖local distribution capture。实验显示当G≤32时，加rotation反而增加perplexity（RTN下从7.40升至30.12 at G=32）。
  - 系统框架层：GPU上使用TensorRT-LLM或vLLM部署量化模型，GEMM在Tensor Cores上以INT4执行→dequantization在CUDA cores上以FP16执行（INT→FP conversion + FP scale/bias + FP accumulate）→mixed-precision execution path增加latency和能耗
  - 编译框架层：论文未明确说明（手工CUDA kernel或RTL，无编译框架自动生成）
  - kernel调度层：MANT PE (G=64, FP16 SF, flexible data format) / LightRot PE (G=128, FP16 SF+FP16 ZP)。每个PE执行INT4 dot product后→FP dequantization unit：partial sum→FP16 SX乘法→FP16 ZX×WSUM加法→FP16 SW乘法→FP accumulate。缺陷：(a) G越小→dequantization频率越高，FP dequantization开销急剧增长（Fig.2）；(b) asymmetric quantization增加ZX项→FP overhead进一步放大；(c) GPU上INT→FP type conversion增加指令数和register pressure
  - 硬件架构层：Tender（8-bit systolic array，无group quantization，W4A4下accuracy severe degradation PPL=23.85 on LLaMA-1-7B）。MANT accelerator（2D systolic PE array + FP16 dequantization per G=64）。LightRot accelerator（2D systolic PE array + FP16 SF/ZP dequantization per G=128 + outlier-aware permutation）。缺陷：FP16 dequantization unit area/energy占比高；small group size下dequantization frequency成为瓶颈

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**GyRot**：通过CoRFiG解耦rotation scope与group granularity、HAP利用Hadamard harmonic rows做outlier alignment、重公式化非对称量化+ceiling ZP rounding实现fully integer dequantization，在algorithm-hardware co-design层面同时解决accuracy和hardware efficiency问题。

  论文方法全栈执行例子（以LLaMA-3-8B W4A4KV4, R=1024, G=32为例）：
  - 算法层（核心创新）：
    (1) **CoRFiG**（解决baseline缺陷：全局rotation破坏fine group的local coherence→accuracy degradation）：
    将rotation scope限制为R=1024（而非全局全部channel），同时保持group size G=32。满足R=2^g·G关系（g=5），确保rotation在R范围内充分flatten分布，但每个outlier最多影响R个channel而非全部→保留group-level local variance。实验：G=32时，CoRFiG (R=1024, PPL=6.91) vs 全局rotation (PPL=7.04)，证明localizing rotation scope改善与fine group的协同。
    (2) **HAP**（解决baseline缺陷：rotation后per-group range扩大→scale factor精度需求上升；INT8 SF时PPL degradation严重）：
    利用Hadamard矩阵递归构造产生的harmonic rows（长度2^k的全+1或全-1向量），将全局outlier channel permute到harmonic rows上。Post-rotation后每个group内outlier乘以同符号（全+1或全-1），per-group range被tightly bounded→scale factor的精度需求大幅降低。实验：CoRFiG+HAP使INT8 SF从PPL=364.17降至6.80（Table IV），证明INT8 SF可与FP16 SF parity。
    (3) **重公式化非对称量化 + Ceiling ZP rounding**（解决baseline缺陷：HAP后per-group分布高度asymmetric→传统非对称量化zero-point分布long-tailed→INT8 ZP degradation）：
    将公式从x̂=⌊x/s_x+z_x⌉改为x̂=⌊(x+z_x)/s_x⌉，z_x从scaled domain计算（−min(x_g)/s_x，小s_x放大尾部）改为unscaled domain计算（−min(x_g)，无除s_x操作）→zero-point分布显著flatten（Fig.5）。ZP量化用⌈·⌉替代round→保证z_Q≥z→消除underflow clipping（Fig.6）。实验：reformulated asymmetric + ceiling rounding使INT8 ZP从PPL=7.93恢复至6.91（Table V）。
  - 系统框架层：GyRot accelerator以RTL实现→综合为ASIC（28nm, 1GHz）→替代GPU部署量化LLM推理。无Serving框架修改（专用硬件方案）
  - 编译框架层：论文未明确说明（RTL手工实现，无编译框架自动生成）
  - kernel调度层（核心创新）：
    GyRot PE：32-way INT4 dot product→fully integer dequantization pipeline (INT8 SX→INT8 ZX×WSUM→INT8 SW)→32-bit integer accumulator。FHT unit：5-stage 32-way add/subtract pipeline (160 units)，支持online Hadamard rotation up to 1024-dim。
    对比baseline PE：MANT/LightRot PE需FP16 multiplier+adder做dequantization→GyRot PE以INT8乘法替代FP16乘法→PE area减65.2%、energy减69.2% vs Tender。
  - 硬件架构层（核心创新）：
    GyRot accelerator: 8×8×32 tensor PE array（3D tensor organization vs 2D systolic of baselines）→2048 parallel ops/cycle。FVU集成nonlinear ops + FHT rotation→消除CPU-GPU data movement for online rotation。WSUM unit预计算per-group weight sum→broadcast共享避免per-PE重复计算。Total: 2.10 mm², 740.95 mW in 28nm。
    Speedup: 1.42–3.40× over Tender/MANT/LightRot。Energy efficiency: 1.20–3.64× improvement。关键：算法创新（CoRFiG+HAP+reformulated asym quant）使INT8 SF/ZP成为可能→硬件创新（integer dequantization PE）利用这一宽松精度需求实现高效全整数数据路径。

  Baseline缺陷→GyRot方案映射：
  | Baseline缺陷 | GyRot方案 | 效果 |
  |-------------|---------|------|
  | 全局rotation破坏fine group local coherence→G=32时PPL从7.40反升至30.12 | CoRFiG: rotation scope限制为R=1024, R=2^g·G=32G, decouple rotation与group granularity | G=32+R1024 PPL=6.91 vs global rotation 7.04 |
  | Rotation后per-group range扩大→INT8 SF精度不足→PPL degraded to 364.17 | HAP: harmonic row alignment→per-group range tightly bounded→relax SF precision requirement | INT8 SF PPL从364.17→6.80 (parity with FP16) |
  | HAP后per-group高度asymmetric→传统非对称量化long-tailed ZP分布→INT8 ZP clipping | Reformulated asym quant (bias-before-scale) + ceiling ZP rounding | INT8 ZP PPL从7.93→6.91 (near FP16 6.81) |
  | FP16 dequantization unit area/energy overhead大→small G加剧频率 | Fully integer dequantization (INT8 SF/ZP datapath in PE) | PE area ↓65.2%, energy ↓69.2% vs Tender |
  | GPU mixed-precision path (INT GEMM→FP dequant→FP accum) | Dedicated accelerator with fused integer dequantization in PE | 1.42–3.40× speedup, 1.20–3.64× energy efficiency |
  | LightRot R=G耦合→无法scale到更小G（如32） | CoRFiG解耦R和G→R=1024固定, G可独立选择32 | G=32正常工作, LightRot在G=32 INT8 SF/ZP下PPL=30.12 |
