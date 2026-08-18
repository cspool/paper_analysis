## AQuant: Repurposing CODEC for VLM Acceleration via Adaptive Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 AQuant 动态视觉 token 量化算法（三阶段）：(1) Exponent-Similarity Detection——从 N 个视觉 token 中按间隔 F=⌊N/M⌋ 选每 F 个连续 token 的中间 token 为 M 个候选 base token，用指数近似 L1 距离 D̄(i,j)=Σ_k |sign(T_k^i)·2^Ē_k^i − sign(T_k^j)·2^Ē_k^j|（只取 sign+exponent 位，整数移位/减法）选出每 token 最近的 base；(2) Adaptive Quantization——delta δ^i=T^i−B^i 内按幅值 top p（默认 25%）元素量化到 INT4、其余 75% 量化到 INT2（bitmask 标记，构成互补稀疏矩阵），base token 保持 INT8；(3) Result Reconstruction——W_q·T = W_q·B + W_q·δ，M 个候选 base 的 W_q·B 预计算、运行时直接查表选择，W_q·δ 低精度 GEMM，二者相加重构 Q。解码阶段把 KV projection 后的 KV-cache 重新量化后存 off-chip、使用时在线重构以省内存带宽。
  - 实验比较：baseline 为 PyTorch 全精度 VLM（GPU-Full，按面积预算核数缩放的 Jetson AGX Xavier 模型）；对比 GPU-Full-unscale（Xavier 实测）、GPU-Mixed-precision（理想混合精度解析上界，speedup=8/(4×0.25+2×0.75)=3.2×）、GPU-AQuant（算法跑在 GPU 上）、GPU-VisPruner（VisPruner 剪 token）、OliVe、LLM.265、CMC、AQuant-Pruning（去 INT2 剪枝消融）。指标：准确率、加速比、能耗效率、prefilling/decoding 分阶段延迟。
- 硬件平台是什么，配置是什么。
  - 算法评估：Python 实现 + PyTorch 框架（开源 VLM 实现），无重训练，校准数据取训练集 10%；GPU 基线实测平台 NVIDIA Jetson AGX Xavier。
- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaVA、VideoLLaVA、Qwen2.5-VL 72B，共 3 个 VLM。数据集/bench 共 14 个：VQAv2、GQA、TextVQA、POPE、MM-Bench、MMVet、Wild、ScienceQA、VisWiZ、ActivityNet、MSVD、TGIF、MSRVTT、Video-MME。超参数：F=18（对应 7.4% INT8 base token）、p=25%。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：AQuant 算法与架构代码论文未明确说明是否开源；模型采用开源实现（LLaVA/VideoLLaVA/Qwen2.5-VL）。基线 CMC、LLM.265、OliVe 由作者复现算法并自建 cycle-accurate 模拟器（CMC: ASPLOS'24；LLM.265: MICRO'25；OliVe: ISCA'23）。
  - 算法 pipeline 执行例子（prefilling，Q=W_q·T 为例，T∈R^{N×K}）：① 设 N 个视觉 token、F=18 → M=N/18 个候选 base token，取每 18 个连续 token 的中间 token；② 指数相似检测：提取每 token 每元素的 sign+exponent（视觉 token 指数集中 [0,8]，2^exponent∈[1,256] 用 9 bit + 1 bit sign = INT10），对每个 token i 计算 D̄(i,j)=Σ_k |sgn(T_k^i)·2^Ē_k^i − sgn(T_k^j)·2^Ē_k^j|（j=1..M），取 argmin 得 B^i；③ 求 delta δ^i=T^i−B^i（窄分布）；④ 每行 δ^i 内 top 25% 大值 → INT4（bitmask 置 1）、其余 → INT2，base token B 保留 INT8；⑤ GEMM 分解：W_q·T ≈ W_q·B + W_q·δ，W_q·B 只对 M 个候选 base 预计算并查表（每 token 直接取对应候选的输出）；W_q·δ 以 INT2/INT4 混合精度乘 INT16 权重；⑥ 逐 token 相加重构 Q 矩阵，再送 Softmax/GELU/LayerNorm 等 FP 非线性算子。指数越界（[0,8] 之外）token 作为 outlier 跳过相似检测直接高精度处理。
  - 效果：平均准确率损失 0.7%（AQuant-Pruning 消融损失 23%）；理论计算相对全 INT8 减少 3.2×；fast-motion 视频仅 0.83% 损失；F 在 12–24 扫描后固定 18（F=24 精度骤降），p 在 4 个 benchmark 扫描后固定 25%（p=20% 时 MSVD 掉 2.1% 不可接受）。
