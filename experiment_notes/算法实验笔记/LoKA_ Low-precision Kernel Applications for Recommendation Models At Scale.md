## LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：LoKA 是一个让 FP8 低精度训练/推理对大型推荐模型（LRM）可用的系统-模型协同设计框架，核心是三个算法/模型级组件（全部以 PyTorch + 三个低精度库 TorchAO/DeepGEMM/FBGEMM 实现）：
    1. **LoKA Probe（分布感知的统计建模）**：在线学习每层激活与权重的统计分布（激活建模为多元高斯，利用 batch 维独立把存储从 O(M²N²) 降到 O(N²)，用批量 Welford tracker 流式更新均值/协方差；权重建模为矩阵正态 MN(M,U,V)，用 Kronecker-factor flip-flop EMA 更新，存储 O(M²+N²)，尺度用 trace 重归一化），再离线从学到的分布采样合成输入/权重做误差与吞吐评估，量化每层 MERE（Mean Element-wise Relative Error，相对 TF32 参考）。每 100 迭代激活、每 10000 迭代异步保存统计参数，开销 ≤1%。
    2. **LoKA Mods（模型组件重设计）**：No Bias（除最终预测层外移除所有 bias，借鉴 DeepSeek/PaLM/Falcon，同时减少 FSDP per-parameter padding 通信开销）；BlockNorm（把归一化改为沿特征维固定块（如 256 元素）的 RMS 归一化 `RMSNorm((Wx+b).view(-1,BlockN)).view(B,N)`，等价于无参数 Grouped RMSNorm，可融合进 GEMM epilogue、避免全局同步与 mean-cancellation 误差）；Hard Swish（`h-swish(x)=x·ReLU6(x+3)/6`，替换 sigmoid 型 Swish，消除指数运算、天然适合低精度且可与 BlockNorm 融合进同一 kernel）。
    3. **LoKA Dispatch**（见 kernel 调度层条目）。
  - 实验比较什么：(1) 无质量损失的 FP8 全轨迹训练——Wukong/Interformer/ELFM 相对原始高精度 baseline 的相对 log loss 全程持平（对比直接 TorchAO FP8 训练的 1.3× 变慢 + 2.5% relative log loss 退化）；(2) 分布感知误差——用学习分布 vs 标准正态输入的 MERE 几何均值增高达 15%，并用 LoKA Probe 发现 FBGEMM 生产 benchmark 的 faulty test code（正常/错误实现的 MERE 相差 47×，随机输入下几乎相同）；(3) LoKA Mods 消融——No Bias 单独贡献最大延迟降低，BlockNorm 提供数值调理与融合可能，Hard Swish 较小，三者合并延迟降低 >2× 且实现全轨迹 loss 中性；(4) 端到端加速——训练最高 1.19×、推理 1.4×（相对已用 LoKA Mods 重写的强 baseline），生产部署 5–20% 训练吞吐 / 10–17% 推理加速。
- 硬件平台是什么，配置是什么。
  - NVIDIA H100、B200、GB200（NVL72）与 AMD MI300X、MI350X 集群，16–256 张 GPU（Wukong H100 6K batch&32 GPU、B200/MI300X 12K&16、GB200/MI350X 20K&16；Interformer 4K&64/8K&64/20K&32；ELFM 2K&256/6K&128/20K&32）。基准分析实验用 64× H100。LoKA 在 H100/B200/MI300X 上开发验证，后在开发期间不可得的 GB200 NVL72 与 MI350X 上无修改直接评估，得到相当加速比。
- 模型是什么。数据集和bench分别是什么。
  - 模型：三个 SOTA LRM 家族——Wukong（257B 参数、24 GFLOPs/sample，含 FMB/LCB 开源组件 [5]）、Interformer（566B、28，Transformer 交互组件）、External Large FM / ELFM（1343B、40，整合 DHEN/DLRM/SUM/DCN 的复合架构）。数据集：生产级行业规模数据集（数百亿样本、数千特征），非公开；bench 即三家公司的生产广告排序/推荐 ranking 任务与 27 个生产 GEMM shape。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：LoKA 本身未开源（截至 2026-08 联网搜索未找到公开仓库，arXiv:2605.10886 未附代码链接）。依赖均开源：PyTorch（pytorch.org）、TorchRec（github.com/pytorch/torchrec，混合并行/嵌入分片）、TorchAO（github.com/pytorch/ao）、DeepGEMM（github.com/deepseek-ai/DeepGEMM）、FBGEMM（github.com/pytorch/FBGEMM）、torch.compile；模型参考 Wukong（arXiv:2403.02545）、Interformer（arXiv:2411.09852）、ELFM（arXiv:2502.17494）。
  - 算法 pipeline 例子（以 Wukong 一个线性层 FP8 训练为例）：
    ① **Probe 在线分布学习**：该层激活 X∈R^{B×K}（batch 独立，只建模特征维）：流式合并统计量 n_new=n_old+B、δ=μ_b−μ_old、μ_new=μ_old+(B/n_new)δ、Σ_new=Σ_old+S_b+(n_old·B/n_new)δδᵀ（S_b 为 batch scatter），样本协方差 Σ=Σ_new/(n_new−1)，FP32 累积；权重 W∈R^{M×N} 建模矩阵正态：每 minibatch 解 L_VL_Vᵀ=V+εI 得 U'=(1/N)(W_c L_V⁻ᵀ)(W_c L_V⁻ᵀ)ᵀ，解 L_UL_Uᵀ=U+εI 得 V'=(1/M)(L_U⁻¹W_c)ᵀ(L_U⁻¹W_c)，EMA 平滑 U''=mU+(1−m)U' 并对称化+正则化，trace 重归一化 s=trace(U)/M、U←U/s、V←sV。
    ② **离线采样与误差量化**：激活 T'=1_Bμᵀ+ZL_Σᵀ（Z~N(0,I_K)，L_Σ 为 Σ+εI 的 Cholesky 因子）；权重 W'=M+L_UZL_Vᵀ。对每层采样 100 对输入-权重，跑 FP8 kernel vs TF32 参考，按 MERE=Σ_mΣ_n|out−ref|/|ref| 量化误差，MERE 高或加速比低的层标记为低精度不安全。
    ③ **LoKA Mods 重写模型**：移除 bias → `out = RMSNorm(xW.view(B, BlockN)).view(B, N)`（BlockN=256，按块独立算 RMS、块内融合激活与(反)量化，训练/推理同块大小保证一致性）→ `out = out * ReLU6(out+3)/6`（Hard Swish）。
    ④ 张量计算实例：输入 x∈R^{2048×256}、权重 W∈R^{256×768} 的 GEMM → FP8 量化（按选定 recipe：tensorwise/rowwise/blockwise scaling，scales 取行/块内最大绝对值）→ FP8 张量核乘加（快速累加 FP32）→ dequantize + BlockNorm(256) + Hard Swish 融合在 epilogue 完成 → 输出，与 TF32 基线比较 MERE 判定是否安全低精度。
