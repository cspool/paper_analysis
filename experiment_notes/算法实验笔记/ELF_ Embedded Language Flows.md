## ELF: Embedded Language Flows

- 属于算法pipeline的实现是什么？实验比较什么？
  提出ELF（Embedded Language Flows），一种在连续embedding空间中使用连续时间Flow Matching的扩散语言模型（DLM）。核心算法设计：(1) Continuous embedding space：使用frozen pretrained T5-small encoder将离散token映射为连续contextual embeddings（512-dim），通过bottleneck线性投影至128-dim后送入model hidden size 768。在连续embedding space中直接执行denoising，仅在最后一步（t=1）通过共享权重的unembedding层映射回离散token；(2) Flow Matching with x-prediction：使用continuous-time Flow Matching（rectified flow），linear interpolant z_t = t·x + (1-t)·ϵ，网络直接预测clean embeddings x（x-prediction），训练目标为MSE loss L_MSE = ‖(x_θ(z_t,t) − x)/(1−t)‖²；(3) Shared-weight denoiser-decoder：单一网络同时作为denoiser（80%概率，MSE loss）和decoder（20%概率，cross-entropy loss），decode branch使用per-token corruption（logit-normal noise schedule，P_mean=0.8, P_std=0.8）模拟不完美denoiser输出，训练共享权重的unembedding层进行最终离散化。无需单独训练decoder；(4) Training-time CFG with self-conditioning：使用self-conditioning（50%概率concatenate前一预测x̂'作为条件）作为CFG的条件信号，训练时即融入CFG formulation：v_target = v + (1−1/ω)(v_sc − v_no_sc)，CFG scale ω∈[0.5,5]。使用in-context conditioning（4 time tokens + 4 CFG-scale tokens + 4 mode tokens prepended to sequence）替代adaLN-Zero，减少参数量（ELF-B from 148M→105M）；(5) SDE-inspired sampler：在ODE基础上每个step注入Gaussian noise（z_back = α·z + (1-α)·ε, α=1-γ·dt），然后对perturbed state调用denoiser，使用clean prediction更新原state。γ=0退化为ODE，γ>0为SDE。实验比较unconditional generation (OWT) 和conditional generation (WMT14 De-En translation, XSum summarization)，对比MDLM、Duo (discrete DLMs)、FLM/FMLM、LangFlow (continuous DLMs)、SeqDiffuSeq、CDCD、E2D2、AR baseline。

- 硬件平台是什么，配置是什么。
  Google TPU v5p × 64（训练）。训练时间：OWT上ELF-B每epoch约1.5小时。推断使用ODE/SDE sampler，支持32/64/128/256/512/1024 sampling steps。

- 模型是什么。数据集和bench分别是什么。
  模型：ELF-B (105M, 12层, hidden 768, 12 heads)、ELF-M (342M, 24层, hidden 1056, 16 heads)、ELF-L (652M, 32层, hidden 1280, 16 heads)。基于Diffusion Transformer (DiT)架构+ SwiGLU + RMSNorm + RoPE + qk-norm。使用frozen pretrained T5-small encoder (35M)作为embedding encoder。Muon optimizer (lr=0.002, batch size=512)。数据集：OpenWebText (OWT, ~9B tokens, 序列长度L=1024, 5 epochs ≈ 45.2B effective tokens)、WMT14 German-English (De-En, L=128, 144M target tokens)、XSum (L=1088, 6M target tokens)。评价指标：Generative Perplexity (Gen.PPL, 用GPT-2 Large评估)、unigram entropy、BLEU、ROUGE-1/2/L。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/lillian039/ELF。算法pipeline（以OWT unconditional generation + ELF-B + 32-step SDE sampling为例）：
  1. 训练阶段：
     a. Encode：使用frozen T5-small encoder将离散token序列s (L=1024)映射为continuous embeddings x (512-dim)→通过bottleneck线性投影至128-dim→再投影至hidden size 768。
     b. Denoising branch (80%概率)：采样t~logit-normal(P_mean=-1.5, P_std=0.8)→z_t = t·x + (1-t)·ϵ（ϵ~N(0,I), noise scale=2）→self-conditioning（50%概率concatenate前次预测x̂'）→prepend control tokens (time, CFG scale, mode)→送入DiT网络预测x̂→计算MSE loss L_MSE = ‖(x̂−x)/(1−t)‖²（实际为v-prediction转换后的velocity loss）。
     c. Decoding branch (20%概率)：t=1, per-token corruption p~logit-normal(0.8,0.8)→z̃ = p·x + (1−p)·ϵ（noise scale=5）→送入同一网络（mode="decode"）→unembedding层W映射为vocabulary logits→cross-entropy loss L_CE。
     d. Training-time CFG：同时计算conditional和unconditional prediction→v_target = v + (1−1/w)(v_sc − v_no_sc)→以CFG-combined target训练单一网络。
  2. 推断阶段（32-step SDE with CFG scale=3, γ=1.5）：
     a. z_0 ~ N(0,I)→对每个time step (logit-normal schedule, T=32 intervals)：self-condition on previous x_pred→调用denoiser得x̂→v = (x̂−z)/(1−t)→SDE step（注入噪声：α=1-γ·dt, z_back=α·z+(1-α)·ε, 在z_back上重新预测x̂，用原z更新z=z+dt·v）。
     b. 最后一步（t=1）：调用decode mode→unembedding→argmax得离散token→输出文本。
  3. ELF-B在32步SDE下Gen.PPL=24.08（vs MDLM 1024步Gen.PPL≈27, Duo 1024步≈34），使用仅45B training tokens（vs baselines 524-577B），10× fewer training tokens。

