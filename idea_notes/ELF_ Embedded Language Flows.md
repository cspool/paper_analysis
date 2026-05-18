## ELF: Embedded Language Flows

- baseline方法是什么？
  Baseline是现有扩散语言模型（DLMs），分为两类：(1) Discrete DLMs（MDLM、Duo等）直接在离散token space定义扩散过程，使用masked/uniform transition matrices，需要categorical reparameterization，CFG难以有效应用；(2) Continuous DLMs（Diffusion-LM、CDCD、FLM、LangFlow等）将token映射为continuous representation但每步均施加token-level cross-entropy supervision（per-step discretization），导致denoising trajectory被耦合到vocabulary-level prediction，限制了flow dynamics的灵活性。Latent Diffusion LMs（LD4LG等）虽然避免per-step vocabulary supervision，但依赖DDPM-style formulation + 单独训练的decoder。

  全栈执行例子（以MDLM/FLM + OWT unconditional generation为例）：
  - 算法层：MDLM使用absorbing-state masking in discrete token space，128/256/1024 steps of iterative unmasking，无法使用CFG。FLM在embedding space做Flow Matching但每步施加cross-entropy loss，将embedding projection回token space做监督，flow trajectory受token-level constraint限制。
  - 系统框架/Serving层：论文未明确说明（所有方法均为research-stage DLM training，未涉及serving deployment）。
  - 编译框架层：论文未明确说明（PyTorch/JAX standard training loop，无定制compilation）。
  - kernel调度层：论文未明确说明（TPU/GPU standard matmul and attention kernels）。
  - 硬件架构层：Google TPU v5p（训练），NVIDIA GPU（baseline推断），无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出ELF，通过以下关键设计解决baseline缺陷：

  **缺陷1：Discrete DLMs在离散token space定义扩散，需categorical transitions，难以apply CFG**
  → ELF将language generation完整体迁移到连续空间：tokens→T5 encoder→continuous embeddings→Flow Matching denoising全程在continuous embedding space→仅在t=1 final step做discretization。这使得CFG（原为continuous quantities设计）可自然应用。ELFB with CFG scale=3 in 32步达到Gen.PPL 24 vs MDLM 1024步 27、Duo 1024步 34。

  **缺陷2：现有continuous DLMs每步施加per-step token-level cross-entropy supervision，将denoising trajectory耦合到vocabulary prediction**
  → ELF在除最后一步外的所有steps使用pure MSE loss in embedding space（L_MSE = ‖(x̂−x)/(1−t)‖²），不做per-step discretization。仅在t=1 final step使用CE loss via shared-weight unembedding。Denoising mode概率0.8 vs decoding mode 0.2，确保主trajectory在continuous space自由演化。

  **缺陷3：Latent Diffusion LMs需要单独训练decoder（autoregressive/non-autoregressive），增加inference component和训练复杂度**
  → ELF使用shared-weight network：同一网络在所有t<1执行denoising（x-prediction+MSE loss），在t=1执行decoding（x-prediction+unembedding+CE loss）。通过binary "mode" token区分。无需单独decoder，减少参数量和训练stage。

  **缺陷4：DDPM-style discrete-time formulation限制sampling flexibility**
  → ELF使用continuous-time Flow Matching（rectified flow），支持ODE和SDE sampler灵活切换。Logit-normal time schedule在few-step regime显著优于uniform schedule。SDE sampler在32步即可达到1024步ODE的质量（Gen.PPL 24 vs 26.6），data efficiency提升10×（45B vs 524B tokens）。

  **缺陷5：x-prediction在high-dim embedding space中比v-prediction/ϵ-prediction更稳定但未被充分exploit for language**
  → ELF证明x-prediction在512/768/1024-dim embedding space中唯一保持稳定（v-prediction在高维退化，ϵ-prediction collapse）。x-prediction使shared-weight design成为可能（denoising和decoding均predict clean embeddings）。

  论文方法全栈执行例子（以ELF-B + OWT + 32-step SDE + TPU v5p×64为例）：
  - 算法层：T5-small encoder frozen→bottleneck 512→128→hidden 768→DiT with SwiGLU/RMSNorm/RoPE/qk-norm→Flow Matching x-prediction (80% MSE) + shared-weight decoding (20% CE)→training-time CFG (ω∈[0.5,5]) with self-conditioning→SDE sampler with γ=1.5, CFG scale=3→32步Gen.PPL 24.08, Entropy 5.15。
  - 系统框架/Serving层：论文未明确说明（research-stage，无serving deployment）。推断时支持batch generation。
  - 编译框架层：论文未明确说明（标准PyTorch/JAX training loop on TPU）。
  - kernel调度层：论文未明确说明（TPU v5p standard matrix multiply and attention kernels）。
  - 硬件架构层：Google TPU v5p × 64，1.5h/epoch，5 epochs total，45.2B effective training tokens。无定制硬件。
