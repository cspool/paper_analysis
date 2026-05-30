# <span id="page-13-0"></span>D GENERAL RELATED WORK

#### D.1 DIFFUSION-BASED LONG VIDEO GENERATION

Recent advances in diffusion models [\(Villegas et al.,](#page-11-9) [2023;](#page-11-9) [He et al.,](#page-9-9) [2022;](#page-9-9) [Chen et al.,](#page-9-10) [2024b;](#page-9-10) [Wang](#page-11-10) [et al.,](#page-11-10) [2025;](#page-11-10) [Guo et al.,](#page-9-11) [2025;](#page-9-11) [Dalal et al.,](#page-9-12) [2025\)](#page-9-12) have explored long video generation. Phenaki [\(Vil](#page-11-9)[legas et al.,](#page-11-9) [2023\)](#page-11-9) compresses video into discrete tokens, enabling variable-length generation from open-domain text. NUWA-XL [\(Yin et al.,](#page-11-11) [2023\)](#page-11-11) extends diffusion to extremely long sequences via a coarse-to-fine "diffusion over diffusion" framework, generating global keyframes and filling intermediate frames in parallel. LVDM [\(He et al.,](#page-9-9) [2022\)](#page-9-9) leverages a compact 3D latent space with hierarchical generation. LaVie [\(Wang et al.,](#page-11-10) [2025\)](#page-11-10) proposes a cascaded pipeline, with joint finetuning, rotary position encoding, and temporal attention. SEINE [\(Chen et al.,](#page-9-10) [2024b\)](#page-9-10) employs smooth shot transitions using a stochastic masking-based diffusion model. LCT [\(Guo et al.,](#page-9-11) [2025\)](#page-9-11) expanded pre-trained short-video models to scene-level contexts for multi-shot coherence, via largescale fine-tuning. Other approaches [\(Dalal et al.,](#page-9-12) [2025\)](#page-9-12) use a test-time training technique to generate minute-long videos. Although these models can generate long-duration videos, they often incur heavy computational costs, motivating more efficient and real-time solutions.

Several recent works extend the generation length of diffusion models in a training-free manner. RI-FLEx [\(Zhao et al.,](#page-12-2) [2025\)](#page-12-2) conducts video length extrapolation by adjusting the intrinsic frequency of position embeddings, mitigating temporal repetition and motion slowdown. FreeNoise [\(Qiu et al.,](#page-10-12) [2024\)](#page-10-12) uses a noise rescheduling strategy and window-based temporal attention. FreeLong [\(Lu et al.,](#page-10-13) [2024\)](#page-10-13) blends temporal frequency components at inference. FreeLong++ [\(Lu & Yang,](#page-10-14) [2025\)](#page-10-14) introduces multi-band spectral fusion to capture and fuse multi-frequency temporal information. In these training-free settings, models achieve at most a 4–8× extension in length (up to 40 seconds), which remains inadequate for long-form scenarios.

#### D.2 AUTOREGRESSIVE LONG VIDEO GENERATION

A growing number of works [\(Chen et al.,](#page-9-0) [2024a;](#page-9-0) [Song et al.,](#page-10-3) [2025;](#page-10-3) [Mao et al.,](#page-10-4) [2025;](#page-10-4) [Yuan et al.,](#page-12-1) [2025;](#page-12-1) [Zhang & Agrawala,](#page-12-0) [2025;](#page-12-0) [Gao et al.,](#page-9-3) [2025;](#page-9-3) [Henschel et al.,](#page-9-4) [2025;](#page-9-4) [Gao et al.,](#page-9-3) [2025\)](#page-9-3) integrate

#### <span id="page-14-0"></span>**Algorithm 1** Streaming Long Tuning **Algorithm 2** Interactive Inference **Require:** Causal video generator $G_{\theta}$ **Require:** Causal video generator $G_{\theta}$ , Prompt set $\mathcal{P}$ **Require:** Prompt sequence $\mathcal{P} = [p_0, \dots, p_n],$ **Require:** Video length $l_{\text{video}}$ , Per clip length $l_{clip}$ switch-index sequence $S = [s_1, \dots, s_n]$ 1: **while** not converged **do Require:** Number of video frames N, diffusion 2: Initialize KV cache $C \leftarrow []$ steps per frame T3: Initialize current video length $l \leftarrow 0$ 1: Initialize model output $\mathbf{x} \leftarrow []$ 4: Sample $(p, p_{\text{next}}) \sim \mathcal{P}$ 2: Initialize KV cache $C \leftarrow []$ 5: Sample switch index s3: $p_{\text{active}} \leftarrow \mathcal{P}.pop(0)$ $s \in \{1, 2, \dots, \lfloor l_{\text{video}}/l_{\text{clip}} \rfloor - 1\}$ 4: **for** i = 1, ..., N **do** 6: $s \leftarrow s \cdot l_{\text{clip}}$ 5: if $i \in \mathcal{S}$ then if $l \geq l_{\mathrm{video}}$ then 7: 6: $p_{\text{active}} \leftarrow \mathcal{P}.\texttt{pop(0)}$ $C \leftarrow []; \quad l \leftarrow 0$ 7: $C \leftarrow \text{recache}(G_{\theta}, \mathbf{x}, C, p_{\text{active}})$ 8: 9: Resample $(p, p_{next})$ and s8: 9: Initialize $x_{t_T}^i \sim \mathcal{N}(0, I)$ 10: 10: for $j = T, \dots, 1$ do $p_{\text{active}} \leftarrow \begin{cases} p, & \text{if } l < s \\ p_{\text{next}}, & \text{otherwise} \end{cases}$ 11: Set $\hat{x}_0^i \leftarrow G_\theta(x_{t_j}^i; t_j, C, p_{\text{active}})$ 11: 12: if j = 1 then 12: 13: $\mathbf{x}.append(\hat{x}_0^i)$ $C \leftarrow G_{\theta}^{KV}(x_j^i, 0, C, p_{\text{active}})$ $C \leftarrow \text{recache}(G_{\theta}, \mathbf{v}, C, p_{\text{active}})$ 13: 14: 14: end if 15: $\mathbf{x} \leftarrow \text{generate\_next\_clip}(G_{\theta}, C, p_{\text{active}})$ 16: Sample $\epsilon \sim \mathcal{N}(0, I)$ 15: $\mathcal{L} \leftarrow \text{DMD\_Loss}(G_{\theta}, \mathbf{x}, p_{\text{active}})$ 16: Set $x_{t_{j-1}}^i \leftarrow \Psi(\hat{x}_0^i, \epsilon, t_{j-1})$ 17: $\mathcal{L}$ .backward() 18: update generator parameter $\theta$ 19: 18: 19: $l \leftarrow l + l_{clip}$ 20: **end for** 21: return x 20: end while

diffusion modeling with AR prediction, an intermediate paradigm between purely diffusion-based approaches and purely AR approaches. Diffusion-forcing (Chen et al., 2024a) formalizes this hybrid paradigm by injecting noise into future tokens and training the model to denoise them, combining diffusion quality with AR efficiency. Streaming T2V (Henschel et al., 2025) extends this idea with short and long-term memory modules for coherent text-to-video generation. Pyramidal-flow (Jin et al., 2025) proposes a multi-scale flow matching design to reduce computation. History-guided video diffusion (Song et al., 2025) further incorporates flexible-length historical context to improve temporal consistency over extended rollouts. SkyReels-V2 (Chen et al., 2025a) couples diffusion forcing with a film-structure planner and multimodal controls. FramePack (Zhang & Agrawala, 2025) compresses input frames into a fixed-size context to address memory and efficiency bottlenecks. Lumos-1 (Yuan et al., 2025) employs large language models (LLMs) style architectures, integrating spatiotemporal modeling under the diffusion-forcing framework. Most recently, LongVie (Gao et al., 2025) introduces multimodal-guided control, unified noise initialization, and degradation-aware training. Recent efforts (Yin et al., 2025; Huang et al., 2025; Gu et al., 2025; Teng et al., 2025; Zhou et al., 2025; Deng et al., 2025) have advanced causal AR-based models for long video generation. CausVid (Yin et al., 2025) reformulates bidirectional video diffusion into a causal AR process, using distribution matching distillation to compress multi-step denoising into a few steps. FAR (Gu et al., 2025) further enhances AR generation by combining a high-resolution short-term context with a compressed long-term context via flexible positional encoding. MAGI-1 (Teng et al., 2025) scales AR video generation to large models and datasets through chunk-wise prediction. Most recently, Self-forcing (Huang et al., 2025) addresses the train-test gap in AR video diffusion by simulating inference conditions during training, rolling out generation with KV cache, and conditioning on model outputs. Despite the promise of purely AR for long video generation, achieving real-time efficiency and maintaining high quality simultaneously remains an open challenge.

Recent works have begun exploring interactive video generation, where users can directly influence generation in real time through text or keyboard prompts. The Matrix (Feng et al., 2024) demonstrates infinite-horizon world generation with first- and third-person control, using a shifted window denoising process. Yume (Mao et al., 2025) builds an interactive world generation pipeline capable

of constructing explorable environments from a single image, video, or text, allowing responsive user navigation. Matrix-Game (Zhang et al., 2025) employs large-scale pretraining and action-labeled finetuning to produce controllable, high-fidelity video conditioned on reference frames, motion context, and user actions. While effective, these methods are specifically tailored for interactive video generation in video game environments, such as Minecraft and GTA. MAGI-1 (Teng et al., 2025) supports general interaction, but its prompt switching requires manual adjustment of KV-cache windows at different steps, which complicates practical use.

#### E Training Prompt Generation

LONGLIVE does not require video data since we adopt a self-training method. It relies only on a set of prompts to teach the model with interaction ability (Yang et al., 2024b; 2021; 2023). To efficiently produce appropriate, reasonable, and safe interactive prompts, we employ the Qwen2-72B-Instruct (Yang et al., 2024a) LLM. Given a source prompt from VidProM (Wang & Yang, 2024), we instruct Qwen2-72B-Instruct to synthesize the next scene under several constraints. The instruction template is shown below.

You are a video-prompt generation specialist. Your task

- $\bullet$  Receive an ORIGINAL\_PROMPT for the first part of a continuous shot.
- $\bullet$  Write one stand-alone English paragraph (80{100 words) that shows the next moment of the same shot.
- \*\*Add exactly one new action/object for the existing main subject.\*\*
- $\bullet$  Keep setting, subject, mood, style, camera scale, and camera movement or angle exactly as in the <code>ORIGINAL\_PROMPT</code> .
- · Elements may vanish only if naturally obscured by the new action.
- Do \*\*not\*\* use phrases like \*still, as before, continues\* that reveal you read the prior text.
- · Use clear mid-level English; avoid rare or literary words.
- End the paragraph with \*\*the same camera keywords that appear at the end of the ORIGINAL\_PROMPT\*\*, separated by single spaces, no brackets.
- \*\*Output format MUST be exactly one line, wrapped between <OUTPUT> and </OUTPUT>.\*\*
- $\bullet$  Do \*\*NOT\*\* add explanations, greetings, headings, numbering, markdown, or extra lines.
- · Anything written outside the two tags will be ignored.

#### <span id="page-15-0"></span>F TRAINING DETAILS

#### F.1 IMPLEMENTATION

We first adapt the pretrained Wan2.1-T2V-1.3B into a chunk-wise autoregressive (AR) causal-attention model. First, we conduct an ODE initialization as the same as self-forcing. Then we train the model with DMD, but switch to short-window attention with frame-sink tokens: the chunk size is 3 latent frames, the local attention window is 9 frames, and the first chunk (3 latent frames) serves as the sink. After this initialization, we perform streaming long-tuning strictly following Algorithm 1: at each iteration, we roll out a 5 s clip and supervise the student using Wan2.1-T2V-14B as the teacher. Optimization uses AdamW for both actor and critic with learning rates  $lr=1.0\times10^{-5}$  (actor) and  $lr_{\rm critic}=2.0\times10^{-6}$ ; we set  $\beta_1=0.0,\,\beta_2=0.999$  for the actor and  $\beta_{1,{\rm critic}}=0.0,\,\beta_{2,{\rm critic}}=0.999$  for the critic. Training is conducted on 64 GPUs with one sample per GPU (global batch size =64). We apply EMA to the actor with decay 0.99, starting at step 200. The maximum sequence length is set to the target inference horizon; both  $60\,\mathrm{s}$  and  $240\,\mathrm{s}$  work well in practice. For the  $60\,\mathrm{s}$  setting, we train for 3,000 iterations.

#### F.2 LORA TUNING

Motivated by LongLora (Chen et al., 2024c), we assume that improving the quality of long context does not require a full model fine-tuning. We therefore adopt LoRA tuning throughout the streaming long tuning procedure. Interestingly, we find that effective long-range generation demands relatively high adapter ranks; in our setup, the resulting adapters require 256 ranks, making roughly 27% of

<span id="page-16-1"></span>Table A: LoRA budget vs. performance on VBench-Long. A moderate budget approaches fullmodel quality with far fewer trainable parameters.

| LoRA rank            | 32    | 64    | 128   | 256   | 512   | Full Model |
|----------------------|-------|-------|-------|-------|-------|------------|
| Trainable Parameters | 44 M  | 87 M  | 175 M | 350 M | 700 M | 1.3 B      |
| Total Score          | 81.08 | 82.68 | 82.98 | 83.12 | 83.04 | 83.52      |

<span id="page-16-2"></span>Table B: INT8-Quantized results on VBench. FPS is measured on a single NVIDIA 5090 GPU.

| Precision | Model Size | Throughput (FPS) | Total | Quality | Semantic |
|-----------|------------|------------------|-------|---------|----------|
| INT8      | 1.4 GB     | 16.4             | 84.31 | 86.20   | 76.74    |
| BF16      | 2.7 GB     | 12.6             | 84.87 | 86.97   | 76.47    |

the model's parameters trainable. Even so, LoRA substantially reduces the training footprint, cutting the parameter/optimizer state to about 27% of that required by full fine-tuning (i.e., 73% savings).

We ablate LoRA tuning in Table [A.](#page-16-1) We measure the 30s long-video quality by VBench-long. Scaling the LoRA budget improves quality until a saturation point, with the rank 256 configuration achieving the best while still training far fewer parameters than full fine-tuning.

#### <span id="page-16-0"></span>G QUANTIZATION

We quantize LONGLIVE to INT8 via post-training quantization [\(Li\\* et al.,](#page-10-15) [2025\)](#page-10-15). As shown in Table [B,](#page-16-2) this reduces LONGLIVE's model size by 1.9× and improves throughput by 1.3×, with minimal degradation on VBench (Table [B\)](#page-16-2).

#### H INTERACTIVE LONG VIDEO SHOWCASES

We present interactive 60s videos generated with six sequential prompts in Figure [A](#page-17-0) and Figure [B.](#page-18-0) See our [Demo Page](https://nvlabs.github.io/LongLive) for more examples.

