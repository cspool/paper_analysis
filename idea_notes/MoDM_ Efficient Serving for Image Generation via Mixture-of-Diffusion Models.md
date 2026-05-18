## MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

- baseline方法是什么？
  Baseline方法分为三类：(1) **Vanilla System**：所有请求用单一large diffusion model（SD3.5L或FLUX）做全量T=50步去噪推理，无任何缓存。全栈执行例子：算法层diffusion model T=50步iterative denoising→系统框架层单一模型worker处理所有请求，无调度优化→编译框架层论文未明确说明→kernel调度层标准PyTorch diffusion推理kernel→硬件架构层A40/MI210 GPU。缺陷：每请求完整去噪过程计算量大、延迟高、吞吐低。(2) **Nirvana (latent caching)**：缓存diffusion model中间latent representations，text-to-text similarity检索复用，跳过少量去噪步数。全栈执行例子：算法层diffusion modeldenoising→系统框架层Nirvana text-to-text检索latent cache→编译框架层论文未明确说明→kernel调度层标准diffusion kernel→硬件架构层A40/MI210 GPU。缺陷：latent cache 2.5MB/张存储开销大；model-dependent，同一cache不能跨模型复用；text-to-text retrieval视觉对齐差（CLIPScore mean 0.22 vs text-to-image 0.28）；>90% cache hit rate下仅20% latency reduction；高请求率下SLO violation频繁。(3) **Pinecone (retrieval-only)**：基于CLIP text embedding similarity直接检索并返回最相似cached image，无refine。缺陷：无generative refinement导致CLIPScore显著低于生成方法，图像-文本对齐弱。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**MoDM (Mixture-of-Diffusion Models)**：基于final image缓存+CLIP text-to-image similarity检索+小型扩散模型refine+大型模型全量推理的混合serving系统，通过PID-driven的Global Monitor动态分配GPU资源实现延迟-质量自适应平衡。

  论文方法全栈执行例子（以SD3.5L作为大模型、SDXL作为小模型、4×A40为例）：
  - 算法层：final image caching替代latent caching（1.4MB vs 2.5MB/张，100K embedding仅0.29GB）；text-to-image CLIP similarity检索（cosine sim公式1）；quality-constrained dynamic k-selection heuristic (Fig.5b, α≥0.95)；noise re-introduction Ĩ=σ_tk·ε+(1-σ_tk)·I*重新进入去噪流程→小模型执行剩余T-k步去噪（公式4 compute savings）
  - 系统框架层：Request Scheduler管理CLIP embedding提取+缓存查取+请求路由(cache-hit→小模型queue, cache-miss→大模型queue)→Global Monitor PID controller (Algorithm 1, Kp=0.6,Ki=0.05,Kd=0.05) 统计R/H_cache/k分布→Quality-Optimized Mode (最大化N_large) 或Throughput-Optimized Mode (cache-hit全用小模型)→N个GPU Worker动态加载大/小模型→FIFO cache维护(>90% hits在4h内)
  - 编译框架层：论文未明确说明
  - kernel调度层：标准PyTorch diffusion kernel，未做kernel修改。cache检索GPU上cosine similarity计算0.05s/100K images
  - 硬件架构层：4×A40 (48GB) 或16×4×MI210 (64GB)，PyTorch RPC跨进程通信

  对应解决Baseline缺陷：
  (1) **Nirvana latent cache存储开销大+model-dependent** → final image cache模型无关（PNG/JPEG标准格式），1.4MB/张 vs 2.5MB/张，跨Stable Diffusion/SANA/FLUX多模型族复用
  (2) **Nirvana text-to-text retrieval视觉对齐差** → text-to-image CLIP similarity检索，CLIPScore mean 0.28 vs 0.22，PickScore mean 20.33 vs 19.52，检索到视觉上更符合prompt的图像
  (3) **Nirvana >90% hit rate仅20% latency reduction** → 混合大/小模型：cache-hit用小模型refine (每步compute cost更低)，cache-miss用大模型保证quality。DiffusionDB上MoDM-SANA达到3.2× throughput、46.7% energy savings、66.3% energy savings with SANA as small model
  (4) **单一模型无法应对负载波动导致SLO violation** → PID-driven Global Monitor动态调整N_large/N_small分配，支持Quality-Optimized和Throughput-Optimized双模式，请求率波动时自动切换small model类型（SDXL→SANA），在4×A40上支持10 req/min without SLO violation (2× threshold)，而Vanilla仅5 req/min、Nirvana仅6 req/min即出现显著SLO violation
  (5) **Pinecone retrieval无refine导致CLIPScore低** → cache-hit请求加噪后用小模型refine T-k步，保证visual quality接近全量大模型(99.7% baseline CLIPScore)，显著优于retrieval-only
  (6) **蒸馏小模型(SD3.5L-Turbo)静态降低质量** → MoDM的mix-of-models策略使cache-hit用小模型+缓存的high-quality image初始化，FID远低于standalone小/蒸馏模型(MoDM-SDXL FID 11.85 vs SDXL standalone 16.29 on DiffusionDB)
