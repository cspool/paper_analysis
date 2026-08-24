# C More Discussion

## <span id="page-16-0"></span>C.1 Using More Continuous Thoughts

In Figure [8](#page-9-1) (II), we present the performance of Coconut on GSM8k using c ∈ {0, 1, 2}. When experimenting with c = 3, we observe a slight performance drop accompanied by increased variance. Analysis of the training logs indicates that adding three continuous thoughts at once – particularly during the final stage transition – leads to a sharp spike in training loss, causing instability. Future work will explore finer-grained schedules, such as incrementally adding continuous thoughts one at a time while removing fewer language tokens, as in iCoT [\(Deng et al.,](#page-11-5) [2024\)](#page-11-5). Additionally, combining language and latent reasoning—e.g., generating the reasoning skeleton in language and completing the reasoning process in latent space—could provide a promising direction for improving performance and stability.

#### C.2 Coconut with Larger Models

<span id="page-16-3"></span>We experimented with Coconut on GSM8k using Llama 3.2-3B and Llama 3-8B [\(Dubey et al.,](#page-11-0) [2024\)](#page-11-0) with c = 1. We train them for 3 epochs in Stage 0, followed by 1 epoch per subsequent stage. The results are shown in Table [5.](#page-16-3)

| Model                      | no-CoT       | Coconut (Ours) |
|----------------------------|--------------|----------------|
| Llama 3.2-3B<br>Llama 3-8B | 26.0<br>42.2 | 31.7<br>43.6   |
|                            |              |                |

Table 5 Experimental results of applying Coconut to larger Llama models. We report performance comparisons between models without CoT reasoning (no-CoT) and our proposed Coconut method.

We observe consistent performance gains across both Llama 3.2-3B and Llama 3-8B models compared to the no-CoT baseline, though these improvements are not as pronounced as those previously demonstrated

<span id="page-16-2"></span><sup>3</sup><https://github.com/huggingface/transformers>

with GPT-2. One possible reason is that larger models have already undergone extensive language-focused pre-training, making the transition to latent reasoning more challenging.

We emphasize that the primary goal of this paper is to highlight the promising attributes of latent-space reasoning and to initiate exploration in this new direction. Universally surpassing language-based CoT likely requires significant research efforts dedicated to latent space pre-training. We are encouraged by recent progress in this area [\(Geiping et al.,](#page-11-15) [2025;](#page-11-15) [Barrault et al.,](#page-11-14) [2024;](#page-11-14) [Gladstone et al.,](#page-11-16) [2025\)](#page-11-16). While these recent models provide scalable methods for latent representation learning, the latent spaces have not yet been explicitly optimized for reasoning. Integrating these recent advancements with Coconut presents an exciting and promising avenue for future research.