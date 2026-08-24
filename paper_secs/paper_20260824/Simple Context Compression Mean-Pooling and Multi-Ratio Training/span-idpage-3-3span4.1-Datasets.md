# <span id="page-3-3"></span>**4.1 Datasets**

The suite comprises two evaluation tiers spanning different context lengths, together covering short (<1K tokens) and mid-range (<8K tokens) inputs.

**Short-context Benchmarks** Six reading comprehension datasets form the primary evaluation [\(Table 1\)](#page-3-0): SQuAD [\(Rajpurkar et al.,](#page-12-5) [2016\)](#page-12-5), NarrativeQA [\(Koˇciský et al.,](#page-10-3) [2018\)](#page-10-3), HotpotQA [\(Yang et al.,](#page-13-1) [2018\)](#page-13-1), AdversarialQA [\(Bartolo et al.,](#page-9-4) [2020\)](#page-9-4), TriviaQA [\(Joshi et al.,](#page-10-4) [2017\)](#page-10-4) (verified subset), and ParaphraseRC [\(Saha et al.,](#page-12-4) [2018\)](#page-12-4). This selection spans reasoning styles from factual extraction to adversarial paraphrasing. The benchmark explicitly separates in-domain and out-of-domain evaluation: training includes SQuAD, NarrativeQA, and HotpotQA (in-domain), while AdversarialQA, TriviaQA, and ParaphraseRC are held out entirely (out-of-domain).[3](#page-3-1)

**Mid-range Context Benchmarks** To assess how compression methods scale to longer inputs, where computational savings are more valuable, we include QA tasks from LongBench-E [\(Bai et al.,](#page-9-5) [2024\)](#page-9-5) with contexts up to 8K tokens [\(Table 2\)](#page-3-2).

<span id="page-3-1"></span><sup>3</sup> In- vs. out-of-domain results are in [Appendix B.](#page-16-0)

## 4.2 Training Data

A key component of the evaluation suite is a standardized training mixture. Performance differences between compression methods can stem from the method itself or from the training data; without controlling for both, it is impossible to attribute gains cleanly. Our training mixture draws from the train splits of the in-domain QA datasets as well as summarization tasks; full details appear in Table 5. As in the construction of the evaluation suite, our guiding principle is to only include datasets whose contexts are guaranteed to contain the necessary evidence for completing the task at hand. We recommend that future evaluations use a shared training mixture, or at minimum report the mixture used, so that methodological contributions can be disentangled from data effects.

#### 4.3 Metrics

We evaluate using *exact match* (EM) and  $F_1$ .<sup>4</sup> We do not use the substring accuracy metric (score of 1 if the gold answer is a substring of the model output) as it is easily exploitable,<sup>5</sup> which forces us to exclude some baselines from primary comparisons.

For each metric we define a *teacher-normalized* version. Given metric M, teacher  $\mathcal{M}$ , and compressor  $f_c$ : let  $M_T$  be the teacher's score with full context,  $M_T^{\varnothing}$  the no-context score, and  $M_{f_c}$  the score with compressed context. The teacher-normalized score is:

$$\frac{M_{f_c} - M_T^{\varnothing}}{M_T - M_T^{\varnothing}} .$$

This scales performance relative to the teacher and corrects for questions answerable without context, enabling fair cross-model comparison regardless of teacher quality.

#### 4.4 Reference Systems

Although the suite can accommodate any compression paradigm, in this work we focus on soft context compression, where we find that simple baselines can improve markedly over existing practice. In addition to the two baselines we introduce (Section 5), we evaluate several existing soft compression methods: ICAE (Ge et al., 2024) and PCC (Dai et al., 2025). To demonstrate the suite's cross-paradigm applicability, we also evaluate LLMLingua2 (Pan et al., 2024), a hard-prompt compression approach, by passing its compressed prompts to our finetuned Qwen3-8B teacher model.

Comparison across methods is challenging due to inconsistencies in training procedures, available code, and supported metrics. Our primary goal is to map the architecture land-scape systematically. For example, PISCO (Louis et al., 2025) was evaluated only using substring accuracy; we omit it from primary comparisons but report substring accuracy in Appendix E.1 (Table 10). These difficulties further motivate standardized evaluation.

## <span id="page-4-2"></span>5 Baseline Methods

We establish two simple baselines for soft context compression, both trained via knowledge distillation from a teacher LLM with access to the full uncompressed context. The first baseline, *mean pooling*, is a compression operator that averages adjacent hidden states after bidirectional encoding, without adding additional parameters beyond the encoder. The second, *bidirectional compression tokens*, is a straightforward modification to the widely-used causal compression-token approach in which the compression tokens attend bidirectionally among themselves. Both baselines are considerably stronger than the standard causal compression-token approach. Figure 1 provides an overview of the processing strategies we compare.

<span id="page-4-0"></span> $<sup>^{4}</sup>$ We show only  $F_{1}$  in the main text; Table 9 shows full EM results.

<span id="page-4-1"></span><sup>&</sup>lt;sup>5</sup>E.g., listing all 50 US states as the answer to any state-valued question.

