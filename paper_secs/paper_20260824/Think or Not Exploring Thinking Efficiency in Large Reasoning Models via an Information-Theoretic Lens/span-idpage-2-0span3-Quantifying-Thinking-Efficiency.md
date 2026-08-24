# <span id="page-2-0"></span>3 Quantifying Thinking Efficiency

This section introduces a formal framework to measure reasoning efficiency by segmenting the thought process, analyzing divergence from ideal reasoning paths, and computing stepwise information gains.

## 3.1 Semantic Segmentation of Thinking Processes

Human reasoning typically unfolds in discrete, sequential steps. The means–ends analysis framework [\[Simon and Newell,](#page-11-9) [1971\]](#page-11-9) views problem solving as a series of goal-subgoal transitions, each representing a cognitive operation. Similarly, ACT-R [\[Anderson et al.,](#page-9-6) [1997,](#page-9-6) [Whitehill,](#page-12-6) [2013\]](#page-12-6) models reasoning as rule-based production sequences, while dual-process theory [\[Kahneman,](#page-10-11) [2011,](#page-10-11) [Evans,](#page-9-7) [2003\]](#page-9-7) characterizes "System 2" reasoning as deliberate and decomposable. Collectively, these theories motivate modeling reasoning as a structured sequence of semantically meaningful steps.

Accordingly, we segment a model's output reasoning path S into discrete semantic units S = {s1, s2, · · · , sn}, where each s<sup>i</sup> represents a minimal step that contributes semantically to the overall process. For example, "solving 2x + 5 = 15" triggers steps as s1: subtract 5 from both sides → s2: divide both sides by 2 → s3: solve for x. These segments serve as the atomic elements for downstream information-theoretic analysis. The segmentation can be performed based on syntactic cues (e.g., clause or sentence boundaries), manual annotation, or automated approaches such as LLMassisted chunking. By operating at this granularity, we enable a finer analysis of how incremental reasoning steps influence uncertainty and information flow throughout the trajectory.

## <span id="page-2-1"></span>3.2 Response-Level: Measuring Information Bias in Entire Trajectories

While S captures the model's observable reasoning path, we posit the existence of a latent, ideal trajectory T = {t1, t2, · · · , tm} representing the correct reasoning steps for a given question Q. This ideal trajectory may correspond to a human-annotated, cognitively plausible reasoning path, or reflect implicit reasoning steps within the model itself [\[Gan et al.,](#page-9-2) [2025\]](#page-9-2), which may differ from its explicit outputs. To measure how closely the model's reasoning aligns with this ground truth, we introduce

information bias, a metric based on mutual information:

InfoBias
$$(S,T) = -I(s_{1:n}, t_{1:m}) = H(s_{1:n}, t_{1:m}) - H(s_{1:n}) - H(t_{1:m}),$$
 (1)

where I denotes mutual information and H represents entropy. This discrepancy can be estimated via sampling, under the assumption that the generated reasoning trajectories s and t are two conditionally independent stochastic processes, and their joint distribution can be approximated through N samples. Applying the KL-based estimation of mutual information [Paninski, 2003], we derive the following upper bound on the information bias:

$$|\hat{I}_N(S,T) - I(S,T)| \le \sqrt{\frac{2\log(2/\delta)}{N}} + \mathcal{O}\left(\frac{1}{N}\right),\tag{2}$$

where  $\delta$  denotes the confidence level. This bound guarantees that the empirical estimate  $\hat{I}_N(S,T)$  converges to the true mutual information I(S,T) as N increases, establishing InfoBias as a statistically consistent metric. Crucially, this enables reliable estimation of the alignment between observable and latent reasoning trajectories using a finite number of sampled inference steps.

#### <span id="page-3-0"></span>3.3 Step-Level: Measuring Information Gain at Each Step

Beyond the trajectory as a whole, at the semantic level, we aim to quantify how each individual reasoning step contributes to answer inference. Efficient reasoning should progressively reduce uncertainty over the answer space [Sui et al., 2025]. Given a set of candidate answers  $A = \{a_1, a_2, \cdots, a_l\}$ , we can compute the conditional entropy at step i:

$$H_i = -\sum_{k=1}^{l} P(a_k|Q; s_{1:i}) \log P(a_k|Q; s_{1:i}), \tag{3}$$

where  $P(a_k|Q;s_{1:i})$  is estimated from the model's output probabilities. Specifically, we concatenate the given question Q, the model's intermediate reasoning steps  $s_{1:i}$ , and the final answer prompt to form the input sequence (See Appendix C.3 for details). The model's predicted probability of the next token is then used as the basis for evaluation. The information gain at step i is:

$$\Delta I_i = H_{i-1} - H_i,\tag{4}$$

which quantifies how much uncertainty is reduced by incorporating step  $s_i$ . This reflects the extent to which each reasoning step clarifies the answer distribution. We further define a **targeted information** gain with respect to the correct answer  $c \in A$ :

$$\Delta I_i^c = -\log P(c|Q; s_{1:i}) - (-\log P(c|Q; s_{1:i-1})) = \log \frac{P(c|Q; s_{1:i-1})}{P(c|Q; s_{1:i})},\tag{5}$$

capturing how each step influences the model's confidence in the correct option. Together,  $\Delta I_i$  and  $\Delta I_i^c$  reveal fine-grained reasoning efficiency, highlighting impactful steps toward the correct answer.

#### <span id="page-3-1"></span>3.4 Empirical Evaluation and Insights

We empirically validate the methods proposed in §3.2 and §3.3, which respectively target the responselevel relationship between reasoning length and InfoBias, and the step-level impact of individual reasoning steps on InfoGain. These analyses aim to assess the effectiveness of the informationtheoretic metrics in capturing the dynamics and quality of reasoning exhibited by LLMs.

