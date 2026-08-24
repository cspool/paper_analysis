# Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning

Minwu Kim\* † Anubhav Shrestha\* Safal Shrestha Aadim Nepal Keith Ross New York University Abu Dhabi

## Abstract

Recent studies have shown that reinforcement learning with verifiable rewards (RLVR) enhances overall accuracy (pass@1) but often fails to improve capability (pass@k) of LLMs in reasoning tasks, while distillation can improve both. In this paper, we investigate the mechanisms behind these phenomena. First, we demonstrate that RLVR struggles to improve capability as it focuses on improving the accuracy of the easier questions to the detriment of the accuracy of the most difficult questions. Second, we show that RLVR does not merely increase the success probability for the easier questions, but in our small model settings, produces quality responses that were absent in its original output distribution. In addition, we show these responses are neither noticeably longer nor feature more reflectionrelated keywords, underscoring the need for more reliable indicators of response quality. Third, from the experiment distilling teacher responses to in-distribution problems, we find that capability does not always improve with distillation. We conjecture that capability improves only when new knowledge is introduced, whereas distilling reasoning patterns only improves accuracy but not capability, sacrificing performance on the most difficult questions, similar to RLVR. Together, these findings offer a clearer understanding of how RLVR and distillation shape reasoning behavior in LLMs.[1](#page-0-0)

## 1 Introduction

Large language models (LLMs) have made rapid progress in complex domains such as mathematics and programming. A key development is the emergence of reasoning models [\(OpenAI,](#page-9-0) [2024;](#page-9-0) [DeepSeek-AI,](#page-9-1) [2025;](#page-9-1) [MoonshotAI,](#page-9-2) [2025\)](#page-9-2), which outperform conventional LLMs by employing more

advanced reasoning strategies. Instead of relying on linear chains of thought (CoTs) [\(Wei et al.,](#page-10-0) [2022\)](#page-10-0), these models exhibit non-linear behaviors such as subgoal formulation, verification, backtracking, and backward chaining [\(Gandhi et al.,](#page-9-3) [2025;](#page-9-3) [Xiang et al.,](#page-10-1) [2025\)](#page-10-1).

A central technique behind recent reasoning models is *reinforcement learning with verifiable rewards (RLVR)* [\(Lambert et al.,](#page-9-4) [2024;](#page-9-4) [DeepSeek-](#page-9-1)[AI,](#page-9-1) [2025\)](#page-9-1). RLVR fine-tunes pre-trained LLMs using rewards based on whether the model's output matches a ground-truth solution. Without explicit supervision, the model learns complex reasoning behaviors during training, making RLVR an effective fine-tuning method for reasoning tasks. In addition, *distillation* with responses from stronger reasoning models also provides strong performance, demonstrating that reasoning ability can be effectively transferred through supervised learning [\(Min](#page-9-5) [et al.,](#page-9-5) [2024;](#page-9-5) [DeepSeek-AI,](#page-9-1) [2025;](#page-9-1) [Muennighoff](#page-9-6) [et al.,](#page-9-6) [2025;](#page-9-6) [Ye et al.,](#page-10-2) [2025\)](#page-10-2).

It is well established that RLVR improves *accuracy*—the probability of generating a correct answer, but whether it also improves *capability*—the probability that a correct answer exists in the model's output distribution—remains debated. Some studies suggest that, with sufficient compute and carefully matched training and test sets in skills and difficulty, RLVR can solve tasks that were previously unsolvable in certain domains [\(Liu et al.,](#page-9-7) [2025a;](#page-9-7) [Setlur et al.,](#page-10-3) [2025;](#page-10-3) [Sun et al.,](#page-10-4) [2025\)](#page-10-4). Others, however, report that in more typical settings—where training and test sets contain heterogeneous problems with uncontrolled knowledge and difficulty—RLVR primarily amplifies existing reasoning rather than expanding capability [\(Dang](#page-9-8) [et al.,](#page-9-8) [2025;](#page-9-8) [Wu et al.,](#page-10-5) [2025;](#page-10-5) [Yue et al.,](#page-10-6) [2025;](#page-10-6) [Zhao](#page-11-0) [et al.,](#page-11-0) [2025\)](#page-11-0). By contrast, it has been observed that distillation improves both accuracy and capability [\(Yue et al.,](#page-10-6) [2025\)](#page-10-6). In this paper, we take a closer look at how RLVR and distillation shape

<sup>\*</sup>Equal contribution.

<span id="page-0-0"></span><sup>†</sup>Correspondence to mwk300@nyu.edu.

<sup>1</sup>Code available at [https://github.com/minwukim/](https://github.com/minwukim/RLvsDistillation) [RLvsDistillation](https://github.com/minwukim/RLvsDistillation)

mathematical reasoning in LLMs under typical settings, where training and test sets involve diverse problems with varying knowledge and difficulty.

Carrying out experiments with two models, Qwen2.5-1.5B-Math [\(Yang et al.,](#page-10-7) [2024\)](#page-10-7) and Qwen2.5-3B [\(Hui et al.,](#page-9-9) [2024\)](#page-9-9), we first demonstrate that RLVR does not improve capability because RLVR focuses on improving the accuracy of the less-difficult questions to the detriment of the accuracy of the most difficult questions, explaining why capability does not improve and can even decrease. We argue that this "sacrificing-difficultproblems" phenomenon is a direct consequence of the underlying policy-gradient algorithm in GRPO (Shao et al., 2024). We further find that RLVR does not merely increase the success probability for the easier questions, but in our small model settings produces responses that are more direct with fewer keywords. In addition, we find these responses are neither noticeably longer nor richer in reflection-related keywords (e.g., "wait", "alternatively"), underscoring the need for more reliable indicators of reasoning quality.

We next examine teacher distillation. A teacher model's responses convey two main elements: (1) reasoning patterns and (2) domain knowledge. To disentangle their effects, we compare three models: the base model, the publicly released DeepSeek reasoning model, which is distilled on 800k responses from DeepSeek-R1 and likely incorporates substantial new knowledge, and our own reasoningonly model, trained only on teacher responses for questions where the base model is already able to produce correct answers. We find that both distilled models yield substantial accuracy gains, but only the DeepSeek model shows clear capability improvement. These results indicate that teacher distillation does not always expand capability, even when accuracy meaningfully improves. While further investigation is needed to confirm, we conjecture that this difference stems from whether new knowledge is introduced during distillation: introducing new knowledge may expand capability, whereas distilling only reasoning patterns improves accuracy but not capability. Interestingly, for the reasoning-only model, we also find that accuracy of the easier questions improves to the detriment of the most difficult questions, mirroring the RLVR.

Taken together, our findings provide a clearer picture of different dynamics in the model behavior during RLVR training and distillation, and offer insights into strategies for enhancing the fundamental

abilities of LLMs.

## <span id="page-1-0"></span>2 Related Work

Training reasoning models. RLVR has emerged as a key method for training LLMs to tackle complex reasoning tasks by generating long CoTs [\(DeepSeek-AI,](#page-9-1) [2025;](#page-9-1) [Lambert et al.,](#page-9-4) [2024;](#page-9-4) [OpenAI,](#page-9-0) [2024\)](#page-9-0). It has shown strong performance across model sizes [\(Gandhi et al.,](#page-9-3) [2025;](#page-9-3) [Hu et al.,](#page-9-10) [2025;](#page-9-10) [Liu et al.,](#page-9-11) [2025b;](#page-9-11) [Xu et al.,](#page-10-8) [2025;](#page-10-8) [Yeo et al.,](#page-10-9) [2025;](#page-10-9) [Zeng et al.,](#page-11-1) [2025\)](#page-11-1) and domains [\(Pan et al.,](#page-9-12) [2025;](#page-9-12) [Shrestha et al.,](#page-10-10) [2025;](#page-10-10) [Xie et al.,](#page-10-11) [2025;](#page-10-11) [Zhang et al.,](#page-11-2) [2025\)](#page-11-2). Numerous RLVR variants have also been proposed to improve performance, data efficiency, and computational cost [\(Fatemi et al.,](#page-9-13) [2025;](#page-9-13) [Liu](#page-9-11) [et al.,](#page-9-11) [2025b;](#page-9-11) [Shao et al.,](#page-10-12) [2025,](#page-10-12) [2024;](#page-10-13) [Wang et al.,](#page-10-14) [2025a](#page-10-14)[,b;](#page-10-15) [Yu et al.,](#page-10-16) [2025;](#page-10-16) [Zuo et al.,](#page-11-3) [2025\)](#page-11-3). Distilling high-quality CoT data is another effective approach for enhancing LLM reasoning. Such data are obtained either by prompting large models [\(Yu et al.,](#page-10-17) [2024;](#page-10-17) [Zelikman et al.,](#page-11-4) [2022\)](#page-11-4) or by human annotation of complex reasoning traces [\(Qin](#page-10-18) [et al.,](#page-10-18) [2024;](#page-10-18) [Xiang et al.,](#page-10-1) [2025;](#page-10-1) [Ye et al.,](#page-10-2) [2025\)](#page-10-2). A widely used strategy now involves distilling long CoT responses from RLVR-trained models into student models, often yielding substantial performance gains [\(Huang et al.,](#page-9-14) [2024;](#page-9-14) [Min et al.,](#page-9-5) [2024;](#page-9-5) [Muennighoff et al.,](#page-9-6) [2025;](#page-9-6) [Shrestha et al.,](#page-10-10) [2025\)](#page-10-10). Our work examines both RLVR and distillation, and evaluates how these two approaches differentially shape reasoning behavior in LLMs.

Capability expansion in RLVR. There is ongoing debate about whether RLVR develops genuinely new capabilities not already present in a model. Several works [\(Dang et al.,](#page-9-8) [2025;](#page-9-8) [Yue et al.,](#page-10-6) [2025;](#page-10-6) [Zhao et al.,](#page-11-0) [2025\)](#page-11-0) argue that RLVR merely amplifies correct reasoning already latent in the model. By contrast, ProRL demonstrates empirically that, given sufficient compute and diverse data, RLVR can enable models to solve previously unsolvable tasks in some domains-such as logic puzzles—suggesting the possibility of true capability expansion [\(Liu et al.,](#page-9-7) [2025a\)](#page-9-7). OMEGA provides a more controlled analysis by carefully adjusting the knowledge and difficulty requirements of training and test math problems. Their results show that models can generalize to higher difficulty levels when the required knowledge is the same, but remain weak at chaining compositional skills or adopting novel, unconventional strategies [\(Sun](#page-10-4) [et al.,](#page-10-4) [2025\)](#page-10-4). Similarly, e3 finds that only problems

with a sufficiently large verification-generation gap benefit from test-time scaling, through experimenting under settings where the problem types of training and test sets are strictly controlled (Setlur et al., 2025). However, outside such carefully constrained conditions—in typical scenarios where both training and test sets consist of heterogeneous problems with uncontrolled knowledge and difficulty-studies consistently find that RLVR does not substantially expand capability. Theoretical analysis conducted by Wu et al. further argues that, in general, the shrinkage of empirical support outweighs its expansion in such scenarios, leading to little capability gain in RLVR (Wu et al., 2025). In this work, we analyze RLVR under such general, uncontrolled math problem settings. By examining how accuracy shifts across difficulty levels, we show that RLVR often fails to improve capability as it tends to deliver gains on easier problems at the expense of performance on harder ones.

Reasoning pattern and knowledge in distillation. Several studies have examined the respective roles of domain knowledge and reasoning patterns in improving accuracy through distillation. For instance, Shrestha et al. distill teacher responses from logic puzzles—where domain knowledge is minimal—and show that transferring reasoning patterns alone can yield substantial performance gains across domains such as mathematics and coding (Shrestha et al., 2025). Similarly, Huan et al. demonstrate that distilling math problem responses leads to notable improvements in reasoning tasks in other domains (Huan et al., 2025). However, work on capability remains limited. Yue et al. suggest that distillation can drive capability expansion, but their analysis does not disentangle the effects of reasoning patterns and knowledge injection (Yue et al., 2025). In contrast, our study explicitly controls for this distinction and investigates how each factor differentially influences model capability.

## <span id="page-2-0"></span>3 Accuracy and Capability

#### 3.1 Formal Definitions

We evaluate models along two dimensions: *accuracy* and *capability*. Informally, accuracy measures how likely a model is to generate a correct answer in a single attempt, while capability measures whether a correct answer exists within the model's response distribution.

Formally, we define accuracy and capability with respect to given model  ${\cal M}$  and evaluation dataset

 $\mathcal{D}=\{1,\dots,N\}$  of N questions. Let  $p_i^M$  denote the probability that model M successfully solves question i in a single attempt. Note that this can be obtained by sampling model M for n times on question i, computing the fraction of correct responses, and taking the limit as  $n\to\infty$ . In theory, an LLM using softmax sampling assigns non-zero probability to all valid outputs, so any answer could eventually be produced. To make capability practically meaningful, we consider a question i to be in-distribution for model M if  $p_i^M > \epsilon$ , where  $\epsilon$  is a small threshold (typically  $10^{-2}$  to  $10^{-3}$ ).

To evaluate performance under multiple attempts, let  $p_{i,k}^M$  denote the probability that model M solves question i at least once across k independent attempts. This probability satisfies

$$p_{i,k}^M = 1 - (1 - p_i^M)^k$$
.

With these definitions in place, we define the model's *accuracy* as the average success rate over the entire dataset:

$$Acc(M) = \frac{1}{N} \sum_{i \in \mathcal{D}} p_i^M.$$

We define the model's pass@k capability as the average success probability over  $\mathcal{D}$  given k passes per question:

$$\operatorname{Cap}_{k}(M) = \frac{1}{N} \sum_{i \in \mathcal{D}} p_{i,k}^{M} = \frac{1}{N} \sum_{i \in \mathcal{D}} \left( 1 - (1 - p_{i}^{M})^{k} \right)$$

It is important to note that if model M' has higher accuracy than model M ( $p_i^{M'} > p_i^M$ ) for a specific question, then it will also have higher pass@k capability for that question. However, this relationship does not always hold taking into account the entire dataset. In fact, as shown in Appendix A.1, it is possible for Acc(M') > Acc(M) while  $Cap_k(M') < Cap_k(M)$ .

#### <span id="page-2-1"></span>3.2 Estimating Accuracy and Capability

In practice, it is infeasible to compute the exact accuracy and capability of a model, as this would require a prohibitively large number of samples per question. Instead, we estimate these quantities empirically using a finite number of samples k. Let  $X_{i,k}$  be the number of correct responses among k samples for question i.

We estimate accuracy as:

$$Acc(M) \approx \frac{1}{N} \sum_{i \in \mathcal{D}} \frac{X_{i,k}}{k}$$

We estimate *pass@k capability* as:

$$\operatorname{Cap}_k(M) \approx \frac{1}{N} \sum_{i \in \mathcal{D}} 1(X_{i,k} > 0)$$

These estimators are unbiased. Throughout this work, we report results using these estimators, typically with k=256. We also consider a question i to be out-of-distribution if  $X_{i,256}=0$ , that is, none of the 256 responses to question i are correct. Under this definition, we can say with 95% confidence that  $p_i^M < 1 - (0.05)^{1/256} \approx 0.012$ , that is, question i is truly out-of-distribution under the threshold  $\epsilon=0.012$ .

