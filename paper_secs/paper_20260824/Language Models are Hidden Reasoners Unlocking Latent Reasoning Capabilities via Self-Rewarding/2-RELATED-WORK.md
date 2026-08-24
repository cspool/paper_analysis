# 2 RELATED WORK

Prompt-based LLM Reasoning Prompt-based reasoning methods prove to be effective across various domains, such as math problem-solving [\(Polu & Sutskever,](#page-12-4) [2020;](#page-12-4) [Hendrycks et al.,](#page-11-4) [2021;](#page-11-4) [Cobbe et al.,](#page-10-3) [2021a\)](#page-10-3), logical reasoning [\(Sprague et al.,](#page-12-5) [2024\)](#page-12-5) and agentic tasks [\(Yao et al.,](#page-13-7) [2023a;](#page-13-7) [Shinn et al.,](#page-12-3) [2023;](#page-12-3) [Yao et al.,](#page-14-3) [2023b\)](#page-14-3). Chain-of-Thoughts or CoT [\(Wei et al.,](#page-13-1) [2022\)](#page-13-1) is the pioneering work that prompts LLMs to decompose challenging tasks into smaller reasoning steps. After that, two primary research directions further improved reasoning capabilities during inference. One direction searched over the reasoning trajectories against a process-based verifier, or reward model [\(Yao et al.,](#page-13-2) [2024;](#page-13-2) [Besta et al.,](#page-10-4) [2024;](#page-10-4) [Lightman et al.,](#page-11-5) [2023\)](#page-11-5). For example, tree-of-thoughts [\(Yao et al.,](#page-13-2) [2024\)](#page-13-2) explored over thoughts by depth-first search (DFS), breadth-first search (BFS) or beam search. The other approach used a critic model to provide verbal feedback, iteratively refining the responses with that feedback [\(Saunders et al.,](#page-12-6) [2022;](#page-12-6) [Shinn et al.,](#page-12-3) [2023;](#page-12-3) [Yao et al.,](#page-14-3) [2023b;](#page-14-3) [Madaan et al.,](#page-12-7) [2023\)](#page-12-7).

Self-Rewarding for LLM Reasoning Reasoning capabilities in LLMs can be enhanced in posttraining through self-rewarding and reinforcement learning. The Self-Taught Reasoner, or STaR [\(Ze](#page-14-1)[likman et al.,](#page-14-1) [2022\)](#page-14-1) introduced a bootstrapping technique that allows LLMs to generate rationales and fine-tune itself with self-generated reasoning paths. Quiet-STaR [\(Zelikman et al.,](#page-14-2) [2024\)](#page-14-2) extended this by training LLMs to infer implicit rationales across arbitrary text, enhancing both reasoning and predictive abilities without task-specific fine-tuning. Reinforced Fine-Tuning, or ReFT [\(Trung](#page-13-8) [et al.,](#page-13-8) [2024\)](#page-13-8) took this further by leveraging reinforcement learning to improve generalization in reasoning tasks like math problem-solving, enabling LLMs to learn from multiple reasoning paths. Self-correction capabilities in LLMs can also be reinforced through self-generated data [\(Kumar et al.,](#page-11-6) [2024\)](#page-11-6). Lastly, [Hoffman et al.](#page-11-7) [\(2024\)](#page-11-7); [Hu et al.](#page-11-8) [\(2023\)](#page-11-8) formulated the reasoning process as latent variable models, aligning LLMs towards more accurate reasoning with fewer annotated data.

### <span id="page-2-2"></span>3 BACKGROUND ANMOTIVULATION

We start by briefly introducing reasoning techniques (*e.g.*, chain-of-thought [\(Wei et al.,](#page-13-1) [2022\)](#page-13-1), ReAct [\(Yao et al.,](#page-13-7) [2023a\)](#page-13-7), *etc*). Given a user query x, the standard procedure to sample the response y is to leverage an autoregressive pretrained LLMs π<sup>θ</sup> (parameterized by θ): y ∼ πθ(· | x). As for prompt-based reasoning techniques such as chain-of-thought [\(Wei et al.,](#page-13-1) [2022\)](#page-13-1), the LLM π<sup>θ</sup> is firstly asked to generate thoughts (*a.k.a* reasoning rationale) before generating the answers to the response:

$$\begin{split} \pmb{x'} := \text{Reason}(\pmb{x}) = \pmb{x} \oplus \text{Prompt Template of Thought}, \\ \pmb{z} \sim \pi_{\theta}(\cdot \mid \pmb{x'}), \quad \pmb{y} \sim \pi_{\theta}(\cdot \mid \pmb{x'} \oplus \pmb{z}) \,, \end{split}$$

where z is the thought or the reasoning rationale path, ⊕ indicates the concatenate operator, and the prompt template of the thought can be some hint prompt such as "Let's think s tep by step" .Empirically, people observe that there is a higher chance for the LLM π<sup>θ</sup> to generate the desired answer y following the above procedure than directly sampling the response y ∼ πθ(· | x). From a statistical perspective, we hypothesize that good reasoning rationales can significantly improve the probability of generating good answers y: ∃ z, *s.t.* πθ(y| x ⊕ z) ≫ πθ(y| x).

To validate the hypothesis, we check the probability of the correct answers y on pretrained LLMs with or without reasoning rationales. In Figure [2,](#page-2-1) we visualize the negative log probability of the correct answers on three different LLMs on GSM8K dataset [\(Cobbe et al.,](#page-10-2) [2021b\)](#page-10-2). We have observed that when the LLMs are conditioned on the reasoning rationales, the probability of the correct answer is remarkably larger than without reasoning rationales. This suggests that good reasoning rationales help the LLMs generate desired answers for complex tasks by increasing their probability, giving them a higher chance of generating the correct answers.

<span id="page-2-1"></span>> **[图片提取文字 (无描述)]:**
> 100 With Reasoning Without Reasoning 80 60 40 20 Mistral-7B-Instruct-v0.3 Meta-Llama-3.1-8B-Instruct Phi-3.5-mini-instruct
![](_page_2_Figure_8.jpeg)

Figure 2: Average negative log probabilities of LLMs to generate correct responses.

<span id="page-2-0"></span><sup>1</sup>We omit the difference between x ′ and x for convenience in the latter notation.

The above observation inspires us that we can potentially further improve the reasoning ability of existing LLMs. One may find some surrogate objective to enhance the quality of the reasoning rationales or improve the ability of LLMs to leverage good reasoning rationales. In the following Proposition 1, we show that *Self-Consistency* Chain-of-Thought (CoT-SC) (Wang et al., 2023), which takes a majority vote of multiple reasoning rationales to improve the reasoning ability, approximates some surrogate objective.

<span id="page-3-0"></span>**Proposition 1.** Denote the user query, model response, and reasoning rationale by  $\mathbf{x}, \mathbf{y}, \mathbf{z}$ , respectively. The distribution of the majority vote answer of the K reasoning rationales obtained by CoT-SC approximates  $p_M(\mathbf{y}|\mathbf{x}) := \mathbb{E}_{\mathbf{z} \sim \pi_{\theta}(\cdot|\mathbf{x})}[\pi_{\theta}(\mathbf{y}|\mathbf{x} \oplus \mathbf{z})]$ , as  $K \to \infty$ .

*Proof.* Given a user query  $\boldsymbol{x}$ , CoT-SC essentially follows the procedure: 1) Sample K *i.i.d* reasoning rationales together with model responses:  $(\boldsymbol{z}_k, \boldsymbol{y}_k) \sim \pi_{\theta}(\cdot|\boldsymbol{x}), \ 1 \leq k \leq K$ . 2) Take the majority vote of  $(\boldsymbol{y}_1, \dots, \boldsymbol{y}_K)$ . For a specific response  $\boldsymbol{y}$ , its frequency can be calculated as  $F(\boldsymbol{y}) := \frac{1}{K} \sum_{k=1}^K \mathbb{1}\{\boldsymbol{y}_k = \boldsymbol{y}\}$ , where  $\mathbb{1}$  is the indicator function. Then the expectation of  $F(\boldsymbol{y})$  is

$$\mathbb{E}_{\boldsymbol{y}_{1},...,\boldsymbol{y}_{K}}F(\boldsymbol{y}) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{E}_{\boldsymbol{y}_{i}} \mathbb{1}\{\boldsymbol{y}_{i} = \boldsymbol{y}\} = \frac{1}{K} \sum_{i=1}^{K} \mathbb{P}_{\boldsymbol{y}_{i} \sim \pi_{\theta}(\cdot | \boldsymbol{x} \oplus \boldsymbol{z}_{i})}[\boldsymbol{y}_{i} = \boldsymbol{y}]$$
$$= \frac{1}{K} \sum_{i=1}^{K} \pi_{\theta}(\boldsymbol{y} | \boldsymbol{x} \oplus \boldsymbol{z}_{i}) \stackrel{K \to \infty}{\longrightarrow} \mathbb{E}_{\boldsymbol{z} \sim \pi_{\theta}(\cdot | \boldsymbol{x})} \pi_{\theta}(\boldsymbol{y} | \boldsymbol{x} \oplus \boldsymbol{z}).$$

CoT-SC essentially leverages  $p_M(\boldsymbol{y}|\boldsymbol{x}) := \int \pi_{\theta}(\boldsymbol{z}|\boldsymbol{x}) \, \pi_{\theta}(\cdot|\boldsymbol{x} \oplus \boldsymbol{z}) d\boldsymbol{z}$  to obtain reasoning rationales and produce final correct answers. Inspired by the conclusion, we could leverage surrogate objectives like  $\mathbb{E}_{\boldsymbol{z} \sim \pi_{\theta}(\cdot|\boldsymbol{x})}[\phi(\pi_{\theta}(\boldsymbol{y}|\boldsymbol{x} \oplus \boldsymbol{z}))]$  to further enhance the reasoning ability of LLMs, where  $\phi$  is some monotonic transform such as logarithm  $(\log(\cdot))$ . Further, we could also optimize the parameters of LLMs to enhance the reasoning abilities of existing LLMs during the training, so we can obtain LLMs with better reasoning abilities with the same inference time budget. In the following sections, we introduce the idea of optimizing LLMs to improve reasoning abilities without external feedback, by proposing a principled framework containing the surrogate objective.

#### 4 OPTIMIZING THE REASONING PROCESS

In this section, we describe how to optimize the reasoning rationale without external feedback. Specifically, we introduce the objective for optimizing the reasoning rationale in Section 4.1 from a variational perspective of LLM training; we derive the gradient estimation for the new objective in Section 4.2, and discuss the sampling procedure together with reward shaping in Section 4.3. We summarize the proposed algorithm, <u>LaTent Reasoning Optimization</u> (LaTRO) in Algorithm 1, and illustrate the overall procedure in Figure 1.

#### <span id="page-3-1"></span>4.1 LATENT REASONING OPTIMIZATION: A VARIATIONAL APPROACH

Suppose we are given a golden dataset  $\mathcal{D}_{Gold} := \{(\boldsymbol{x}_i, \boldsymbol{y}_i)\}_{i=1}^N$  consisting of N query and answer pairs, where  $(\boldsymbol{x}, \boldsymbol{y})$  denotes the query and the answer respectively. A standard finetuning procedure to fit the LLM  $\pi_{\theta}$  to the dataset  $\mathcal{D}_{Gold}$  is by likelihood maximization:

<span id="page-3-2"></span>
$$\max_{\theta} \mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}) \sim \mathcal{D}_{Gold}} \left[ \log \pi_{\theta}(\boldsymbol{y} \mid \boldsymbol{x}) \right] , \tag{1}$$

where  $\theta$  are the parameters of the LLM  $\pi_{\theta}$  to optimize. Based on the discussion in Section 3, it is more feasible to optimize  $\pi_{\theta}$  with additional reasoning rationale path z, compared with standard finetuning objective in Equation (1). Hence, we can introduce another "reasoner" q(z|z) to sample the latent reasoning rationales that can help the optimization procedure of  $\pi_{\theta}$ . This is achievable by

optimizing the following lower bound:

$$\log \pi_{\theta}(\boldsymbol{y}|\boldsymbol{x}) = \log \int \pi_{\theta}(\boldsymbol{y} \mid \boldsymbol{x} \oplus \boldsymbol{z}) \pi_{0}(\boldsymbol{z} \mid \boldsymbol{x}) d\boldsymbol{z}$$

$$= \log \int \pi_{\theta}(\boldsymbol{y} \mid \boldsymbol{x} \oplus \boldsymbol{z}) \frac{q(\boldsymbol{z}|\boldsymbol{x})}{q(\boldsymbol{z}|\boldsymbol{x})} \pi_{0}(\boldsymbol{z}|\boldsymbol{x}) d\boldsymbol{z}$$

$$\geq \max_{q(\boldsymbol{z}|\boldsymbol{x})} \mathbb{E}_{q(\boldsymbol{z}|\boldsymbol{x})} \left[ \log \pi_{\theta}(\boldsymbol{y}|\boldsymbol{x} \oplus \boldsymbol{z}) \right] - D_{\mathrm{KL}} [q(\boldsymbol{z}|\boldsymbol{x})||\pi_{0}(\boldsymbol{z}|\boldsymbol{x})], \qquad (2)$$

where  $\pi_0$  is a prior reference LLM that regularizes the "reasoner"  $q(\boldsymbol{z}|\boldsymbol{x})$ , and the lower bound is achieved via Jensen's inequality (Higgins et al., 2017). Based on the literature of variational Bayes (Kingma, 2013), one can either learn and optimize  $q(\boldsymbol{z}|\boldsymbol{x})$  via variational Expectation Maximization (EM) (Abdolmaleki et al., 2018; Liu et al., 2022), or introduce another parameterized LLM  $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$  and optimize  $\phi$  to amortize the cost. Additionally, from the discussion in Section 3, we know  $\pi_{\theta}$  itself can also serve as a naive "reasoner", since  $\pi_{\theta}$  is an autoregressive LLM.

To simplify the learning procedure, we propose to use  $\pi_{\theta}$  as the "reasoner"  $q(\boldsymbol{z}|\boldsymbol{x})$ . As a result, we can jointly learn one single LLM  $\pi_{\theta}$ , that is capable of generating good reasoning rationale together with providing correct answers given the query and its own generated reasoning rationale. To be more specific, we can define the learning objective as follows:

<span id="page-4-1"></span>
$$\max_{\theta} J(\theta) := \mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}) \sim \mathcal{D}_{Gold}} \left[ \mathbb{E}_{\boldsymbol{z} \sim \pi_{\theta}(\cdot | \boldsymbol{x})} \left[ \underbrace{\log \pi_{\theta}(\boldsymbol{y} | \boldsymbol{x} \oplus \boldsymbol{z})}_{R_{\theta}(\boldsymbol{z}, \boldsymbol{y}, \boldsymbol{x})} \right] - D_{KL} [\pi_{\theta}(\boldsymbol{z} | \boldsymbol{x}) | | \pi_{0}(\boldsymbol{z} | \boldsymbol{x})] \right], \quad (3)$$

where we specify the reference LLM  $\pi_0$  to be the original  $\pi_\theta$  before the optimization. Furthermore,  $\log \pi_\theta(\boldsymbol{y}|\,\boldsymbol{x}\oplus\boldsymbol{z})$  in Equation (3) can be viewed as the reward function  $R_\theta(\boldsymbol{z},\boldsymbol{y},\boldsymbol{x})$  to evaluate the quality of the rationale  $\boldsymbol{z}$  given the pair  $(\boldsymbol{x},\boldsymbol{y})$ , since the reasoning rationale  $\boldsymbol{z}$  with higher likelihood  $\log \pi_\theta(\boldsymbol{y}|\,\boldsymbol{x}\oplus\boldsymbol{z})$  indicates that it would provide a higher probability for the model to answer the question correctly.

Remark By substituting  $\log \pi_{\theta}(\boldsymbol{y} | \boldsymbol{x} \oplus \boldsymbol{z})$  with  $R_{\theta}(\boldsymbol{z}, \boldsymbol{y}, \boldsymbol{x})$ , Equation (3) exactly recovers the standard optimization objective defined in offline RL (Levine et al., 2020), RLHF (Ouyang et al., 2022; Rafailov et al., 2024) literature. Though Equation (3) unifies the learning procedure of the "reasoner"  $\pi_{\theta}(\boldsymbol{z} | \boldsymbol{x})$  and the "reward" function  $R_{\theta}(\boldsymbol{z}, \boldsymbol{y}, \boldsymbol{x}) := \log \pi_{\theta}(\boldsymbol{y} | \boldsymbol{x} \oplus \boldsymbol{z})$ , we can break down these two procedures to analyze them separately. When we fix  $R_{\theta}(\boldsymbol{z}, \boldsymbol{y}, \boldsymbol{x})$  and optimize the "reasoner"  $\pi_{\theta}(\boldsymbol{z} | \boldsymbol{x})$ , the procedure can be interpreted as *self-improvement* learning, where we improve  $\pi_{\theta}(\boldsymbol{z} | \boldsymbol{x})$  on self-generated synthetic reasoning rationale. When we fix  $\pi_{\theta}(\boldsymbol{z} | \boldsymbol{x})$  and optimize  $R_{\theta}(\boldsymbol{z}, \boldsymbol{y}, \boldsymbol{x})$ , the procedure can be interpreted as *self-reward* learning, where we learn the self-reward function  $\log \pi_{\theta}(\boldsymbol{y} | \boldsymbol{x} \oplus \boldsymbol{z})$ . The procedure can also be considered finetuning optimization given the learned reasoning rationale and query. Fortunately, we can naturally enjoy the benefits of these two self-learning procedures with the new reasoning finetuning objective.

#### <span id="page-4-0"></span>4.2 Gradient estimation for LaTRO

From previous RL literature, we know that estimating  $\nabla_{\theta}J(\theta)$  in Equation (3) involves the use of policy gradient methods, which usually suffers from high variances with the naive REINFORCE estimators (Williams, 1992). Inspired by the recent work on policy gradient for LLMs (Ahmadian et al., 2024), we also leverage the REINFORCE Leave-One-Out (RLOO) (Kool et al., 2019) to optimize the "reasoner"  $\pi_{\theta}(z|x)$ , where we can achieve lower variances of gradient estimation by sampling multiple rationales. We summarize the empirical gradient estimation for solving LaTRO in Proposition 2.

<span id="page-4-2"></span>**Proposition 2.** (LaTRO Gradient Estimation) Suppose we are given a set of training data  $\mathcal{D}_{Gold} := \{ \boldsymbol{x}_i, \boldsymbol{y}_i \}_{i=1}^N$ , we sample K i.i.d reasoning rationales  $\boldsymbol{z}_1^{(i)}, \boldsymbol{z}_2^{(i)}, \dots, \boldsymbol{z}_K^{(i)} \sim \pi_{\theta}(\cdot | \boldsymbol{x}_i)$  for each query and answer pair  $(\boldsymbol{x}_i, \boldsymbol{y}_i)$ . The empirical gradient estimator for  $\nabla_{\theta} J(\theta)$  is expressed as

<span id="page-4-3"></span>
$$\nabla_{\theta} \widehat{J}(\theta) := \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} \left( \nabla_{\theta} \log \pi_{\theta}(\boldsymbol{z}_{k}^{(i)} \mid \boldsymbol{x}_{i}) \cdot A_{k}^{(i)} + \nabla_{\theta} \log \pi_{\theta}(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i} \oplus \boldsymbol{z}_{k}^{(i)}) \right), \tag{4}$$

$$\textit{with } A_k^{(i)} = r(\boldsymbol{z}_k^{(i)}) - \frac{1}{K-1} \sum_{i \neq k}^K r(\boldsymbol{z}_j^{(i)}) \,, \\ r(\boldsymbol{z}_k^{(i)}) := \log \pi_{\theta}(\boldsymbol{y}_i \mid \boldsymbol{x}_i \oplus \boldsymbol{z}_k^{(i)}) - \beta \log \frac{\pi_{\theta}(\boldsymbol{z}_k^{(i)} \mid \boldsymbol{x}_i)}{\pi_0(\boldsymbol{z}_k^{(i)} \mid \boldsymbol{x}_i)} \,, \\$$

#### Algorithm 1: <u>LaT</u>ent <u>Reasoning Optimization</u> (LaTRO)

<span id="page-5-1"></span>**Input:** Language model  $\pi_{\theta}$ , learning rate  $\eta$ , KL penalty factor  $\beta$ , MC sample size K, maximum generation length L, sample temperature T, number of epochs M, training dataset  $\mathcal{D}_{Gold}$ . **Output:** An optimized language model  $\pi_{\theta}$ .

where  $\beta \geq 0$  is the coefficient to control the KL penalty. The proof can be found in Appendix A.1.

The first gradient term in Equation (4) serves as policy gradient to improve the ability of the LLM  $\pi_{\theta}$  to generate high-quality reasoning rationales, and  $\log \pi_{\theta}(\boldsymbol{y}|\boldsymbol{x} \oplus \boldsymbol{z})$  can be viewed as the evaluator for reasoning rationale, which is further used to calculate the advantages. The second gradient term in Equation (4), which is the gradient of supervised finetuning loss, essentially helps the LLM  $\pi_{\theta}$  to leverage the reasoning rationales to produce correct answers.

#### <span id="page-5-0"></span>4.3 PRACTICAL CONSIDERATIONS

To reduce computation overhead and better control the sampling of reasoning rationales during training, we limit their maximum token length to L. The rationale ends either at the EOS token or at the start of a predefined answer template (e.g., "The answer is"). We then use the truncated rationale z, along with the query z and the answer z, for further computation.

We also encourage the LLM to finish its reasoning process with L tokens. Inspired by the implementation of the RLOO trainer in the TRL library (von Werra et al., 2020), we introduce a constant penalty for rationales truncated by the maximum token length L. This penalty encourages the generation of rationales that fit within the specified token limit.

#### 5 EXPERIMENTS

#### 5.1 SETUP

We evaluate the performance of the proposed method across two datasets: a mathematical reasoning dataset (GSM8K, Cobbe et al. (2021b)) and a logical reasoning dataset (ARC-Challenge, Talmor et al. (2019)). The sizes of the datasets are listed in Table 1.

**Training.** For each dataset, we fine-tune three base models: Phi-3.5-mini-instruct (Abdin et al., 2024), Mistral-7B-Instruct-v0.3 (Jiang et al., 2023), and Meta-Llama-3.1-8B-Instruct (Dubey et al., 2024), abbreviated as Phi-3.5, Mistral-7B, and Llama-3.1-8B, respectively. We provide two baseline comparisons: the base model and the super-

Table 1: Size of the datasets

<span id="page-5-2"></span>

| Name          | Training | Evaluation |
|---------------|----------|------------|
| GSM8K         | 7473     | 1319       |
| ARC-Challenge | 1119     | 1172       |

vised fine-tuned (SFT) model. For GSM8K, LaTRO fine-tuning excludes golden rationales from the solutions in the training set, while the SFT model is trained using golden rationales. For ARC-Challenge, as suggested in (Zheng et al., 2024), the model is trained to generate answers to the text of

multiple-choice questions rather than selecting labels. Since no golden rationales are available for ARC-Challenge, the SFT model is trained to directly generate answers.

Evaluation. For GSM8K, we evaluate all models with CoT prompting, and for ARC-Challenge, we evaluate the SFT baseline with direct answer generation, while the base model and the LaTRO fine-tuned model with CoT prompting. All evaluations are conducted with zero-shot prompts. We report both greedy decoding (GD) results and self-consistency (with temperature T = 1) results. We choose a self-consistency sample size k = 8 (maj@8) in Table [2](#page-7-0) after observing that more than 8 samples did not bring further performance improvement (see Figure [3](#page-8-0) (b) for details).

Implementation. LaTRO is implemented on the high level as in Algorithm [1,](#page-5-1) with additional engineering techniques as discussed in section [4.3.](#page-5-0) LaTRO is implemented using the widely recognized transformers [\(Wolf et al.,](#page-13-12) [2020\)](#page-13-12) and TRL [\(von Werra et al.,](#page-13-10) [2020\)](#page-13-10) libraries, with PyTorch [\(Ansel](#page-10-6) [et al.,](#page-10-6) [2024\)](#page-10-6) as backend. DeepSpeed ZeRO [\(Rasley et al.,](#page-12-11) [2020\)](#page-12-11) is used in stage 3, along with Flash Attention 2 [\(Dao et al.,](#page-10-7) [2022\)](#page-10-7) to enhance training efficiency. The models were trained on a machine equipped with 8xH100 80GB GPUs, using bfloat16 precision.

Hyperparameters. AdamW optimizer with a learning rate of 5 × 10<sup>−</sup><sup>7</sup> , no warm-up steps, and a linear decay strategy is used. The Monte Carlo (MC) sample size K = 16 and the batch size of the data loader 3 are predetermined, resulting in an effective batch size of 48. Gradient accumulation steps and training batch size are subsequently adjusted to prevent out-of-memory errors during training. A temperature of T = 1 is used for MC sampling, and a penalty factor γ = 2 is applied for incomplete rationales. The KL penalty is set at β = 0.05 for GSM8K and 0.25 for ARC-Challenge. Except for the results presented in Section [5.3,](#page-7-1) the maximum generation length is maintained at L = 500. We train all models up to six epochs for GSM8K, and 12 epochs for ARC-Challenge. The checkpoint with best test accuracy is chosen.

For the SFT baseline experiments, we use a batch size of 32 and adjust the learning rate to ensure that the evaluation loss decreases and finally converges. All SFT baselines are trained for a maximum of 12 epochs. The checkpoint with the best test accuracy is selected.

In addition to the main quantitative results, we conduct ablation studies on two factors: 1. The maximum generation length L, where we study the effects of tuning L in both training and inference times; 2. The self-consistency samples k, where we explore to what extent LaTRO can still benefit from inference-time scaling.

The main quantitative results, qualitative analysis of sample responses, and results of the ablation study are presented in Sections [5.2](#page-6-0) to [5.4,](#page-9-3) respectively. Additional details on our prompt templates and more samples can be found in Appendices [B](#page-15-1) and [C.](#page-16-0)

### <span id="page-6-0"></span>5.2 RESULTS

In this subsection, we present evaluation results that demonstrate how effectively LaTRO enhances the reasoning abilities of LLMs on downstream datasets. The detailed results are provided inTable [2.](#page-7-0)

For the GSM8K dataset, LaTRO fine-tuned models outperform all base models by up to 19.5% (Mistral-7B, 47.8% → 67.3%) and show an average improvement of 12.5% across the three models examined with greedy decoding. The greatest improvement margin is observed for Mistral-7B, while the smallest is seen for Llama-3.1-8B, consistent with our initial findings in Figure [2,](#page-2-1) where Mistral-7B exhibited the lowest log probability for directly answering questions and Llama-3.1- 8B exhibited the highest. With self-consistency, the improvements are by up to 16.5% (Phi-3.5, 74.0% → 90.5%) and the average improvement is 13.1%. Furthermore, LaTRO models demonstrate superior performance relative to SFT baselines, with an average improvement of 9.6% for greedy decoding and 13.2% for self-consistency. It is worth noting that for the SFT baseline of Llama-3.1-8B, overfitting on the test set is still observed after tuning the learning rate.

For ARC-Challenge, LaTRO fine-tuned models still outperform the baselines, though with a smaller margin. When using greedy decoding, the improvements over the base models are up to 1.6% with an average increase of 1%. We see more increment with self-consistency, where the improvement margins are on average 2.4%. Comparing to SFT baslines, we find that all three models are very sensitive when fine-tuning to directly generate the answer of ARC-Challenge questions. They perform

<span id="page-7-0"></span>Table 2: Zero-shot accuracy (%) comparison between LaTRO and the baselines on GSM8K and ARC-Challenge datasets. The models are fine-tuned on corresponding training datasets. The base model are marked with "N/A" in the training method. GD stands for greedy decoding at inference time and maj@8 stands for self-consistency with 8 samples. The models are evaluated by default using CoT, except that † indicates the direct answer generation is applied during evaluation.

| Base Model   | Training Method | Inference Method | GSM8K | ARC-Challenge |
|--------------|-----------------|------------------|-------|---------------|
| Phi-3.5      | N/A             | GD               | 72.9  | 85.1          |
|              |                 | maj@8            | 74.0  | 86.0          |
|              | SFT             | GD               | 75.8  | 81.0†         |
|              |                 | maj@8            | 77.1  | 80.5†         |
|              | LaTRO           | GD               | 87.6  | 86.4          |
|              |                 | maj@8            | 90.5  | 87.5          |
| Mistral-7B   | N/A             | GD               | 47.8  | 74.1          |
|              |                 | maj@8            | 58.2  | 74.1          |
|              | SFT             | GD               | 57.2  | 70.0†         |
|              |                 | maj@8            | 59.9  | 70.6†         |
|              | LaTRO           | GD               | 67.3  | 74.3          |
|              |                 | maj@8            | 73.8  | 78.9          |
| Llama-3.1-8B | N/A             | GD               | 76.8  | 81.4          |
|              |                 | maj@8            | 79.7  | 84.4          |
|              | SFT             | GD               | 73.2  | 77.0†         |
|              |                 | maj@8            | 74.7  | 76.4†         |
|              | LaTRO           | GD               | 80.1  | 83.0          |
|              |                 | maj@8            | 87.0  | 85.3          |

even inferior to the unoptimized base models. When using greedy decoding, the improvements of LaTRO fine-tuned models over the SFT baselines are on an average of 5.2%, and by up to 6% (Llama-3.1-8B). In the case of self-consistency, LaTRO performs better than the base models by an average of 2.4%, and surpasses the SFT models by an average of 8.1%. On the less surprising results compared to GSM8K, we conjecture that for ARC-Challenge, the models are already good at producing the answer either directly or through CoT prompting. Hence, further optimization of the reasoning process did not yield significant improvement.

#### <span id="page-7-1"></span>5.3 ABLATION STUDY

In this subsection, we present our ablation study on the effect of different parameters in LaTRO. For consistency, we fix the base model to Phi-3.5 and the dataset to GSM8K throughout the ablation experiments.

How many tokens are enough? [Liu et al.](#page-12-12) [\(2024\)](#page-12-12) demonstrated that when the input length is n, a transformer model with a hidden size of O(log n) can solve problems equivalent to Boolean circuits of size m, using m CoT steps. However, the empirical determination of sufficient CoT tokens for optimal performance remains underexplored. In this section, we report zero-shot accuracy with generation length L ranging from 200 to 1000 tokens at inference time. Additionally, a Phi-3.5 model is fine-tuned with L = 200 for comparison. We distinguish two LaTRO fine-tuned models, referred to as LaTRO and LaTRO200. As shown in Figure [3\(](#page-8-0)a) accuracy gains plateau when L ≥ 500, suggesting 500 tokens might suffice for grade school math problems. In contrast, limiting L to 200 reduces accuracy, unless the model is trained accordingly. Interestingly, LaTRO significantly improves performance under this constraint by training the model to generate more concise rationales.

<span id="page-8-0"></span>> **[图片提取文字 (无描述)]:**
> 0.9 0.90 0.8 0.7 0.850.6 0.80 0.5 LaTRO<sub>200</sub> 0.4 0.75LaTRO LaTRO SFT 0.3 Base Model Base Model 0.70 200 300 400 500 1000 16 (b) Zero-shot maj@k accuracy with different k. (a) Zero-shot accuracy with different L.
![](_page_8_Figure_0.jpeg)

Figure 3: Ablation study results on GSM8K with base model Phi-3.5. In (a), the x-axis represents various maximum token length L of reasoning rationales, y-axis is the accuracy, and the plot shows the zero-shot performance v.s. various maximum token lengths for different methods. In (b), the x-axis represents the # of sampled reasoning rationales, the y-axis is the accuracy, and the plot shows the zero-shot performance v.s. the # of reasoning rationales used in the majority vote.

This finding implies LaTRO can also train LLMs to condense rationales for limited inference-time compute. A sample from LaTRO<sup>200</sup> is provided in Appendix [C](#page-16-0) for qualitative analysis.

Is inference-time scaling still beneficial? While LaTRO shifts much of the reasoning computation to training time, it can still benefit from the inference-time scaling. To explore this, We evaluate the models again using self-consistency with the number of samples k ranging from 1 to 32. As shown in Figure [3](#page-8-0) (b), with k = 8 rationales and majority voting, LaTRO achieves 90% accuracy on GSM8K. This indicates that LaTRO equipped with self-consistency can further enhance performance, even with LaTRO's focus on training-time optimization.

<span id="page-8-1"></span>Question: Frankie and Binkie went bowling together. Frankie's score was 15 better more than twice as high as Binkie's. If Binkie bowled a score of 90, what was Frankie's score? Let's think step by step.

Response (Base model): Let's break this problem down into steps:

- 1. We know that the difference between Frankie's score (F) and Binkie's score (B) is 15. So we can express Frankie's score as F = B + 15
- 2. We also know that Frankie's score is twice as high as Binkie's score. So we can write another equation: F = 2B
- 3. Now, we have two equations with two unknowns. Let's solve for Frankie's score (F) by setting the two equations equal to each other: - 2B = B + 15 - Multiply both sides by 1 to get rid of the fraction: 2B - B = 15 - Simplify: B = 15 (since subtracting B from 2B results in 15)
- 4. Now that we know Binkie's score, we can find Frankie's score using either of the equations we created earlier. Let's use the first one (F = B + 15): - F = 15 + 15 - Frankie's score is 30 more than Binkie's score, which is 15 in this case. So Frankie's score is 15 + 15 = 30 + 15 = 45

The final answer is that Frankie's score was 45.

Reponse (SFT): Twice as many as Binkie's 90 score is equal to 2\*90= «2\*90=180»180. Therefore, 180-(180-15)=165 was the number Frankie bowled. The answer is 165

Response (LaTRO): Here's how to solve the problem: 1. We know that Binkie's score is 90. 2. The problem states that Frankie's score was 15 better more than twice as high as Binkie's score. This means that Frankie's score is 15 + (2 \* 90) = 15 + 180 = 195.

Figure 4: Sample responses of a GSM8K question from Mistral-7B models, the errors are highlighted in red.

#### <span id="page-9-3"></span>5.4 CASE STUDY

We take a closer look at the responses generated by the LaTRO fine-tuned models. We select a question from GSM8K and compare the responses from the base, the SFT model, and the LaTRO finetuned model. We choose the set of responses from the Mistral-7B models that we evaluated. As can be seen in Figure [4,](#page-8-1) the base model not only generates a lengthy response, it also makes a logical mistake at the first step, where the correct equation to establish here is "F = 2B + 15". The SFT model simplifies the answer and makes the first step correct. However, in the second step it first makes a wrong equation, then makes an arithmetic error when evaluating this equation. Further, LaTRO can give a concise and correct answer. We include more sample responses in Appendix [C.](#page-16-0)

