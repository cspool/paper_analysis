#### ALPHA ONE: Reasoning Models Thinking Slow and Fast at Test Time

Junyu Zhang † Runpei Dong † Han Wang Xuying Ning Haoran Geng Peihao Li Xialin He Yutong Bai Jitendra Malik Saurabh Gupta Huan Zhang

University of Illinois Urbana-Champaign UC Berkeley †Equal contributions. Correspondence: {junyuz6, runpeid2, huanz}@illinois.edu

## Abstract

This paper presents ALPHA ONE ( α1), a universal framework for modulating reasoning progress in large reasoning models (LRMs) at test time. α1 first introduces α moment, which represents the scaled thinking phase with a universal parameter α. Within this scaled preα moment phase, it dynamically schedules slow thinking transitions by modeling the insertion of reasoning transition tokens as a Bernoulli stochastic process. After the α moment, α1 deterministically terminates slow thinking with the end-of-thinking token, thereby fostering fast reasoning and efficient answer generation. This approach unifies and generalizes existing monotonic scaling methods by enabling flexible and dense slow-to-fast reasoning modulation. Extensive empirical studies on various challenging benchmarks across mathematical, coding, and scientific domains demonstrate α1's superior reasoning capability and efficiency. Project page: [https://alphaone-project.github.io/](alphaone-project.github.io)

## 1 Introduction

"The most effortful forms of slow thinking are those that require you to think fast."

> *Thinking, Fast and Slow [\(Kahneman](#page-10-0) , [2011](#page-10-0) )*

Large Reasoning Models (LRMs) such as OpenAI o1 [\(Jaech et al.](#page-10-1) , [2024\)](#page-10-1) and DeepSeek-R1 [\(DeepSeek-AI et al.,](#page-9-0) [2025\)](#page-9-0) have demonstrated unprecedented progress in approaching humanlike system-2 reasoning capabilities, enabling *slow thinking*—slowing down *reasoning progress* [1](#page-0-0) at test time—for solving complex reasoning problems that require high-order cognitive processing. These advanced models are trained to utilize slow thinking via reinforcement learning, enabling LRMs

<span id="page-0-1"></span>> **[图片提取文字 (无描述)]:**
> End Less Reasoning Reasoning Progress Vanilla  $\alpha 1$ /Fast reasoning 🐥 </think> "Wait," Slow reasoning Start α Moment Reasoning Time
![](_page_0_Figure_11.jpeg)

Figure 1: Conceptual illustration of reasoning modulation strategies. Our α1 employs a *slow-to-fast* reasoning schedule controlled by α. α1 scales more efficiently than *monotonously increasing* method s1 (yellow) and generally outperforms *monotonously decreasing* (purple) approaches.

to slow down reasoning progress automatically. Is such automatic slowing down of reasoning progress determined by LRMs sufficiently reliable? According to [Kahneman](#page-10-0) [\(2011\)](#page-10-0), humans typically think fast first and activate slow thinking when running into difficulty, through a conscious control of system-1-to-2 reasoning transitioning, resulting in overall comprehensive but efficient reasoning. While similar to human systems and interesting results have been observed, a lot of works have pointed out that the LRMs themselves are prone to *overthinking* [\(Chen et al.](#page-9-1) , [2024b](#page-9-1) ; [Sui et al.](#page-11-0) , [2025](#page-11-0) ; [Pu et al.](#page-11-1) , [2025](#page-11-1) ; [Yang et al.](#page-12-0) , [2025c\)](#page-12-0) or *underthinking* [\(Su et al.,](#page-11-2) [2025;](#page-11-2) [Yang et al.,](#page-12-1) [2025d;](#page-12-1) [Wang et al.,](#page-12-2) [2025\)](#page-12-2). This is because of the inability of LRMs to find the optimal human-like system-1-to-2 reasoning transitioning and limited reasoning capabilities, leading to unsatisfactory reasoning performance.

To overcome this limitation, existing works scale LRMs at test time in mainly two ways. i) *paral-*

<span id="page-0-0"></span><sup>1</sup>Consider reasoning progress as a metric ranging from 0 to 1, indicating the start and the end of reasoning, respectively. It increases slowly or fast at the pace of thinking. See Fig. [1](#page-0-1) and Section [2](#page-2-0) for a more illustrative and detailed explanation.

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> Vanilla Ouestion Let *K* be the number of sequences Okay, so I need to figure out ... \n\n First, let me parse the problem. ... Then, ... \n\n\Wait, but that can't be, because... </think> ... So, the  $A_1, A_2, \dots, A_n$ .... What is the ----> Wait Frequency: 23 remainder when K is divided by 10? remainder when K is divided by 10 is 0.  $\alpha \leftarrow 1.8$  $\alpha \leftarrow 1.4$ Alpha0ne  $\alpha \leftarrow$  $p_{\mathsf{wait}}$  Wait,...  $p_{\mathsf{wait}}$ Wait, actually,  $p_{\textsf{wait}}$ Wait, hold on. Wait, \n</think> Okay, so I need reading the The number K is The problem to figure out ... ...\n\n problem ... \n\n the sum of ...\n\n ...\n\n  $\ln \ln$  $\boldsymbol{p}_{\textsf{wait}}$ > Wait, ...  $\boldsymbol{p}_{\textsf{wait}}$ ...\n\n First, let me  $\alpha$  Moment parse the Wait, ... \n\n problem...\n\n Therefore, the remainder is 5. This seems  $p_{\tt wait}$   $\geq \tt Wait ...$ similar to the concept of Wait Frequency: Generation ...\n\n building ... \n\n 47 ( $\alpha = 1.4$ )  $\leftarrow$  ---  $p_{\textsf{wait}}$  --->  $\alpha = 1.4$ Early Slow Reasoning Less Reasoning More Reasoning Later Fast Reasoning
![](_page_1_Figure_0.jpeg)

Figure 2: Overview of ALPHAONE (α1). Here represents α moment (Section [3.1\)](#page-3-0). α1 applies dense reasoning modulation via a user-defined slow thinking scheduling in pre-α moment. In addition, α1 utilizes a post-α moment modulation by replacing slow thinking transitioning tokens "wait" to "</think>", which fosters fast thinking. Specifically, α determines when the slow-to-fast reasoning transition occurs. For example, reducing α from 1.4 to 1.0 shifts the α moment earlier, resulting in shorter slow reasoning phase and accelerating the annealing of pwait.

*lel scaling*: this line of research follows a *Best of* N strategy and typically samples N times and outputs the best answer using criteria such as self-consistency [\(Wang et al.,](#page-12-3) [2023;](#page-12-3) [Wan et al.,](#page-11-3) [2024a;](#page-11-3) [Zhou et al.,](#page-13-0) [2025;](#page-13-0) [Ma et al.,](#page-10-2) [2025\)](#page-10-2) and perplexity [\(Fang et al.,](#page-9-2) [2025\)](#page-9-2). ii) *sequential scaling*: this family of approaches addresses the overthinking/underthinking issues via early reasoning stopping [\(Fu et al.,](#page-9-3) [2025;](#page-9-3) [Yang et al.,](#page-12-4) [2025a;](#page-12-4) [Xu](#page-12-5) [et al.,](#page-12-5) [2025\)](#page-12-5) and promoting for reinforcing reasoning [\(Muennighoff et al.,](#page-11-4) [2025;](#page-11-4) [Wang et al.,](#page-12-2) [2025\)](#page-12-2), respectively. For example, [Xu et al.](#page-12-5) [\(2025\)](#page-12-5) proposes Chain of Draft, prompting LRMs to think fast strictly within 5 words to significantly reduce overthinking. s1 [\(Muennighoff et al.,](#page-11-4) [2025\)](#page-11-4) proposes to foster reasoning continuously via appending a slow-reasoning transition token "wait" multiple times when LRMs are about to end. However, it is unclear if such monotonous reasoning increment or reduction is optimal, and the appropriate moment for slow thinking transitioning is still underexplored. Hence, instead of test-time scaling with an automatic slowing down by LRMs themselves or simply increasing or reducing slow thinking, we are interested in finding: *Can we modulate reasoning progress universally, and develop a better slow thinking transitioning strategy with it?*

To answer this question, we present ALPHAONE (α1), which efficiently scales LRMs at test time through a *universal* reasoning progress modulation. We introduce *alpha moment*, parameterized by α ≥ 0, where the thinking process is scaled by α times throughout the whole generation sequence. To be specific, within a certain token length scaled by α, we stochastically append the reasoning transition token "wait" after structural delimiters "\n\n" under Bernoulli(pwait), inspired by the observation that these two frequently co-exist [\(Yang et al.,](#page-12-0) [2025c\)](#page-12-0). Here, pwait is scheduled to change over time to *activate* slow thinking. For example, a simple linear annealing over time indicates a slow thinking first, then fast thinking strategy.

However, we observe that amplifying slow thinking enables LRMs to sustain it automatically. Thus, when pwait reaches 0, we replace "wait" with "</think>" to deactivate slow thinking and switch to fast reasoning. In this fashion, α1 unifies prior methods like s1 [\(Muennighoff et al.,](#page-11-4) [2025\)](#page-11-4), where α1 reduces to s1 if pwait is 1 or 0 at the end of a reasoning segment within a certain reasoning token length. However, different from these works that only explore sparse slow reasoning modulation, α1 modulates reasoning continuously, supporting both *sparse* and *dense* modulation strategies.

**Takeaways** We present some insightful findings from evaluating three different  $\alpha 1$  LRMs, ranging from 1.5B to 32B across six reasoning benchmarks, including math, code generation, and scientific problem reasoning: i) Slow thinking first, then fast thinking, leads to better LRM reasoning. Surprisingly, this differs from humans who commonly think fast, followed by slow thinking (Kahneman, 2011), emphasizing the requirement of dedicated test-time scaling strategies for LRMs. ii) Slow thinking can bring efficient test-time scaling. While slow thinking slows down reasoning, the overall token length is significantly reduced with  $\alpha 1$ , inducing more informative reasoning progress brought by slow thinking. iii) Slow thinking transitioning in high frequency is helpful. Interestingly, we find that  $\alpha 1$  appending "wait" significantly more (e.g., over  $2 \times$  more than s1) achieves much better results.

### <span id="page-2-0"></span>2 Background & Problem Statement

Revisiting Reasoning Models Following the success of OpenAI's o1 model (Jaech et al., 2024), modern LRMs solve complex reasoning problems via a *thinking-then-answering* paradigm (DeepSeek-AI et al., 2025; Qwen Team, 2025; Huang et al., 2024b). Generally, a special end-of-thinking token "

end-of-thinking moment, transitioning from the thinking phase to the answering phase. During the thinking process, LRMs automatically transit between slow thinking and fast thinking, utilizing self-reflection as chain of thoughts (Wei et al., 2022).

Slow Thinking Transitioning To leverage human-like system-2 slow thinking that helps solve complex reasoning problems, o1-style LRMs automatically transit between fast thinking and slow thinking. To be specific, during the thinking process, LRMs frequently generate slow thinking transitioning tokens such as "wait", "hmm", and "alternatively", etc. Once these tokens are generated, LRMs slow down reasoning, where previous reasoning chains are self-reflected and corrected immediately. Hence, reasoning following the transitioning token can be viewed as slow thinking, while the rest is generally fast thinking.

**Reasoning Progress** Let the overall answer sequence generation process be a *reasoning progress*  $\mathcal{P} \in [0,1]$ , where 0 and 1 indicate the start and the end of reasoning, respectively. Notably, reasoning progress represents the overall problem-solving

progress instead of the number of generated tokens, where a reasoning progress closer to 1 represents the reasoning chain is more informative. For example, the reasoning progress can be closer to 1 while generating fewer tokens, indicating more efficient reasoning. However, it is intractable to measure the exact progress obtained. Hence, we define the reasoning progress following a reasoning velocity assumption. Given the total time t=T>0 spent on generating the whole sequence, the reasoning velocity at timestep t,  $\mathcal{V}_t$  is defined as  $\frac{d\mathcal{P}}{dt}$ , where dt is the infinitesimal of time. We assume:

**Assumption 1.** The reasoning velocity of slow thinking is smaller than that of fast thinking.

See Fig. 1, different reasoning strategies result in different reasoning progress achieved over time.

## 2.1 Reasoning Progress Modulation: A Universal View of Test-Time Scaling

There are mainly two components that must be modulated: i) Thinking phase budget. As discussed before, o1-like LRMs follow a "think-then-answer" paradigm. Therefore, modulating reasoning via scaling up or down the thinking phase budget is required. ii) Slow thinking scheduling. Within the thinking phase, the transition to slow thinking should also be modulated, thus increasing or reducing slow thinking according to a certain plan specified by users (e.g., slow thinking first, and then fast thinking). With user-defined scheduling, the modulation of slow thinking transitions vary arbitrarily, ranging from sparse modulation—where little is adjusted—to dense modulation, where adjustments are frequent and extensive.

Based on the above analysis, we establish a unified perspective on test-time scaling and identify key limitations in existing approaches—namely, their failure to consider both reasoning schedule and overall thinking budget jointly. For instance, s1 modulates reasoning by sparsely increasing slow thinking (i.e., adding two wait tokens), but overlooks broader thinking budget adjustments (Muennighoff et al., 2025). Conversely, Chain-of-Draft (CoD) reduces the thinking budget while neglecting the scheduling of slow thinking (Xu et al., 2025). As a result, while LRMs are indirectly guided to reason more or less—sometimes achieving deeper reasoning or pruning unproductive thoughts—we instead aim to explicitly and universally modulate the reasoning process by jointly considering both components, as introduced next.

## 3 ALPHAONE

We introduce ALPHAONE (α1), a universal reasoning progress modulation framework for test-time scaling of LRMs, which is illustrated in Fig. [2.](#page-1-0) In the following, we first introduce α moment in Section [3.1,](#page-3-0) a moment that the thinking phase budget is scaled at least α×. In Section [3.2](#page-3-1) and Section [3.3,](#page-3-2) we detail how we modulate slow thinking scheduling pre-α moment and modulating fast thinking encouragement post-α moment, respectively.

## <span id="page-3-0"></span>3.1 α Moment for Universal Modulation

To modulate the thinking phase budget, we propose to scale the thinking phase by at least α×, where α > 1 is a universal modulating parameter. Formally, given the average thinking phase token length Nthink > 0 generated by an LRM, we scale the thinking phase token length to αN, where the moment when the generated token length reaches αN is dubbed as "α *moment*". In addition to scaling the thinking phase, we modulate the thinking phase via slow thinking scheduling before the α moment, thus achieving both controllable and scalable thinking. Note that α moment does not represent the new thinking phase transitioning moment, because the thinking phase typically continues after α moment, which we will elaborate later.

# <span id="page-3-1"></span>3.2 Pre-α Moment Modulation

Following previous works [\(Yang et al.,](#page-12-0) [2025c;](#page-12-0) [Muennighoff et al.,](#page-11-4) [2025\)](#page-11-4), we activate slow thinking before α moment via appending "wait" after a frequently co-generated structural delimiters "\n\n". Moreover, the activation of slow thinking is conducted following a user-specified scheduling plan, such as slow thinking, then fast thinking.

Stochastic Reasoning Transitioning Our α1 achieves such scheduling by modeling the activation of slow thinking as a Bernoulli stochastic process. Specifically, α1 append "wait" following Bernoulli(pwait). Let t = 0, 1, . . . , T<sup>m</sup> be the timestamps of generated tokens before α moment, where T<sup>m</sup> = αNthink represents the timestamp of α moment. pwait is determined by a user-specified scheduling function S(t),

<span id="page-3-3"></span>
$$\boldsymbol{p}_{\text{wait}} := \mathcal{S}(t), t = 0, 1, \dots, T_m. \tag{1}$$

This scheduling function can be arbitrary functions, such as linear annealing and linear increase. α1 adopts linear annealing, which we find the most effective and efficient (See Section [4.3.1\)](#page-5-0).

## <span id="page-3-2"></span>3.3 Post-α Moment Modulation

While an LRM significantly increases slow thinking through pre-α modulation, this extended thinking phase often exhibits *slow thinking inertia*, making it difficult to transition back to fast thinking. Notably, without post-α moment modulation, the LRM substantially reduces the likelihood of generating "</think>". Furthermore, inserting a few "</think>" tokens does not effectively overcome the inertia, failing to fully restore fast thinking.

Deterministic Reasoning Termination After the α moment, we guide α1 to transition into fast reasoning by disabling further slow thinking. Specifically, any generated slow reasoning transition token "wait" is replaced with "</think>" to explicitly mark the end of the thinking phase, reinforcing a shift to fast thinking before entering the answering phase. This deterministic termination strategy allows α1 to conclude reasoning naturally and consistently, enabling more efficient test-time scaling.

## 4 Experiments

## 4.1 Experimental Setup

Benchmarks To comprehensively evaluate the reasoning capability of LRMs, we conduct systematic evaluations on six benchmarks covering three reasoning categories: i) mathematical reasoning, including AIME 2024 (AIME24) [\(Mathematical](#page-11-6) [Association of America,](#page-11-6) [2024\)](#page-11-6), , AMC23 [\(AI-MO,](#page-8-0) [2024\)](#page-8-0), and Minerva-Math (Minerva) [\(Lewkowycz](#page-10-4) [et al.,](#page-10-4) [2022\)](#page-10-4); ii) code generation, including Live-CodeBench (LiveCode) [\(Jain et al.,](#page-10-5) [2025\)](#page-10-5); iii) scientific problems, including OlympiadBench (Olympiad) [\(He et al.,](#page-10-6) [2024\)](#page-10-6). We report the problem-solving accuracy by average Pass@1 (%), and the average number of generated tokens.

Base Models We use three o1-like open-source LRMs as the base model, including DeepSeek R1 distilled DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B [\(DeepSeek-AI](#page-9-0) [et al.,](#page-9-0) [2025\)](#page-9-0), as well as a recently larger LRM Qwen QwQ 32B [\(Qwen Team,](#page-11-5) [2025\)](#page-11-5).

Implementations Without additional specifications, we use a temperature of 0.6, top-p of 0.95, and the maximum token length is set to 8192. We set α as 1.4, and we obtain the average thinking phase token length generated by an LRM on any benchmark by randomly sampling 10 test questions and averaging the generated token length before benchmarking. See more details in Section [A.](#page-14-0)

<span id="page-4-0"></span>Table 1: Systematic comparison of reasoning results on mathematical, coding, and science reasoning benchmarks with DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, and Qwen QwQ 32B. P@1: Pass@1 (%); #Tk: number of generated tokens; ∆P@1 (%): average Pass@1 result boost over the base model. <sup>∗</sup>For a fair comparison, S1 [\(Muennighoff et al.,](#page-11-4) [2025\)](#page-11-4) directly applies budget forcing at test-time without supervised fine-tuning, which is same as CoD and our α1 that are training-free.

|                               | MATHEMATICAL |      |           |      |          |      | CODING   |      | SCIENCE  |      |          |      |       |
|-------------------------------|--------------|------|-----------|------|----------|------|----------|------|----------|------|----------|------|-------|
| Method                        | AIME24       |      | AMC23     |      | Minerva  |      | MATH500  |      | LiveCode |      | Olympiad |      |       |
|                               | P@1          | #Tk  | P@1       | #Tk  | P@1      | #Tk  | P@1      | #Tk  | P@1      | #Tk  | P@1      | #Tk  | ∆P@1  |
| DeepSeek-R1-Distill-Qwen-1.5B |              |      |           |      |          |      |          |      |          |      |          |      |       |
| BASE                          | 23.3         | 7280 | 57.5      | 5339 | 32.0     | 4935 | 79.2     | 3773 | 17.8     | 6990 | 38.8     | 5999 | N/A   |
| s1∗                           | 26.7+3.4     | 7798 | 57.5+0.0  | 6418 | 31.6-0.4 | 5826 | 78.2-1.0 | 4733 | 17.0-0.8 | 7025 | 38.5-0.3 | 6673 | +0.15 |
| COD                           | 30.0+6.7     | 6994 | 65.0+7.5  | 5415 | 29.0-3.0 | 4005 | 81.4+2.2 | 3136 | 20.3+2.5 | 6657 | 40.6+1.8 | 5651 | +2.95 |
| α1 (Ours)                     | 30.0+6.7     | 5916 | 70.0+12.5 | 4952 | 34.2+2.2 | 4586 | 81.0+1.8 | 3852 | 24.8+7.0 | 5426 | 45.5+6.7 | 4944 | +6.15 |
| DeepSeek-R1-Distill-Qwen-7B   |              |      |           |      |          |      |          |      |          |      |          |      |       |
| BASE                          | 46.7         | 6648 | 82.5      | 4624 | 40.4     | 4191 | 87.6     | 3239 | 43.5     | 5885 | 50.4     | 5385 | N/A   |
| s1∗                           | 46.7+0.0     | 7295 | 80.0-2.5  | 5673 | 42.3+1.9 | 6510 | 92.8+5.2 | 5848 | 44.0+0.5 | 5979 | 54.2+3.8 | 6007 | +1.48 |
| COD                           | 43.3-3.4     | 6078 | 87.5+5.0  | 3594 | 43.4+3.0 | 2142 | 88.8+1.2 | 2094 | 45.0+1.5 | 5593 | 53.5+3.1 | 4520 | +1.73 |
| α1 (Ours)                     | 50.0+3.3     | 6827 | 90.0+7.5  | 4397 | 42.3+1.9 | 4124 | 91.2+3.6 | 4337 | 49.8+6.3 | 5067 | 55.7+5.3 | 4883 | +4.65 |
| Qwen QwQ-32B                  |              |      |           |      |          |      |          |      |          |      |          |      |       |
| BASE                          | 40.0         | 4058 | 77.5      | 2901 | 47.8     | 2199 | 90.2     | 1951 | 67.0     | 5092 | 53.6     | 3230 | N/A   |
| s1∗                           | 43.3+3.3     | 4221 | 77.5+0.0  | 3068 | 46.7-1.1 | 2433 | 90.8+0.6 | 2218 | 66.5-0.5 | 5260 | 55.1+1.5 | 3454 | +0.63 |
| COD                           | 46.7+6.7     | 3959 | 80.0+2.5  | 2400 | 47.4-0.4 | 1464 | 90.6+0.4 | 1421 | 66.8-0.2 | 4984 | 57.2+3.6 | 2844 | +2.10 |
| α1 (Ours)                     | 53.3+13.3    | 3141 | 87.5+10.0 | 2286 | 46.0-1.8 | 1441 | 89.4-0.8 | 1668 | 75.8+8.8 | 5824 | 56.1+2.5 | 2504 | +5.33 |

Baselines We compare our α1 against the vanilla LRM and two training-free, test-time scaling baselines. i) BASE: The original LRM that transitions between slow and fast thinking *automatically*, without any external modulation. ii) S1 [\(Muennighoff](#page-11-4) [et al.,](#page-11-4) [2025\)](#page-11-4): A baseline that enforces a *monotonically increasing* slow thinking pattern by appending approximately two "wait" tokens near the end of the reasoning phase to prolong slow thinking. For a fair comparison, we apply S1 at test time without supervised fine-tuning used in its original implementation. iii) CHAIN OF DRAFT (COD) [\(Xu et al.,](#page-12-5) [2025\)](#page-12-5): A baseline that enforces a *monotonically decreasing* slow thinking pattern by prompting the model to constrain each slow thinking step to no more than five words, thereby sharply reducing the thinking budget.

## 4.2 Main Results

Table [1](#page-4-0) shows the systematic comparison results of our α1 and baseline methods, and we observe: i) α1 consistently yields a higher problem-solving

accuracy than all baseline methods across all models and benchmarks. Notably, compared to the base model, α1 improves the 1.5B LRM by a clear margin of +6.15%, while reducing nearly 14% token length. This demonstrates both the effectiveness and efficiency of α1. ii) Compared to baseline test-time scaling methods, including s1 and CoD, α1 still achieves significantly better results. Specifically, the average accuracy boost over all benchmarks and models of α1 is +3.12% and +4.62% higher than CoD and s1, respectively. iii) Surprisingly, we observe that while α1 modulates reasoning densely without restrictions on reducing the thinking budget (instead, we use α > 1 that increases the thinking budget), the average thinking phase token length generated by α1 is only about +4.4% higher than the monotonically decreasing baseline CoD (4231 *vs.* 4053), which is about +21.0% more efficient than the monotonically increasing baseline s1 (4231 *vs.* 5357). This indicates that α1 achieves more efficient reasoning than baselines, which we provide analysis later.

<span id="page-5-1"></span>> **[图片提取文字 (无描述)]:**
> Linear Increase Exponential Anneal Linear Anneal Vanilla Constant  $p_{\mathsf{wait}}$  $p_{\mathtt{wait}}$  $p_{\mathtt{wait}}$  $p_{\mathsf{wait}}$  $p_{\mathsf{wait}}$  $p_{\mathsf{wait}}$ Generation Generation Generation Generation Generation Generation Dense Reasoning Modulation Sparse Reasoning Modulation
![](_page_5_Figure_0.jpeg)

Figure 3: **Visualization of different scheduling strategies.** We detail the functions in Section 4.3.1. Here  $\bigcirc$  represents  $\alpha$  moment, which we elaborate in Section 3.1, and  $\clubsuit$  denotes the end of the thinking phase.

<span id="page-5-2"></span>> **[图片提取文字 (无描述)]:**
> AMC23 Olympiad AMC23 Olympiad 56 91 46 +5.3% 89 **?+12.5%** ^+6.7% (%) (62) (62) +7.5% 87 85 (%) 40 <u>ا</u> @ 83 38 81 Pass **S** 50 36 34 52 48 32 Vanilla Linear Increase Exponential Anneal Linear Anneal Constant 51 (a) DeepSeek-R1-Distill-(b) DeepSeek-R1-Distill-(c) DeepSeek-R1-Distill-(d) DeepSeek-R1-Distill-Owen-1.5B Owen-1.5B Owen-7B Owen-7B
![](_page_5_Figure_2.jpeg)

Figure 4: Ablation study of different scheduling strategies on (a-b) AMC23 and (c-d) OlympaidBench.

#### 4.3 Analytic Results

In this section, we analyze  $\alpha 1$  by systematically addressing the following five questions:

## <span id="page-5-0"></span>**4.3.1** What scheduling strategy is better?

As shown in Fig. 3, we study four variants of scheduling strategies for  $\mathcal{S}(t)$  defined in Eq. (1), where  $T_m = \alpha \overline{N}_{\text{think}}$  represents the timestamp of  $\alpha$  moment:

- Constant:  $\mathcal{S}(t) := p_{\text{constant}}$ , where  $p_{\text{constant}} \in [0,1]$  is a constant probability. This represents a consistently more slow thinking strategy, and the increase is large when  $p_{\text{constant}}$  is larger. Note that when  $p_{\text{constant}} = 0$  and  $\alpha = 1$ , it degenerates to vanilla reasoning models; and when  $p_{\text{constant}} < 0.1$  and  $\alpha > 1$ , it degenerates to s1-like model, where only about two "wait" are appended.
- Linear increase:  $S(t) := \frac{1}{T_m}t$ , where  $t = \{0, 1, \ldots, T_m\}$  and  $\frac{1}{T_m} > 0$  indicates the increasing coefficient. This scheduling function indicates a fast-to-slow thinking strategy.
- Exponential anneal:  $S(t) := \exp(-\gamma t)$ , where  $t = \{0, 1, \dots, T_m\}$  and  $\gamma > 0$  is a hyperparameter that controls annealing speed (here we use  $\gamma = 0.3$ ). This scheduling function indicates a slow-to-fast thinking strategy.

• Linear anneal:  $S(t) := -\frac{1}{T_m}t + 1$ , where  $-\frac{1}{T_m} < 0$  indicates the annealing coefficient. Its modulation is similar to exponential anneal scheduling.

Fig. 4 shows the results of  $\alpha 1$  using these four different scheduling strategies. We observe: i) Linear anneal consistently yields the highest reasoning accuracy, indicating that the *slow thinking first*, then fast thinking is a better slow thinking scheduling strategy. ii) Similar to linear anneal, exponential anneal also follows an annealing slow thinking scheduling, where the improvement on the 1.5B model further demonstrates the efficacy of the slow thinking, then fast thinking strategy. However, such annealing scheduling may lead to an unstable performance boost compared to linear anneal.

# **4.3.2** Can $\alpha$ -moment scale the thinking phase budget?

Fig. 5 shows the results of  $\alpha 1$  with different  $\alpha$ -moments determined by scaling  $\alpha$  from 0 to a maximum value subject to the 8192 token length budget. We observe: i)  $\alpha$ -moment enables a *scalable thinking phase budgeting*. By scaling up  $\alpha$ , the average thinking phase token length is accordingly scaled up. ii) Interestingly, while the thinking phase is scaled up, there exists a trade-off between the optimal value of  $\alpha$  and the resulting reasoning accu-

<span id="page-6-0"></span>> **[图片提取文字 (无描述)]:**
> AMC23 AMC23 MATH500 MATH500 6.8 6.8 5.6 75 90 84 5.6 5.8 92 Base: 82.5% 5.8 4.8 82 85 70 4.6 90 4.8 4.8 80 80 65 88 3.6 78 3.8 3.8 3.2 Base: 79.2% Base: 87.6% 86 60 76 2.6 Base: 57.5% 2.8 2.4 74 84 55 1.8 1.6 1.8 1.6 65 82 50 0.8 60 2.5 2.5 2.5  $\alpha$ Average Thinking Phase Token Length (x1000) Pass@1 (%) (b) DeepSeek-R1-Distill-(c) DeepSeek-R1-Distill-(d) DeepSeek-R1-Distill-(a) DeepSeek-R1-Distill-Owen-1.5B Owen-7B Owen-1.5B Owen-7B
![](_page_6_Figure_0.jpeg)

Figure 5: **Scaling property of**  $\alpha$ **.** We scale  $\alpha$  from 0 to the maximum value restricted by the maximum token length, and plot the corresponding reasoning Pass@1 and average thinking phase token length on AMC23 and MATH500.

<span id="page-6-2"></span>> **[图片提取文字 (无描述)]:**
> +11.1 OlympiadBench +4.48 -0.58+12.83 LiveCodeBench +9.43 -2.73 +4.19 MATH500 +8.95 -2.49 Minerva-Math +4.24 -9.98-0.86 +21.94 AMC23 +18.1+9.98 +10.95 AIME24 +4.86 α1 (Ours) 
> ⊠ CoD 
> □ s1
![](_page_6_Figure_2.jpeg)

Figure 6: Scaling efficiency analysis with REP using Deepseek-R1-distill-Qwen-1.5B. The REP metric is introduced in Eq. (2).

racy. This indicates that monotonously increasing the thinking phase budget does not consistently bring better reasoning performance, and it is critical to find the optimal  $\alpha$ -moment that results in a satisfactory improvement.

#### 4.3.3 Does $\alpha 1$ scale more efficiently?

To quantitatively evaluate how different methods trade off reasoning efficiency and accuracy, we introduce the  $\mathcal{F}_{REP}(\mathcal{A}_{method}; \mathcal{A}_{base}, T_{norm})$  (Reasoning Efficiency-Performance, REP) metric. The REP metric is defined as:

<span id="page-6-1"></span>
$$\mathcal{F}_{\text{REP}}(\mathcal{A}_{\text{method}}; \mathcal{A}_{\text{base}}, T_{\text{norm}}) = \frac{\mathcal{A}_{\text{method}} - \mathcal{A}_{\text{base}}}{T_{\text{norm}}}$$
(2)

where  $A_{\text{method}}$  and  $A_{\text{base}}$  denote the reasoning accuracy of the evaluated method and the base model, respectively.  $T_{\text{norm}}$  is the normalized thinking

phase token length, computed by dividing the current thinking phase token length by the maximum token length. Higher REP indicates stronger performance with better reasoning efficiency.

We report the REP of CoD, s1, and  $\alpha$ 1 on six reasoning benchmarks with Deepseek-R1-distill-Qwen-1.5B. Fig. 6 shows that  $\alpha$ 1 achieves higher REP on most benchmarks, indicating a more favorable balance between reasoning performance and efficiency. Notably,  $\alpha$ 1 outperforms CoD by +6.62 and s1 by +11.68 on Olympiad-Bench, and exceeds CoD by +14.22 on Minerva-Math.

# **4.3.4** How frequent should slow thinking transitioning be?

 $\alpha$ 1 modulate slow thinking transitioning via sampling from  $Bernoulli(p_{wait})$ , which leads to another question of how large should  $p_{\mathrm{wait}}$  be that can bring a better result. To study this question, we use the constant scheduling function and scale  $p_{\text{constant}}$  from 0 to 1 to increase the frequency of transitioning to slow thinking. This is because the constant scheduling is a sampling process with a certain probability, and the value of the probability determines how frequently the slow thinking transitioning token will be sampled. Fig. 7 shows the results, from which we observe: i) An extremely low or high frequency of transitioning to slow thinking brings unsatisfactory results (e.g.,  $p_{constant} = 0.1$ ). Similar to the scaling of the thinking phase dedget (e.g., modulating  $\alpha$ ), the slow thinking frequency also needs to be carefully selected. ii) While an extremely dense or sparse slow thinking transitioning leads to unsatisfactory results, the reasoning performance is decent across a large range of  $p_{\text{constant}}$ , demonstrating that increasing slow thinking generally brings improved reasoning.

<span id="page-7-0"></span>> **[图片提取文字 (无描述)]:**
> Olympiad Olympiad AMC23 AMC23 58 86 45 75 44 % <sup>56</sup> 70 84 43 (%) (%) (%) 65 42 Pass@1 Pass@1 Pass@1 Pass@1 41 60 40 ≈ 5 52 55 ≈ 5 39 ≈ 5] ≈ s1 78 38 50 0.2 0.4 0.6 0.6 0.8 0.4 0.6 0.8 0.4 0.6  $\boldsymbol{p}_{\text{constant}}$  $\boldsymbol{p}_{\text{constant}}$  $\boldsymbol{p}_{\text{constant}}$  $p_{\rm constant}$ (a) DeepSeek-R1-Distill-(b) DeepSeek-R1-Distill-(d) DeepSeek-R1-Distill-(c) DeepSeek-R1-Distill-Qwen-1.5B Owen-1.5B Owen-7B Owen-7B
![](_page_7_Figure_0.jpeg)

Figure 7: Scaling property of "**wait**" frequency under constant scheduling on AMC23 and OlympiadBench. Increasing pconstant leads to a higher frequency of yielding "wait" in the Bernoulli process Bernoulli(pwait).

<span id="page-7-1"></span>Table 2: Ablation study on post-α moment modulation. Without post-α modulation represents our α1 without the suppression of the slow thinking inertia after the α moment.

| Method                        | Post-α Moment |      | AIME24 | AMC23 |      |  |  |  |
|-------------------------------|---------------|------|--------|-------|------|--|--|--|
|                               | Modulation    | P@1  | #Tk    | P@1   | #Tk  |  |  |  |
| DeepSeek-R1-Distill-Qwen-1.5B |               |      |        |       |      |  |  |  |
| BASE                          | N/A           | 23.3 | 7280   | 57.5  | 5339 |  |  |  |
| α1 (Ours)                     | ×             | 26.7 | 7929   | 47.5  | 6903 |  |  |  |
| α1 (Ours)                     | ✓             | 30.0 | 5916   | 70.0  | 4951 |  |  |  |
| DeepSeek-R1-Distill-Qwen-7B   |               |      |        |       |      |  |  |  |
| BASE                          | N/A           | 38.8 | 5999   | 82.5  | 4624 |  |  |  |
| α1 (Ours)                     | ×             | 30.0 | 7666   | 75.0  | 5878 |  |  |  |
| α1 (Ours)                     | ✓             | 50.0 | 6826   | 90.0  | 4397 |  |  |  |

# 4.3.5 Is post-α moment modulation necessary?

Typical test-time scaling methods focus on the modulation of slow thinking within the thinking phase, while α1 consists of a post-α moment modulation that encourages fast thinking. To validate its necessity of enforcing fast thinking in the end, we conduct an ablation study on utilizing the post-α moment modulation, shown in Table [2.](#page-7-1) We observe: i) Pre-α moment modulation of slow thinking is insufficient. When the post-α moment modulation is reduced to a single operation, the performance of α1 significantly drops. This is because the increase of slow thinking during pre-α moment brings a slow thinking inertia (as discussed before in Section [3.3\)](#page-3-2), leading to a slow thinking intensive reasoning. ii) By utilizing a post-α moment modulation, α1 successfully ends in a fast thinking, which demonstrates the necessity of *combining both slow thinking and fast thinking*.

## 5 Related Works

#### 5.1 Large Reasoning Models

Large Reasoning Models are rapidly emerging as a family of foundation models [\(Bommasani et al.,](#page-9-4) [2021\)](#page-9-4) that target human-level system-2 reasoning [\(Kahneman,](#page-10-0) [2011\)](#page-10-0). Starting from OpenAI's o1 [\(Jaech et al.,](#page-10-1) [2024\)](#page-10-1) in 2024, numerous efforts follow this "thinking-then-answering" paradigm. Notably, o1-like Large Language Models (LLMs) can solve increasingly complex reasoning problems after a thorough chain of thoughts [\(Wei et al.,](#page-12-6) [2022;](#page-12-6) [Yao et al.,](#page-12-7) [2023;](#page-12-7) [Besta et al.,](#page-9-5) [2024\)](#page-9-5), such as the IMO competition. These advanced models are mainly developed via large-scale reinforcement learning (RL) to align human preference [\(Chris](#page-9-6)[tiano et al.,](#page-9-6) [2017;](#page-9-6) [Schulman et al.,](#page-11-7) [2017;](#page-11-7) [Shao et al.,](#page-11-8) [2024b;](#page-11-8) [DeepSeek-AI et al.,](#page-9-0) [2025\)](#page-9-0), where a reward model is used to judge model answers [\(Uesato et al.,](#page-11-9) [2022;](#page-11-9) [Lightman et al.,](#page-10-7) [2024\)](#page-10-7). Notable efforts replicating o1's success include DeepSeek R1, Qwen QwQ, and Phi-4 [\(Abdin et al.,](#page-8-1) [2024;](#page-8-1) [DeepSeek-AI](#page-9-0) [et al.,](#page-9-0) [2025;](#page-9-0) [Qwen Team,](#page-11-5) [2025\)](#page-11-5), which typically utilize a special end-of-thinking token "</think>", after which a solution is output to the user. Recently, some researchers have explored applying RL during post-training fine-tuning, where promising results have been obtained [\(Chow et al.,](#page-9-7) [2025;](#page-9-7) [Qu et al.,](#page-11-10) [2025;](#page-11-10) [Zuo et al.,](#page-13-1) [2025\)](#page-13-1).

#### 5.2 Reasoning with Test-Time Scaling

Reasoning with test-time scaling has recently become a useful strategy that empowers LLMs with a scalable reasoning capability at test time. The mainstream scaling methods lie in two categories, *i.e.*, i) parallel scaling and ii) sequential scaling. The key idea of parallel scaling is Best-of-N (BoN)

sampling, where the best choice is selected using uncertainty criteria like self-consistency [\(Wang](#page-12-3) [et al.,](#page-12-3) [2023\)](#page-12-3), reward model [\(Lightman et al.,](#page-10-7) [2024;](#page-10-7) [Cobbe et al.,](#page-9-8) [2021\)](#page-9-8), or perplexity [\(Fang et al.,](#page-9-2) [2025\)](#page-9-2). Specifically, one line of work focuses on sequence-level sampling [\(Cobbe et al.,](#page-9-8) [2021;](#page-9-8) [Gui](#page-9-9) [et al.,](#page-9-9) [2024;](#page-9-9) [Wan et al.,](#page-11-3) [2024a;](#page-11-3) [Sun et al.,](#page-11-11) [2024;](#page-11-11) [Chow et al.,](#page-9-7) [2025;](#page-9-7) [Sessa et al.,](#page-11-12) [2025;](#page-11-12) [Amini et al.,](#page-9-10) [2025;](#page-9-10) [Zhou et al.,](#page-13-0) [2025;](#page-13-0) [Zeng et al.,](#page-12-8) [2025;](#page-12-8) [Kang](#page-10-8) [et al.,](#page-10-8) [2025\)](#page-10-8), while another line of work utilizes token-/step- level sampling including beam-/treebased searching [\(Kool et al.,](#page-10-9) [2019;](#page-10-9) [Xie et al.,](#page-12-9) [2023;](#page-12-9) [Zhang et al.,](#page-12-10) [2023;](#page-12-10) [Hao et al.,](#page-10-10) [2023;](#page-10-10) [Qiu et al.,](#page-11-13) [2024;](#page-11-13) [Gao et al.,](#page-9-11) [2024;](#page-9-11) [Yu et al.,](#page-12-11) [2024;](#page-12-11) [Wan et al.,](#page-11-14) [2024b;](#page-11-14) [Chen et al.,](#page-9-12) [2024a\)](#page-9-12). Meanwhile, sequential scaling enhances or reduces slow thinking over a single answer generation process. This technique typically relies on an iterative refinement and revision of answers generated by LLMs themselves [\(Zelikman et al.,](#page-12-12) [2022;](#page-12-12) [Madaan et al.,](#page-11-15) [2023\)](#page-11-15) or external feedback [\(Chen et al.,](#page-9-13) [2024c;](#page-9-13) [Gou et al.,](#page-9-14) [2024;](#page-9-14) [Huang et al.,](#page-10-11) [2024a;](#page-10-11) [Kamoi et al.,](#page-10-12) [2024;](#page-10-12) [Zheng et al.,](#page-13-2) [2024\)](#page-13-2). Following this line of research, recent works have been devoted to addressing the underthinking and overthinking issues of modern LRMs via reinforcing [\(Muennighoff et al.,](#page-11-4) [2025\)](#page-11-4) and restricting [\(Xu et al.,](#page-12-5) [2025\)](#page-12-5) slow thinking, respectively. Given the non-conflict between parallel scaling and sequential scaling, there exists another group of hybrid scaling methods that leverage both strategies [\(Li et al.,](#page-10-13) [2025;](#page-10-13) [Zeng et al.,](#page-12-8) [2025\)](#page-12-8).

## 6 Conclusions

In this paper, we study the problem of test-time scaling of large reasoning models with our framework, ALPHAONE (α1). ALPHAONE starts from a universal view of reasoning modulation targeting two key aspects: thinking phase budgeting and slow thinking scheduling. We introduce α-moment, which is determined by α that scales the thinking phase budget by at least α×. ALPHAONE operates by scheduling slow thinking before the α-moment, and fast thinking after the α-moment that eliminates slow thinking inertia. Using ALPHAONE, we investigate the test-time scaling from various aspects, including the overall slow and fast thinking transitioning plan, thinking phase budget scaling property, and efficiency of test-time scaling, *etc*. Insightful findings are obtained, *e.g.*, slow thinking first, then fast thinking leads to better reasoning capability of LRMs.

## Limitations and Broader Impact

Limitations While ALPHAONE provides a universal view of test-time scaling of LRMs, and a significant performance boost has been achieved, we identify some possible limitations as follows. i) AL-PHAONE targets at o1-style LRMs, where tokens such as "wait" is proved effective in transitioning into slow thinking. However, future LRMs may use a different slow thinking transitioning strategy, leading to a possibility of incompatibility with our framework. ii) ALPHAONE relies on α-moment throughout reasoning modulation, and the average thinking phase token length is typically required. This paper obtains it by first running LRMs on 10 random samples, which requires marginal cost. However, in case that no test questions are available, ALPHAONE can only rely on an empirical thinking phase length that may be suboptimal.

Broader Impact This work targets complex reasoning problems with LRMs, which we believe will lead to no ethical concerns. However, since LRMs are modern variants of LLMs, any ethical concerns raised by LLMs can potentially exist.

# Acknowledgments

Huan Zhang is partially funded by the AI2050 program at Schmidt Sciences (AI2050 Early Career Fellowship). The authors thank Heng Dong for his valuable suggestions on this project.

## References

<span id="page-8-1"></span>Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, and 8 others. 2024. [Phi-4 technical report.](https://arxiv.org/abs/2412.08905) *CoRR*, abs/2412.08905.

<span id="page-8-0"></span>AI-MO. 2024. [AIMO Validation Dataset - AMC.](https://huggingface.co/datasets/AI-MO/aimo-validation-amc) [https://huggingface.co/datasets/AI-MO/](https://huggingface.co/datasets/AI-MO/aimo-validation-amc) [aimo-validation-amc](https://huggingface.co/datasets/AI-MO/aimo-validation-amc). Accessed: 2025-05-19.

<span id="page-8-2"></span>Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, and 8 others. 2022. [Flamingo: a visual](http://papers.nips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html) [language model for few-shot learning.](http://papers.nips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing*

- *Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*.
- <span id="page-9-10"></span>Afra Amini, Tim Vieira, Elliott Ash, and Ryan Cotterell. 2025. [Variational best-of-n alignment.](https://openreview.net/forum?id=W9FZEQj3vv) In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net.
- <span id="page-9-5"></span>Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. 2024. [Graph](https://doi.org/10.1609/AAAI.V38I16.29720) [of thoughts: Solving elaborate problems with large](https://doi.org/10.1609/AAAI.V38I16.29720) [language models.](https://doi.org/10.1609/AAAI.V38I16.29720) In *Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada*, pages 17682–17690. AAAI Press.
- <span id="page-9-4"></span>Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri S. Chatterji, Annie S. Chen, Kathleen Creel, Jared Quincy Davis, Dorottya Demszky, and 34 others. 2021. [On the op](https://arxiv.org/abs/2108.07258)[portunities and risks of foundation models.](https://arxiv.org/abs/2108.07258) *CoRR*, abs/2108.07258.
- <span id="page-9-12"></span>Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. 2024a. [Alphamath almost zero: Process supervision](http://papers.nips.cc/paper_files/paper/2024/hash/30dfe47a3ccbee68cffa0c19ccb1bc00-Abstract-Conference.html) [without process.](http://papers.nips.cc/paper_files/paper/2024/hash/30dfe47a3ccbee68cffa0c19ccb1bc00-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024*.
- <span id="page-9-1"></span>Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2024b. [Do](https://doi.org/10.48550/ARXIV.2412.21187) [NOT think that much for 2+3=? on the overthinking](https://doi.org/10.48550/ARXIV.2412.21187) [of o1-like llms.](https://doi.org/10.48550/ARXIV.2412.21187) *CoRR*, abs/2412.21187.
- <span id="page-9-13"></span>Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. 2024c. [Teaching large language mod](https://openreview.net/forum?id=KuPixIqPiq)[els to self-debug.](https://openreview.net/forum?id=KuPixIqPiq) In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net.
- <span id="page-9-7"></span>Yinlam Chow, Guy Tennenholtz, Izzeddin Gur, Vincent Zhuang, Bo Dai, Aviral Kumar, Rishabh Agarwal, Sridhar Thiagarajan, Craig Boutilier, and Aleksandra Faust. 2025. [Inference-aware fine-tuning for best-of](https://openreview.net/forum?id=77gQUdQhE7)[n sampling in large language models.](https://openreview.net/forum?id=77gQUdQhE7) In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net.
- <span id="page-9-6"></span>Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. [Deep](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) [reinforcement learning from human preferences.](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) In

- *Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA*, pages 4299–4307.
- <span id="page-9-8"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. [Training verifiers to solve math word prob](https://arxiv.org/abs/2110.14168)[lems.](https://arxiv.org/abs/2110.14168) *CoRR*, abs/2110.14168.
- <span id="page-9-0"></span>DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, and 81 others. 2025. [Deepseek-r1: Incentivizing reasoning capa](https://doi.org/10.48550/ARXIV.2501.12948)[bility in llms via reinforcement learning.](https://doi.org/10.48550/ARXIV.2501.12948) *CoRR*, abs/2501.12948.
- <span id="page-9-15"></span>Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. 2024. [Dream-](https://openreview.net/forum?id=y01KGvd9Bw)[LLM: Synergistic multimodal comprehension and](https://openreview.net/forum?id=y01KGvd9Bw) [creation.](https://openreview.net/forum?id=y01KGvd9Bw) In *The Twelfth International Conference on Learning Representations*.
- <span id="page-9-2"></span>Lizhe Fang, Yifei Wang, Zhaoyang Liu, Chenheng Zhang, Stefanie Jegelka, Jinyang Gao, Bolin Ding, and Yisen Wang. 2025. [What is wrong with perplex](https://openreview.net/forum?id=fL4qWkSmtM)[ity for long-context language modeling?](https://openreview.net/forum?id=fL4qWkSmtM) In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net.
- <span id="page-9-16"></span>Li Fei-Fei. 2023. *[The Worlds I See: Curiosity, Explo](https://www.google.com/books/edition/The_Worlds_I_See/6BmhEAAAQBAJ?hl=en&gbpv=0)[ration, and Discovery at the Dawn of AI](https://www.google.com/books/edition/The_Worlds_I_See/6BmhEAAAQBAJ?hl=en&gbpv=0)*. Flatiron books: a moment of lift book.
- <span id="page-9-3"></span>Yichao Fu, Junda Chen, Yonghao Zhuang, Zheyu Fu, Ion Stoica, and Hao Zhang. 2025. [Reasoning without](https://openreview.net/pdf?id=wpK4IMJfdX) [self-doubt: More efficient chain-of-thought through](https://openreview.net/pdf?id=wpK4IMJfdX) [certainty probing.](https://openreview.net/pdf?id=wpK4IMJfdX) In *ICLR 2025 Workshop on Foundation Models in the Wild*.
- <span id="page-9-11"></span>Zitian Gao, Boye Niu, Xuzheng He, Haotian Xu, Hongzhang Liu, Aiwei Liu, Xuming Hu, and Lijie Wen. 2024. [Interpretable contrastive monte carlo tree](https://doi.org/10.48550/ARXIV.2410.01707) [search reasoning.](https://doi.org/10.48550/ARXIV.2410.01707) *CoRR*, abs/2410.01707.
- <span id="page-9-14"></span>Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2024. [CRITIC: large language models can self-correct with](https://openreview.net/forum?id=Sx038qxjek) [tool-interactive critiquing.](https://openreview.net/forum?id=Sx038qxjek) In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. Open-Review.net.
- <span id="page-9-9"></span>Lin Gui, Cristina Garbacea, and Victor Veitch. 2024. [Bonbon alignment for large language models and](http://papers.nips.cc/paper_files/paper/2024/hash/056521a35eacd9d2127b66a7d3c499c5-Abstract-Conference.html) [the sweetness of best-of-n sampling.](http://papers.nips.cc/paper_files/paper/2024/hash/056521a35eacd9d2127b66a7d3c499c5-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024*.

- <span id="page-10-10"></span>Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. [Rea](https://doi.org/10.18653/V1/2023.EMNLP-MAIN.507)[soning with language model is planning with world](https://doi.org/10.18653/V1/2023.EMNLP-MAIN.507) [model.](https://doi.org/10.18653/V1/2023.EMNLP-MAIN.507) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pages 8154–8173. Association for Computational Linguistics.
- <span id="page-10-15"></span>Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. [Training large language models to reason in a contin](https://doi.org/10.48550/ARXIV.2412.06769)[uous latent space.](https://doi.org/10.48550/ARXIV.2412.06769) *CoRR*, abs/2412.06769.
- <span id="page-10-6"></span>Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. [Olympiadbench:](https://doi.org/10.18653/V1/2024.ACL-LONG.211) [A challenging benchmark for promoting AGI with](https://doi.org/10.18653/V1/2024.ACL-LONG.211) [olympiad-level bilingual multimodal scientific prob](https://doi.org/10.18653/V1/2024.ACL-LONG.211)[lems.](https://doi.org/10.18653/V1/2024.ACL-LONG.211) In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024*, pages 3828–3850. Association for Computational Linguistics.
- <span id="page-10-11"></span>Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024a. [Large language](https://openreview.net/forum?id=IkmD3fKBPQ) [models cannot self-correct reasoning yet.](https://openreview.net/forum?id=IkmD3fKBPQ) In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net.
- <span id="page-10-3"></span>Zhen Huang, Haoyang Zou, Xuefeng Li, Yixiu Liu, Yuxiang Zheng, Ethan Chern, Shijie Xia, Yiwei Qin, Weizhe Yuan, and Pengfei Liu. 2024b. [O1 replication](https://doi.org/10.48550/ARXIV.2411.16489) [journey - part 2: Surpassing o1-preview through sim](https://doi.org/10.48550/ARXIV.2411.16489)[ple distillation, big progress or bitter lesson?](https://doi.org/10.48550/ARXIV.2411.16489) *CoRR*, abs/2411.16489.
- <span id="page-10-1"></span>Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, and 80 others. 2024. [Openai o1 system card.](https://doi.org/10.48550/ARXIV.2412.16720) *CoRR*, abs/2412.16720.
- <span id="page-10-5"></span>Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. [Live](https://openreview.net/forum?id=chfJJYC3iL)[codebench: Holistic and contamination free evalua](https://openreview.net/forum?id=chfJJYC3iL)[tion of large language models for code.](https://openreview.net/forum?id=chfJJYC3iL) In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net.
- <span id="page-10-17"></span>Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. 2025. [T2i-r1: Reinforcing image gen](https://arxiv.org/abs/2505.00703)[eration with collaborative semantic-level and token](https://arxiv.org/abs/2505.00703)[level cot.](https://arxiv.org/abs/2505.00703) *arXiv preprint arXiv:2505.00703*.

- <span id="page-10-0"></span>Daniel Kahneman. 2011. *[Thinking, fast and slow](https://www.google.com/books/edition/_/SHvzzuCnuv8C?hl=en)*. macmillan.
- <span id="page-10-12"></span>Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang. 2024. [When can llms](https://doi.org/10.1162/TACL_A_00713) *Actually* cor[rect their own mistakes? A critical survey of self](https://doi.org/10.1162/TACL_A_00713)[correction of llms.](https://doi.org/10.1162/TACL_A_00713) *Trans. Assoc. Comput. Linguistics*, 12:1417–1440.
- <span id="page-10-8"></span>Zhewei Kang, Xuandong Zhao, and Dawn Song. 2025. [Scalable best-of-n selection for large language mod](https://doi.org/10.48550/ARXIV.2502.18581)[els via self-certainty.](https://doi.org/10.48550/ARXIV.2502.18581) *CoRR*, abs/2502.18581.
- <span id="page-10-9"></span>Wouter Kool, Herke van Hoof, and Max Welling. 2019. [Stochastic beams and where to find them: The](http://proceedings.mlr.press/v97/kool19a.html) [gumbel-top-k trick for sampling sequences without](http://proceedings.mlr.press/v97/kool19a.html) [replacement.](http://proceedings.mlr.press/v97/kool19a.html) In *Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA*, volume 97 of *Proceedings of Machine Learning Research*, pages 3499–3508. PMLR.
- <span id="page-10-16"></span>Seongyun Lee, Sue Hyun Park, Yongrae Jo, and Minjoon Seo. 2024. [Volcano: Mitigating multimodal](https://doi.org/10.18653/V1/2024.NAACL-LONG.23) [hallucination through self-feedback guided revision.](https://doi.org/10.18653/V1/2024.NAACL-LONG.23) In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024*, pages 391–404. Association for Computational Linguistics.
- <span id="page-10-4"></span>Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. [Solving quan](http://papers.nips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html)[titative reasoning problems with language models.](http://papers.nips.cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*.
- <span id="page-10-13"></span>Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E. Gonzalez, and Ion Stoica. 2025. [S\\*: Test time scal](https://doi.org/10.48550/ARXIV.2502.14382)[ing for code generation.](https://doi.org/10.48550/ARXIV.2502.14382) *CoRR*, abs/2502.14382.
- <span id="page-10-7"></span>Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. [Let's verify step by step.](https://openreview.net/forum?id=v8L0pN6EOi) In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. Open-Review.net.
- <span id="page-10-14"></span>Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. [Visual instruction tuning.](http://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*.
- <span id="page-10-2"></span>Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. 2025. [Reasoning](https://arxiv.org/abs/2504.09858) [models can be effective without thinking.](https://arxiv.org/abs/2504.09858) *arXiv preprint arXiv:2504.09858*.

- <span id="page-11-15"></span>Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. [Self-refine: Itera](http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)[tive refinement with self-feedback.](http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*.
- <span id="page-11-6"></span>Mathematical Association of America. 2024. [Ameri](https://maa.org/math-competitions/american-invitational-mathematics-examination-aime)[can Invitational Mathematics Examination – AIME.](https://maa.org/math-competitions/american-invitational-mathematics-examination-aime) *American Invitational Mathematics Examination – AIME 2024*. Accessed: 2025-05-15.
- <span id="page-11-4"></span>Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel J. Candès, and Tatsunori Hashimoto. 2025. [s1: Simple test-time](https://doi.org/10.48550/ARXIV.2501.19393) [scaling.](https://doi.org/10.48550/ARXIV.2501.19393) *CoRR*, abs/2501.19393.
- <span id="page-11-16"></span>OpenAI. 2024. [Introducing gpt-4o and more tools to](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/) [chatgpt free users.](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/)
- <span id="page-11-20"></span>OpenAI. 2025. [Thinking with images.](https://openai.com/index/thinking-with-images/) Accessed: 2025- 05-25.
- <span id="page-11-1"></span>Xiao Pu, Michael Saxon, Wenyue Hua, and William Yang Wang. 2025. [Thoughtterminator:](https://arxiv.org/abs/2504.13367) [Benchmarking, calibrating, and mitigating over](https://arxiv.org/abs/2504.13367)[thinking in reasoning models.](https://arxiv.org/abs/2504.13367) *arXiv preprint arXiv:2504.13367*.
- <span id="page-11-18"></span>Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma. 2024. [Shapellm: Universal 3d object understand](https://doi.org/10.1007/978-3-031-72775-7_13)[ing for embodied interaction.](https://doi.org/10.1007/978-3-031-72775-7_13) In *Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLIII*, volume 15101 of *Lecture Notes in Computer Science*, pages 214–238. Springer.
- <span id="page-11-21"></span>Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, Jiazhao Zhang, Jiawei He, Jiayuan Gu, Xin Jin, Kaisheng Ma, Zhizheng Zhang, He Wang, and Li Yi. 2025. [Sofar: Language](https://doi.org/10.48550/ARXIV.2502.13143)[grounded orientation bridges spatial reasoning and](https://doi.org/10.48550/ARXIV.2502.13143) [object manipulation.](https://doi.org/10.48550/ARXIV.2502.13143) *CoRR*, abs/2502.13143.
- <span id="page-11-13"></span>Jiahao Qiu, Yifu Lu, Yifan Zeng, Jiacheng Guo, Jiayi Geng, Huazheng Wang, Kaixuan Huang, Yue Wu, and Mengdi Wang. 2024. [Treebon: Enhancing](https://doi.org/10.48550/ARXIV.2410.16033) [inference-time alignment with speculative tree-search](https://doi.org/10.48550/ARXIV.2410.16033) [and best-of-n sampling.](https://doi.org/10.48550/ARXIV.2410.16033) *CoRR*, abs/2410.16033.
- <span id="page-11-10"></span>Yuxiao Qu, Matthew Y. R. Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, and Aviral Kumar. 2025. [Optimizing](https://doi.org/10.48550/ARXIV.2503.07572) [test-time compute via meta reinforcement fine-tuning.](https://doi.org/10.48550/ARXIV.2503.07572) *CoRR*, abs/2503.07572.
- <span id="page-11-5"></span>Qwen Team. 2025. Preview of qwen qwen1.5- 32b. [https://qwenlm.github.io/blog/](https://qwenlm.github.io/blog/qwq-32b-preview/) [qwq-32b-preview/](https://qwenlm.github.io/blog/qwq-32b-preview/). Accessed: 2025-03-20.

- <span id="page-11-7"></span>John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. [Proximal policy](https://arxiv.org/abs/1707.06347) [optimization algorithms.](https://arxiv.org/abs/1707.06347) *CoRR*, abs/1707.06347.
- <span id="page-11-12"></span>Pier Giuseppe Sessa, Robert Dadashi-Tazehozi, Léonard Hussenot, Johan Ferret, Nino Vieillard, Alexandre Ramé, Bobak Shahriari, Sarah Perrin, Abram L. Friesen, Geoffrey Cideron, Sertan Girgin, Piotr Stanczyk, Andrea Michi, Danila Sinopalnikov, Sabela Ramos Garea, Amélie Héliou, Aliaksei Severyn, Matthew Hoffman, Nikola Momchev, and Olivier Bachem. 2025. [BOND: aligning llms with](https://openreview.net/forum?id=0tAXMiSufG) [best-of-n distillation.](https://openreview.net/forum?id=0tAXMiSufG) In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net.
- <span id="page-11-19"></span>Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. 2024a. [Visual cot: Advancing multi-modal](https://openreview.net/forum?id=aXeiCbMFFJ) [language models with a comprehensive dataset and](https://openreview.net/forum?id=aXeiCbMFFJ) [benchmark for chain-of-thought reasoning.](https://openreview.net/forum?id=aXeiCbMFFJ) In *The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track*.
- <span id="page-11-8"></span>Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024b. [Deepseekmath: Pushing the](https://doi.org/10.48550/ARXIV.2402.03300) [limits of mathematical reasoning in open language](https://doi.org/10.48550/ARXIV.2402.03300) [models.](https://doi.org/10.48550/ARXIV.2402.03300) *CoRR*, abs/2402.03300.
- <span id="page-11-2"></span>Jinyan Su, Jennifer Healey, Preslav Nakov, and Claire Cardie. 2025. [Between underthinking and overthink](https://arxiv.org/abs/2505.00127)[ing: An empirical study of reasoning length and cor](https://arxiv.org/abs/2505.00127)[rectness in llms.](https://arxiv.org/abs/2505.00127) *arXiv preprint arXiv:2505.00127*.
- <span id="page-11-0"></span>Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, and Xia Ben Hu. 2025. [Stop overthinking: A survey on](https://doi.org/10.48550/ARXIV.2503.16419) [efficient reasoning for large language models.](https://doi.org/10.48550/ARXIV.2503.16419) *CoRR*, abs/2503.16419.
- <span id="page-11-11"></span>Hanshi Sun, Momin Haider, Ruiqi Zhang, Huitao Yang, Jiahao Qiu, Ming Yin, Mengdi Wang, Peter Bartlett, and Andrea Zanette. 2024. [Fast best-of-n decoding](https://openreview.net/forum?id=348hfcprUs) [via speculative rejection.](https://openreview.net/forum?id=348hfcprUs) In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*.
- <span id="page-11-17"></span>Chameleon Team. 2024. [Chameleon: Mixed](https://doi.org/10.48550/ARXIV.2405.09818)[modal early-fusion foundation models.](https://doi.org/10.48550/ARXIV.2405.09818) *CoRR*, abs/2405.09818.
- <span id="page-11-9"></span>Jonathan Uesato, Nate Kushman, Ramana Kumar, H. Francis Song, Noah Y. Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. 2022. [Solving math word problems with process- and](https://doi.org/10.48550/ARXIV.2211.14275) [outcome-based feedback.](https://doi.org/10.48550/ARXIV.2211.14275) *CoRR*, abs/2211.14275.
- <span id="page-11-3"></span>Guangya Wan, Yuqi Wu, Jie Chen, and Sheng Li. 2024a. [Dynamic self-consistency: Leveraging rea](https://doi.org/10.48550/ARXIV.2408.17017)[soning paths for efficient LLM sampling.](https://doi.org/10.48550/ARXIV.2408.17017) *CoRR*, abs/2408.17017.
- <span id="page-11-14"></span>Ziyu Wan, Xidong Feng, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang.

- 2024b. [Alphazero-like tree-search can guide large](https://openreview.net/forum?id=C4OpREezgj) [language model decoding and training.](https://openreview.net/forum?id=C4OpREezgj) In *Fortyfirst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024*. Open-Review.net.
- <span id="page-12-13"></span>Han Wang, Gang Wang, and Huan Zhang. 2024. [Steer](https://doi.org/10.48550/ARXIV.2411.16721)[ing away from harm: An adaptive approach to defend](https://doi.org/10.48550/ARXIV.2411.16721)[ing vision language model against jailbreaks.](https://doi.org/10.48550/ARXIV.2411.16721) *CoRR*, abs/2411.16721.
- <span id="page-12-3"></span>Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. [Self-consistency](https://openreview.net/forum?id=1PL1NIMMrw) [improves chain of thought reasoning in language](https://openreview.net/forum?id=1PL1NIMMrw) [models.](https://openreview.net/forum?id=1PL1NIMMrw) In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net.
- <span id="page-12-2"></span>Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2025. [Thoughts are all over the](https://doi.org/10.48550/ARXIV.2501.18585) [place: On the underthinking of o1-like llms.](https://doi.org/10.48550/ARXIV.2501.18585) *CoRR*, abs/2501.18585.
- <span id="page-12-15"></span>Haoran Wei, Youyang Yin, Yumeng Li, Jia Wang, Liang Zhao, Jianjian Sun, Zheng Ge, and Xiangyu Zhang. 2024. [Slow perception: Let's perceive geometric](https://doi.org/10.48550/ARXIV.2412.20631) [figures step-by-step.](https://doi.org/10.48550/ARXIV.2412.20631) *CoRR*, abs/2412.20631.
- <span id="page-12-6"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. [Chain-of-thought prompting](http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) [elicits reasoning in large language models.](http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*.
- <span id="page-12-17"></span>Yana Wei, Liang Zhao, Kangheng Lin, En Yu, Yuang Peng, Runpei Dong, Jianjian Sun, Haoran Wei, Zheng Ge, Xiangyu Zhang, and Vishal M. Patel. 2025. [Perception in reflection.](https://doi.org/10.48550/ARXIV.2504.07165) *CoRR*, abs/2504.07165.
- <span id="page-12-16"></span>Penghao Wu and Saining Xie. 2024. [V\\*: Guided vi](https://doi.org/10.1109/CVPR52733.2024.01243)[sual search as a core mechanism in multimodal llms.](https://doi.org/10.1109/CVPR52733.2024.01243) In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024*, pages 13084–13094. IEEE.
- <span id="page-12-9"></span>Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Qizhe Xie. 2023. [Self-evaluation guided beam search for](http://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html) [reasoning.](http://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*.
- <span id="page-12-14"></span>Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. 2024. [Llava-critic: Learning to evaluate](https://doi.org/10.48550/ARXIV.2410.02712) [multimodal models.](https://doi.org/10.48550/ARXIV.2410.02712) *CoRR*, abs/2410.02712.
- <span id="page-12-5"></span>Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. 2025. [Chain of draft: Thinking faster by writing](https://doi.org/10.48550/ARXIV.2502.18600) [less.](https://doi.org/10.48550/ARXIV.2502.18600) *CoRR*, abs/2502.18600.

- <span id="page-12-4"></span>Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. 2025a. [Dynamic early exit in reasoning models.](https://arxiv.org/abs/2504.15895) *arXiv preprint arXiv:2504.15895*.
- <span id="page-12-19"></span>Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. 2025b. [Embodiedbench: Comprehensive benchmark](https://doi.org/10.48550/ARXIV.2502.09560)[ing multi-modal large language models for vision](https://doi.org/10.48550/ARXIV.2502.09560)[driven embodied agents.](https://doi.org/10.48550/ARXIV.2502.09560) *CoRR*, abs/2502.09560.
- <span id="page-12-0"></span>Wang Yang, Xiang Yue, Vipin Chaudhary, and Xiaotian Han. 2025c. [Speculative thinking: Enhancing](https://arxiv.org/abs/2504.12329) [small-model reasoning with large model guidance at](https://arxiv.org/abs/2504.12329) [inference time.](https://arxiv.org/abs/2504.12329) *arXiv preprint arXiv:2504.12329*.
- <span id="page-12-1"></span>Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. 2025d. [Towards thinking-optimal scaling](https://doi.org/10.48550/ARXIV.2502.18080) [of test-time compute for LLM reasoning.](https://doi.org/10.48550/ARXIV.2502.18080) *CoRR*, abs/2502.18080.
- <span id="page-12-7"></span>Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. [Tree of thoughts: Deliberate problem solving](http://papers.nips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html) [with large language models.](http://papers.nips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*.
- <span id="page-12-18"></span>En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang, and Wenbing Tao. 2025. [Perception](https://doi.org/10.48550/ARXIV.2504.07954)[r1: Pioneering perception policy with reinforcement](https://doi.org/10.48550/ARXIV.2504.07954) [learning.](https://doi.org/10.48550/ARXIV.2504.07954) *CoRR*, abs/2504.07954.
- <span id="page-12-11"></span>Fei Yu, Anningzhe Gao, and Benyou Wang. 2024. [Ovm,](https://doi.org/10.18653/V1/2024.FINDINGS-NAACL.55) [outcome-supervised value models for planning in](https://doi.org/10.18653/V1/2024.FINDINGS-NAACL.55) [mathematical reasoning.](https://doi.org/10.18653/V1/2024.FINDINGS-NAACL.55) In *Findings of the Association for Computational Linguistics: NAACL 2024, Mexico City, Mexico, June 16-21, 2024*, pages 858– 875. Association for Computational Linguistics.
- <span id="page-12-12"></span>Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. 2022. [Star: Bootstrapping reasoning with](http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) [reasoning.](http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*.
- <span id="page-12-8"></span>Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Yunhua Zhou, and Xipeng Qiu. 2025. [Revisiting the test-time](https://doi.org/10.48550/ARXIV.2502.12215) [scaling of o1-like models: Do they truly possess test](https://doi.org/10.48550/ARXIV.2502.12215)[time scaling capabilities?](https://doi.org/10.48550/ARXIV.2502.12215) *CoRR*, abs/2502.12215.
- <span id="page-12-10"></span>Shun Zhang, Zhenfang Chen, Yikang Shen, Mingyu Ding, Joshua B. Tenenbaum, and Chuang Gan. 2023. [Planning with large language models for code gen](https://openreview.net/forum?id=Lr8cOOtYbfL)[eration.](https://openreview.net/forum?id=Lr8cOOtYbfL) In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net.

- <span id="page-13-2"></span>Jingnan Zheng, Han Wang, An Zhang, Tai D. Nguyen, Jun Sun, and Tat-Seng Chua. 2024. [Ali-agent: As](http://papers.nips.cc/paper_files/paper/2024/hash/b35c38f70065ac6c694089ca93a015bb-Abstract-Conference.html)[sessing llms' alignment with human values via agent](http://papers.nips.cc/paper_files/paper/2024/hash/b35c38f70065ac6c694089ca93a015bb-Abstract-Conference.html)[based evaluation.](http://papers.nips.cc/paper_files/paper/2024/hash/b35c38f70065ac6c694089ca93a015bb-Abstract-Conference.html) In *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024*.
- <span id="page-13-0"></span>Zhi Zhou, Tan Yuhao, Zenan Li, Yuan Yao, Lan-Zhe Guo, Xiaoxing Ma, and Yu-Feng Li. 2025. [Bridging internal probability and self-consistency](https://doi.org/10.48550/ARXIV.2502.00511) [for effective and efficient LLM reasoning.](https://doi.org/10.48550/ARXIV.2502.00511) *CoRR*, abs/2502.00511.
- <span id="page-13-3"></span>Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. 2025. [Dynamath: A dy](https://openreview.net/forum?id=VOAMTA8jKu)[namic visual benchmark for evaluating mathematical](https://openreview.net/forum?id=VOAMTA8jKu) [reasoning robustness of vision language models.](https://openreview.net/forum?id=VOAMTA8jKu) In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net.
- <span id="page-13-1"></span>Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang Sun, Ganqu Cui, Ning Ding, and Bowen Zhou. 2025. [Ttrl: Test-time rein](https://arxiv.org/abs/2504.16084)[forcement learning.](https://arxiv.org/abs/2504.16084) *CoRR*, abs/2504.16084.

# APPENDIX

| A |     | Additional Implementation Details  | 15 |
|---|-----|------------------------------------|----|
|   | A.1 | Computaional Budget                | 15 |
|   | A.2 | Hyper-parameters & Parameters      | 15 |
|   | A.3 | Benchmarks                         | 15 |
| B |     | Additional Ablation Study          | 16 |
|   | B.1 | Scheduling Strategy                | 16 |
|   | B.2 | Scaling Efficiency Analysis        | 16 |
|   | B.3 | Slow Thinking Transitioning Tokens | 16 |
| C |     | Artifacts Statements               | 16 |
|   | C.1 | Model Artifacts<br>                | 16 |
|   | C.2 | Data Artifacts<br>                 | 16 |
| D |     | Future Works                       | 17 |
| E |     | Qualitative Examples               | 17 |

## <span id="page-14-0"></span>A Additional Implementation Details

#### <span id="page-14-1"></span>A.1 Computaional Budget

We used 8 NVIDIA L40S GPUs and 4 NVIDIA A100 80GB GPUs for the experiments.

#### <span id="page-14-2"></span>A.2 Hyper-parameters & Parameters

For reproducibility, we provide the complete set of average thinking phase token length Nthink in Table [3,](#page-15-7) which are obtained by randomly sampling 10 test questions on each benchmark and averaging the generated token lengths. Since the effective range of α observed in Figure [5](#page-6-0) is relatively broad, practical implementations can tolerate variance in this measurement.

#### <span id="page-14-3"></span>A.3 Benchmarks

AIME 2024 The AIME 2024 dataset is a specialized benchmark collection consisting of 30 problems from the 2024 American Invitational Mathematics Examination [\(Mathematical Association](#page-11-6) [of America,](#page-11-6) [2024\)](#page-11-6). These problems cover core secondary-school mathematics topics such as arithmetic, combinatorics, algebra, geometry, number theory and probability. The collection places rigorous demands on both solution accuracy and conceptual depth.

AMC 2023 The AMC 2023 dataset consists of 40 problems selected from the AMC 12A and 12B contests. These exams are sponsored by the Mathematical Association of America and target U.S. students in grade 12 and below, featuring challenges in algebra, geometry, number theory, and combinatorics [\(AI-MO,](#page-8-0) [2024\)](#page-8-0).

Minerva Math Minerva Math [\(Lewkowycz](#page-10-4) [et al.,](#page-10-4) [2022\)](#page-10-4) consists of 272 undergraduate-level STEM problems harvested from MIT's Open-CourseWare. These problems span solid-state chemistry, information and entropy, differential equations, and special relativity. Each includes a clearly delineated answer—191 verifiable by numeric checks and 81 by symbolic solutions. The benchmark is specifically designed to evaluate multi-step scientific reasoning capabilities in language models.

MATH500 MATH500 comprises a selection of 500 problems extracted from the MATH benchmark [\(Lightman et al.,](#page-10-7) [2024\)](#page-10-7). The collection covers a range of high-school mathematics domains, including Prealgebra, Algebra and Number Theory. To ensure comparability with prior work, we use the exact problem set originally curated by OpenAI for evaluation.

LiveCodeBench LiveCodeBench [\(Jain et al.,](#page-10-5) [2025\)](#page-10-5) is a contamination-free benchmark for evaluating large language models on code. The suite is continuously updated, gathering new problems over time. It currently comprises 400 Python programming tasks released between May 2023 and March 2024, each paired with test samples for correctness verification. Beyond basic code generation, LiveCodeBench also measures advanced capabilities such as self-repair, code execution and test-output prediction.

OlympiadBench OlympiadBench [\(He et al.,](#page-10-6) [2024\)](#page-10-6) consists of 8,476 Olympiad-level problems that evaluate mathematical and physical reasoning in AI systems. It features a wide difficulty range, open-ended problem generation, expert solution annotations, detailed difficulty labels, and multilingual coverage. The subset we use in our paper contains 675 open-ended, text-only math competition problems in English.

<span id="page-15-7"></span>Table 3: Average thinking phase token length  $\overline{N}_{\text{think}}$  across different benchmarks. The results are obtained by running LRMs on randomly sampled 10 samples.

| Model                         | AIME24 | AMC23 | Minerva | MATH500 | LiveCodeBench | OlympiadBench |
|-------------------------------|--------|-------|---------|---------|---------------|---------------|
| DeepSeek-R1-Distill-Qwen-1.5B | 4130   | 3303  | 3101    | 2435    | 2172          | 3417          |
| DeepSeek-R1-Distill-Qwen-7B   | 4751   | 3243  | 3064    | 2352    | 3120          | 3330          |
| Qwen QwQ-32B                  | 2597   | 2124  | 1710    | 1493    | 4915          | 2052          |

<span id="page-15-8"></span>> **[图片提取文字 (无描述)]:**
> Constant Exponential Anneal Vanilla Linear Anneal 51 Linear Increase 52 30 50 +6.7% +3.3% **%** 28 © 26 @ 46 Pass 22 42 20 40 (a) DeepSeek-R1-Distill-(b) DeepSeek-R1-Distill-Qwen-1.5B Qwen-7B
![](_page_15_Figure_2.jpeg)

Figure 8: **Ablation study of different scheduling strategies** on AIME24.

#### <span id="page-15-0"></span>**B** Additional Ablation Study

#### <span id="page-15-1"></span>**B.1** Scheduling Strategy

In addition to the results in Fig. 4 tested on AMC23 and Olympiad, we also show the results tested on AIME24 in Fig. 8. From the results, we observe that the linear increase consistently yields the best performance, which aligns with our previous observation. This further provides evidence that slow-then-fast thinking is an efficient slow-thinking scheduling strategy.

## <span id="page-15-2"></span>**B.2** Scaling Efficiency Analysis

As shown in Fig. 9,  $\alpha 1$  consistently achieves positive REP with Deepseek-R1-distill-Qwen-7B, demonstrating stable gains over the base model. Similar to Fig. 6, it outperforms CoD and s1 across nearly all benchmarks, particularly on Live-CodeBench and AIME24.

#### <span id="page-15-3"></span>**B.3** Slow Thinking Transitioning Tokens

We provide an ablation study on different slow-thinking transitioning tokens on the AIME2024 dataset. As illustrated in Table 4, the empirical results show that using "Wait," can help the model excel in both performance and efficiency. Other reasoning transition tokens like "Hmm," and "Alternatively," do not achieve comparable results, likely because they introduce less effective cues for reasoning modulation.

<span id="page-15-9"></span>> **[图片提取文字 (无描述)]:**
> OlympiadBench +7.66 +12.07 LiveCodeBench +4.34 +1.25 +7.66 MATH500 +7.06 +4.19 Minerva-Math +16.09 +2.84 +15.92 AMC23 +15.66 -4.99 AIME24 +4.43 -7.60 α1 (Ours)
![](_page_15_Figure_11.jpeg)

Figure 9: **Scaling efficiency analysis with REP** using Deepseek-R1-distill-Qwen-7B.

## <span id="page-15-4"></span>C Artifacts Statements

#### <span id="page-15-5"></span>C.1 Model Artifacts

We utilize three models in our work: DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B, both released under the MIT License, which permits commercial use, modification, and redistribution. These models are distilled from Qwen-2.5 series (Apache 2.0 License). Additionally, we use Qwen QwQ-32B, which is released under the Apache License 2.0, allowing both research and commercial usage. We comply with all respective license terms in our use of these models.

#### <span id="page-15-6"></span>C.2 Data Artifacts

We employ publicly available datasets in our experiments. AIME24, Minerva-Math, LiveCodeBench, and OlympiadBench are released under the MIT License, which permits unrestricted use, modification, and redistribution. The AMC23 dataset does not have an explicitly specified license, so we treat it as having an unspecified license and exercise caution in its usage. We ensure full compliance with the respective license terms of all datasets used.

# <span id="page-16-0"></span>D Future Works

While our α1 has been demonstrated successful and effective in scaling LRMs at test time, there are some intriguing future works that we are considering:

- *More sophisticated slow thinking scheduling.* This work focuses on simple strategies like the slow-to-fast schedule, which shows strong performance. However, optimal scheduling remains an open question, as human reasoning patterns are complex and not yet fully understood [\(Kahne](#page-10-0)[man,](#page-10-0) [2011\)](#page-10-0). Promising directions include modulating reasoning progress during both training and inference, or learning a separate *progress modulation model* aligned with human preferences—akin to a progress reward model [\(Uesato](#page-11-9) [et al.,](#page-11-9) [2022;](#page-11-9) [Lightman et al.,](#page-10-7) [2024\)](#page-10-7).
- *Transitioning-token-agnostic modulation.* As shown in Table [4,](#page-16-2) the choice of transitioning token (e.g., "wait") affects performance due to model-specific training data. This limitation is shared by many test-time scaling methods relying on open-source LRMs like DeepSeek-R1 [\(DeepSeek-AI et al.,](#page-9-0) [2025\)](#page-9-0), in contrast to restricted-access models like OpenAI o1. While α1 supports flexible token choices, removing the dependency on transitioning tokens altogether could further enhance generalization.
- *Multimodal reasoning with multimodal LLMs.* Multimodal LLMs are rapidly advancing and show growing potential in reasoning tasks [\(Alayrac et al.,](#page-8-2) [2022;](#page-8-2) [Liu et al.,](#page-10-14) [2023;](#page-10-14) [Ope](#page-11-16)[nAI,](#page-11-16) [2024;](#page-11-16) [Team,](#page-11-17) [2024;](#page-11-17) [Dong et al.,](#page-9-15) [2024;](#page-9-15) [Qi](#page-11-18) [et al.,](#page-11-18) [2024;](#page-11-18) [Zou et al.,](#page-13-3) [2025;](#page-13-3) [Wang et al.,](#page-12-13) [2024\)](#page-12-13). Although they currently trail behind text-only LRMs, efforts to enhance their reasoning abilities are gaining momentum [\(Hao et al.,](#page-10-15) [2024;](#page-10-15) [Xiong et al.,](#page-12-14) [2024;](#page-12-14) [Wei et al.,](#page-12-15) [2024;](#page-12-15) [Wu and Xie,](#page-12-16) [2024;](#page-12-16) [Shao et al.,](#page-11-19) [2024a;](#page-11-19) [Lee et al.,](#page-10-16) [2024;](#page-10-16) [Wei](#page-12-17) [et al.,](#page-12-17) [2025;](#page-12-17) [Jiang et al.,](#page-10-17) [2025;](#page-10-17) [Yu et al.,](#page-12-18) [2025\)](#page-12-18). Multimodal reasoning remains challenging, as it requires both image understanding and generation [\(OpenAI,](#page-11-20) [2025\)](#page-11-20). We aim to extend our α1 framework to this domain, fostering synergistic multimodal comprehension and creation [\(Dong](#page-9-15) [et al.,](#page-9-15) [2024\)](#page-9-15). Another promising direction is embodied reasoning—grounding multimodal understanding in real-world interactions with spatial intelligence [\(Fei-Fei,](#page-9-16) [2023;](#page-9-16) [Qi et al.,](#page-11-21) [2025;](#page-11-21) [Yang](#page-12-19) [et al.,](#page-12-19) [2025b\)](#page-12-19).

<span id="page-16-2"></span>Table 4: Ablation study on different slow thinking transitioning tokens on AIME24 (8192).

| Transitioning Token |      | Deepseek-R1-1.5B | Deepseek-R1-7B |      |  |
|---------------------|------|------------------|----------------|------|--|
|                     | P@1  | #Tk              | P@1            | #Tk  |  |
| "Wait,"             | 30.0 | 5916             | 50.0           | 6827 |  |
| "Hmm,"              | 20.0 | 6595             | 46.7           | 6374 |  |
| "Alternatively,"    | 16.7 | 6713             | 43.3           | 6603 |  |

# <span id="page-16-1"></span>E Qualitative Examples

We present qualitative examples from different models and benchmarks to illustrate both cases: instances where α1 helps the model answer correctly, and instances where it still fails to produce the correct answer. Examples show that by appending "wait" frequently after "\n\n" can slow down the thinking process and may help the model achieve better performance. The examples can be found in the following pages.

# Successful Example of DeepSeek-R1-Distill-Qwen-1.5B on OlympiadBench: Base fails while α1 passes

**Question**: A chemist has three bottles, each containing a mixture of acid and water: - bottle A contains 40g of which 10 is acid, - bottle B contains 50g of which 20 is acid, and, - bottle C contains 50g of which 30 is acid. She uses some of the mixture from each of the bottles to create a mixture with mass 60g of which 25 is acid. Then she mixes the remaining contents of the bottles to create a new mixture. What percentage of the new mixture is acid?

#### **Response from Base**: <think>

Okay, so I've got this chemistry problem here, and I need to figure out the percentage of acid in the new mixture. Let me try to understand the problem step by step.

First, there are three bottles, each with a mixture of acid and water. The masses and acid concentrations are given as follows:

- Bottle A: 40 grams, 10% acid.
- Bottle B: 50 grams, 20% acid.
- Bottle C: 50 grams, 30% acid.

The chemist is going to use some mixture from each bottle to create a new mixture that's 60 grams with 25% acid. Then, after using some from each bottle, she mixes the remaining contents to create a new mixture. ... </think>

The chemist has three bottles containing acid and water with the following compositions:

- Bottle A: 40 grams of solution, 10% acid... The percentage of the new mixture is 18.75 ).

### **Response from** α**1**: <think>

Okay, so I've got this chemistry problem here, and I need to figure out the percentage of acid in the new mixture. Let me try to understand the problem step by step.

Wait, actually, the problem says that the chemist uses some mixture from each bottle to create a 60g mixture with 25% acid. Then, she mixes the remaining contents of the bottles to create a new mixture. Hmm, I need to clarify if she uses some from each bottle to make 60g, and then mixes the leftover from each bottle. Or is it that she uses some from each bottle to make 60g, and then uses the leftovers from each bottle to make another mixture? The wording is a bit unclear.

Wait, let me re-read the problem.

Wait, actually, the problem says: ... </think>

The percentage of acid in the new mixture is 17.5%.

#### **Step-by-Step Explanation:**

- 1. **Initial Acid Quantities:**
- Bottle A: 40g of 10% acid → 4g acid.
- Bottle B: 50g of 20% acid → 10g acid...

**Answer:** The new mixture has 17.5 % acid.

**Correct Answer**: 17.5

## Failure Example of DeepSeek-R1-Distill-Qwen-1.5B on AMC23: $\alpha$ 1 still fails

**Question**: There is a unique sequence of integers  $a_1,a_2,\cdots,a_{2023}$  such that  $\tan 2023x = \frac{a_1tanx + a_3tan^3x + a_5tan^5x + \cdots + a_{2023}tan^{2023}x}{1 + a_2tan^2x + a_4tan^4x + \cdots + a_{2022}tan^{2022}x}$  whenever  $\tan 2023x$  is defined. What is  $a_{2023}$ ?

#### Response from $\alpha 1$ : <think>

Okay, so I need to find the value of  $a_{2023}$  in the given expression for tan(2023x). The expression is a fraction where the numerator is a sum of odd-powered terms of (tan x) multiplied by coefficients  $a_1, a_3, \ldots, a_{2023}$ , and the denominator is a sum of even-powered terms of (tan x) multiplied by coefficients  $a_2, a_4, \ldots, a_{2022}$ . The problem states that this sequence of integers is unique, so I need to figure out what  $a_{2023}$  is.

 $\underline{\text{Wait}}$ , before jumping into that, I should recall the tangent multiple-angle formulas. I remember that  $\tan(nx)$  can be expressed as a ratio of polynomials in  $\tan x$ , where the numerator is an odd-degree polynomial and the denominator is an even-degree polynomial ...

After carefully considering the expansion of  $\tan(2023x)$  and testing various formulas ... which is the coefficient of  $\tan^{2023}x$  in the numerator, we look at the coefficient for ( k = 2023 ) in the numerator sum:

$$[a_{2023}=(-1)^{2023-1}{2023\choose 2023}=(-1)^{2022}\times 1=1]$$
 Therefore,  $a_{2023}=1.\nn[\n]$ 

Correct Answer: -1

# Successful Example of DeepSeek-R1-Distill-Qwen-7B on Minerva Math: Base fails while α1 passes

**Question**: A particular star has an absolute magnitude M = −7. If this star is observed in a galaxy that is at a distance of 3Mpc, what will its apparent magnitude be?

#### **Response from Base**: <think>

Okay, so I need to find the apparent magnitude of a star given its absolute magnitude and the distance to the galaxy it's in. Hmm, let's see. I remember that absolute magnitude is how bright a star appears at a standard distance, which I think is 10 parsecs. Apparent magnitude, on the other hand, is how bright it appears from Earth, which depends on how far away it actually is. The formula that relates absolute magnitude (M), apparent magnitude (m) ...</think>

To find the apparent magnitude of a star given its absolute magnitude and the distance to the galaxy it is in, we use the distance modulus formula. The formula is:\n\n[ m - M = 5log(d) - 5 ]\n\nwhere (m) is the apparent magnitude...Rounding to one decimal place, the apparent magnitude is approximately 20.4.\n\n[\n 20.4 \n]

### **Response from** α**1**: <think>

Okay, so I need to find the apparent magnitude of a star given its absolute magnitude and the distance to the galaxy it's in. Hmm, I remember that absolute magnitude is how bright a star appears at a standard distance, which is 10 parsecs, right? And apparent magnitude is how bright it appears from Earth, which depends on its actual distance. The formula that relates absolute magnitude (M), apparent magnitude (m), and distance (d) in parsecs is the distance modulus formula. I think it's something like m = M + 5 + 5 log10(d). Wait, let me make sure. Is it plus or minus 5? I think it's plus 5 because as the distance increases, the apparent magnitude increases, meaning it becomes dimmer. So, yes, m = M + 5 log10(d) + 5.

Wait, hold on, no, that's not right...Then, subtract 5:\n\n[ 25.3855 - 5 = 20.3855 ]

So, the apparent magnitude is approximately 20.3855, which rounds to 20.39.\n \nTherefore, the apparent magnitude of the star is 20.39 .

**Correct Answer**: 20.39

## Failure Example of DeepSeek-R1-Distill-Qwen-7B on LiveCodeBench: α1 still fails

**Question**: You are given a 0-indexed string s and a dictionary of words dictionary. You have to break s into one or more non-overlapping substrings such that each substring is present in dictionary. There may be some extra characters in s which are not present in any of the substrings.

Return the minimum number of extra characters left over if you break up s optimally.

#### Example 1:

Input: s = "leetscode", dictionary = ["leet","code","leetcode"]. Output: 1 Explanation: We can break s in two substrings: "leet" from index 0 to 3 and "code" from index 5 to 8. There is only 1 unused character (at index 4), so we return 1.

## Example 2:

Input: s = "sayhelloworld", dictionary = ["hello","world"]. Output: 3 Explanation: We can break s in two substrings: "hello" from index 3 to 7 and "world" from index 8 to 12. The characters at indices 0, 1, 2 are not used in any substring and thus are considered as extra characters. Hence, we return 3.

Constraints:1 <= s.length <= 50, 1 <= dictionary.length <= 50, 1 <= dictionary[i].length <= 50, dictionary[i] and s consists of only lowercase English letters, dictionary contains distinct words

### **Response from** α**1**: <think>

Okay, I need to solve this problem where I have to find the minimum number of extra characters left when breaking a string s into substrings that are all present in a given dictionary. Hmm, let's think about how to approach this. Wait, the problem says that the string must be broken into one or more non-overlapping substrings, each of which is in the dictionary. ... </think>

To solve this problem, we need to break a given string 's' into substrings such that each substring is present in a given dictionary. ...

## ### Solution Code

"'python\n \nclass Solution:\n def minExtraChar(self, s: str, dictionary: List[str]) -> int:\n n = len(s)\n dp = [float('inf')] \* (n + 1)\n dp[0] = 0 # Base case: 0 characters, 0 extra\n \n for i in range(1, n + 1):\n for l in range(1, i + 1):\n start = i - l\n if start < 0:\n continue\n substring = s[start:i]\n if substring in dictionary:\n if dp[start] + (n - i) < dp[i]:\n dp[i] = dp[start] + (n - i)\n\n return dp[n]

### ### Explanation

- **Initialization**: The DP array is initialized to handle the base case where no characters are processed, resulting in zero extra characters. ...
- **Result Extraction**: The final result is found in 'dp[n]', which gives the minimum number of extra characters after processing the entire string. This approach ensures that we efficiently find the optimal way to split the string into valid substrings while minimizing the extra characters, leveraging dynamic programming to explore all possible valid splits.

## Successful Example of QwQ-32B on MATH500: Base fails while α1 passes

**Question**: There are 360 people in my school. 15 take calculus, physics, and chemistry, and 15 don't take any of them. 180 take calculus. Twice as many students take chemistry as take physics. 75 take both calculus and chemistry, and 75 take both physics and chemistry. Only 30 take both physics and calculus. How many students take physics?

## **Response from Base**: <think>

Okay, let's try to tackle this problem step by step. Hmm, so there are 360 people in the school. Out of these, 15 don't take any of the three subjects: calculus, physics, or chemistry. That means 360 - 15 = 345 students take at least one of those subjects.

Alright, let me note down all the given information:

- Total students: 360
- Take all three (calculus, physics, chemistry): 15
- Don't take any: 15
- Take calculus: 180...</think>

Given the problem, we need to determine the number of students taking physics. Here is the step-by-step reasoning:\n\n

- 1. **Total students**: 360
- 2. **Students not taking any subjects**: 15, so students taking at least one subject: (360 - 15 = 345)
- 3. **Students taking calculus (C)**: 180
- 4. **Students taking all three subjects (C, P, Ch)**: 15...Thus, the number of students taking physics is ( 115 ).

### **Response from** α**1**: <think>

Okay, let's try to tackle this problem step by step. Hmm, so there are 360 people in the school. 15 take all three subjects: calculus, physics, and chemistry. Also, 15 don't take any of them. That means the number of students taking at least one of the subjects is 360 - 15 = 345.

Wait, the total number in the union of calculus, physics, and chemistry is 345. Let me note down the given data first:

- Total students: 360
- Take all three (Calculus, Physics, Chemistry): ...</think>

```
So simplifying:
```

```
180 + P + Ch - 180 +15 = P + Ch +15 = 345
```

Hence, P + Ch = 330. Since Ch = 2P, substituting:

```
P + 2P = 3P = 330 → P = 110.
```

All steps check out, and the Venn diagram regions confirm consistency. Therefore, the number of students taking physics is 110 .

**Final Answer**\n 110 .

**Correct Answer**: 110

## Failure Example of QwQ-32B on AIME24: $\alpha$ 1 still fails

Question: Torus T is the surface produced by revolving a circle with radius 3 around an axis in the plane of the circle that is a distance 6 from the center of the circle (so like a donut). Let S be a sphere with a radius 11. When T rests on the outside of S, it is externally tangent to S along a circle with radius  $r_i$ , and when T rests on the outside of S, it is externally tangent to S along a circle with radius  $r_o$ . The difference  $r_i - r_o$  can be written as  $\frac{m}{n}$ , where m and n are relatively prime positive integers. Find m+n.

```
[asy] unitsize(0.3 inch); draw(ellipse((0,0), 3, 1.75)); draw((-1.2,0.1).(-0.8,-0.03)..(-0.4,-0.11)..(0,-0.15)..(0.4,-0.11).. (0.8,-0.03)..(1.2,0.1)); draw((-1,0.04)..(-0.5,0.12)..(0,0.16)..(0.5,0.12).. (1,0.04)); draw((0,2.4)-(0,-0.15)); draw((0,-0.15)-(0,-1.75), dashed); draw((0,-1.75)-(0,-2.25)); draw(ellipse((2,0), 1, 0.9)); draw((2.03,-0.02)-(2.9,-0.4)); [/asy]
```

#### Response from $\alpha 1$ : <think>

Okay, so I need to solve this problem about a torus and a sphere. Let me try to understand the setup first. The torus T is formed by revolving a circle with radius 3 around an axis in the same plane, which is 6 units away from the center of the circle. That makes sense—it's like a donut shape where the tube radius is 3 and the distance from the center of the tube to the center of the torus is 6. Then there's a sphere S with radius 11. The problem says that when the torus rests on the outside of the sphere, it can be externally tangent to the sphere along a circle with radius  $r_i$ , and similarly along another circle with radius  $r_o$ . The question is asking for the difference  $r_i$  -  $r_o$ , expressed as a fraction m/n where m and n are coprime, and then m + n.

Wait, actually, the problem mentions...</think>

Okay, let's start by setting up a coordinate system where the sphere is centered at the origin ((0, 0, 0)) with radius 11. The torus (T) is generated by revolving a circle of radius 3 around an axis that is 6 units away from the center of the generating circle. Let's place the generating circle in the (x)-(y) plane, centered at ((6, 0, 0)). The axis of revolution is the (z)-axis, so the torus's axis is the (z)-axis.

When the torus rests on the outside of the sphere, they are externally tangent along a circle. To find the radii  $(r_i)$  and  $(r_o)$  of these circles of tangency, we need to consider the cross-sectional view in the (x)-(y) plane. Here, the sphere appears as a circle of radius 11 centered at the origin, and the generating circle of the torus is a circle of radius 3 centered at ((6, 0)).

For the torus to be externally tangent to the sphere, the distance between their centers must equal the sum of their radii. However, since the sphere is much larger, the torus must be positioned such that the distance from the sphere's center to the torus's generating circle's center plus the sphere's radius equals the distance from the sphere's center to the farthest point on the torus. ...

Correct Answer: 127