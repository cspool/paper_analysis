# SELF-MOE: TOWARDS COMPOSITIONAL LARGE LAN-GUAGE MODELS WITH SELF-SPECIALIZED EXPERTS

Junmo Kang<sup>∗</sup> Georgia Tech Leonid Karlinsky MIT-IBM Watson AI Lab Hongyin Luo MIT Zhen Wang UCSD

Jacob Hansen MIT James Glass MIT David Cox MIT-IBM Watson AI Lab Rameswar Panda MIT-IBM Watson AI Lab

Rogerio Feris MIT-IBM Watson AI Lab Alan Ritter Georgia Tech

# ABSTRACT

We present Self-MoE, an approach that transforms a monolithic LLM into a compositional, modular system of self-specialized experts, named MiXSE (MiXture of Self-specialized Experts). Our approach leverages self-specialization, which constructs expert modules using self-generated synthetic data, each equipping a shared base LLM with distinct domain-specific capabilities, activated via selfoptimized routing. This allows for dynamic and capability-specific handling of various target tasks, enhancing overall capabilities, without extensive humanlabeled data and added parameters. Our empirical results reveal that specializing LLMs may exhibit potential trade-offs in performances on non-specialized tasks. On the other hand, our Self-MoE demonstrates substantial improvements (6.5%p on average) over the base LLM across diverse benchmarks such as knowledge, reasoning, math, and coding. It also consistently outperforms other methods, including instance merging and weight merging, while offering better flexibility and interpretability by design with semantic experts and routing. Our findings highlight the critical role of modularity, the applicability of Self-MoE to multiple base LLMs, and the potential of self-improvement in achieving efficient, scalable, and adaptable systems.

# 1 INTRODUCTION

The remarkable success of Large Language Models (LLMs) has been largely attributed to their generalist nature, allowing them to perform a wide variety of tasks [\(Brown et al., 2020;](#page-10-0) [Touvron et al.,](#page-15-0) [2023;](#page-15-0) [Jiang et al., 2023;](#page-12-0) [Team et al., 2024\)](#page-14-0). Predominantly designed as monolithic architectures, these models rely extensively on large-scale data to embed generalized language capabilities across vast parameter spaces. While effective, this monolithic architecture, as illustrated in Figure [1,](#page-1-0) inherently suffers from significant drawbacks such as inefficiency in scaling [\(Zhang et al., 2024;](#page-16-0) [Wan](#page-15-1) [et al., 2024\)](#page-15-1), susceptibility to forgetting previously learned information when adapted to specialized tasks [\(Kotha et al., 2024;](#page-12-1) [Huang et al., 2024\)](#page-12-2), and a lack of transparency which leads to the black-box nature [\(Zhao et al., 2023\)](#page-16-1).

Meanwhile, the increasing demand to handle domain-specific or expert-level tasks has highlighted the need for specialization of LLMs [\(Cheng et al., 2024;](#page-10-1) [Ling et al., 2023;](#page-13-0) [Feng et al., 2024\)](#page-11-0). However, effective tuning often relies on high-quality, human-annotated data, which is costly and challenging to scale [\(Kang et al., 2023\)](#page-12-3), especially in specialized domains where expertise is scarce and valuable [\(Wu et al., 2023\)](#page-15-2). Self-specialization [\(Kang et al., 2024\)](#page-12-4) offers a promising alternative, aligning models with self-generated synthetic data. While this technique has proven effective in cross-task generalization within a target expert domain, we posit that it may compromise performance in areas outside the target domain.

<sup>∗</sup>Correspondence to junmo.kang@gatech.edu

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: Concept of Self-MoE, illustrating the transformation from a monolithic LLM to a compositional system, MiXSE, without extensive resources and addition of significant parameters. MiXSE distinguishes itself from traditional MoEs and other models in post-training, lightweight semantic experts, and/or self-generated synthetic data. The results showcase MiXSE's improved capabilities over the base LLM (e.g., Gemma-7B) across all domains, unlike the knowledge-specialized LLM that compromises other capabilities.

In this paper, we explore the following question: *How can we build compositional LLMs that enjoy versatile expertise, while using minimal resources?* We introduce Self-MoE (Figure [1\)](#page-1-0), an approach that transforms a monolithic model into a compositional [\(Zaharia et al., 2024\)](#page-16-2) system, called MiXSE (MiXture of Self-specialized Experts). This approach differs from prior MoE work using LoRA [\(Hu](#page-12-5) [et al., 2022\)](#page-12-5), which either relies on human-labeled data [\(Wu et al., 2024\)](#page-15-3) or assumes the existence of trained modules [\(Huang et al., 2023;](#page-12-6) [Muqeeth et al., 2024\)](#page-14-1). Instead, our Self-MoE constructs individual lightweight expert modules from scratch using synthetic data, inspired by the concept of self-specialization. Each module is integrated with a shared base LLM, and the entire system is enhanced by a self-optimized routing mechanism. In contrast to monolithic models, which often suffer from forgetting issues when adapted or merged under fixed, static parameters, our modular design preserves the integrity and semantics of each expert. This allows for dynamic, precise handling of various target domain tasks, boosting the model's overall capability, adaptability, and interpretability.

Through extensive empirical studies conducted across a variety of popular domains, including knowledge, reasoning, math, and coding, we find that specialization often comes with trade-offs, typically degrading performance in non-targeted domains. However, our Self-MoE demonstrates substantial overall improvements over a base LLM across all target domains without compromising performance on other tasks. Notably, the compositional nature of our MiXSE appears to exploit synergies among experts, even outperforming all individual specialized experts.

Moreover, MiXSE clearly surpasses other strong baselines such as instance merging and weight merging, under similar settings, while offering better flexibility and interpretability. Detailed analyses highlight the critical role of the routing mechanism and the contribution of semantic experts in achieving these results. Our interpretable visualizations of routing distributions further elucidate how tasks are dynamically allocated to the most relevant experts. Lastly, we further validate that there are no issues related to forgetting unlike monolithic baselines, and that our approach can be applied to various model families and sizes. In summary, our key contributions are as follows:

- We highlight the inherent limitations of monolithic model specialization, where focusing on a specific capability often comes at the cost of degrading performance in other domains.
- We propose Self-MoE, which allows a base, monolithic LLM to upgrade into a modular system of lightweight, self-specialized experts, without requiring extensive human supervision, compute resources, or overhead in active parameters.

#### **Self-Specialization**

#### MiXSE (MiXture of Self-Specialized Experts)

![](_page_2_Figure_2.jpeg)

<span id="page-2-0"></span>![](_page_2_Figure_3.jpeg)

Figure 2: Overview of the **Self-MoE** approach to building a compound system of specialized experts and a router in a self-improving manner. In the Self-Specialization phase (left side), the base LLM is aligned with self-generated synthetic data for each target specialization, producing lightweight expert modules. The right side shows MiXSE where each self-specialized expert is dynamically engaged based on the decisions of the self-optimized router.

We provide comprehensive experiments and analyses across a range of benchmarks, where Self-MoE demonstrates consistent improvements with an average of 6.5%p across domains over a base LLM, outperforming various baselines. Our ablation studies validate the impact of modularity, routing strategies, and the use of self-generated synthetic data. Moreover, our analyses explore routing distributions, forgetting issues, and the applicability of our approach to five different base LLMs.

#### <span id="page-2-1"></span>2 Problem Statement

The primary focus of this work is on self-improving LLMs' target capabilities on the fly, specifically under settings constrained by minimal resources and without the addition of significant parameters. Traditional LLMs, which are generally monolithic, require expensive human-labeled data to be better specialized, thereby limiting their adaptability and scalability when resources are constrained. We hypothesize that a modular, compositional model utilizing self-generated synthetic data for self-improvement can dramatically improve specific target capability, adaptability, and interpretability while reducing dependency on expensive human-annotated datasets.

Specifically, given a base LLM  $\Theta_0$  and a minimal set of seed data (e.g., 100) for each of the target capabilities  $\{T_i\}_{i=1}^n$  (e.g., knowledge, math), our goal is to transform  $\Theta_0$  into an enhanced compositional model  $\Theta_{comp}$  where n target expert modules  $\{\Delta\Theta_i\}_{i=1}^n$  are effectively integrated. Formally, the Self-MoE transformation function is defined as:

$$f_{trans}: (\Theta_0, \{T_i\}_{i=1}^n) \to \Theta_{comp} = \Theta_0 \cup \{\Delta\Theta_i\}_{i=1}^n$$

Here, under our problem setting, the number of parameters of  $\Theta_0$  and  $\Theta_{comp}$  should not be significantly different, necessitating that the expert modules  $\Delta\Theta_i$  be lightweight (i.e., LoRA (Hu et al., 2022)). The available seed data are limited but can be reasonably collected (e.g., 100). Importantly, we do not assume the availability of larger/teacher models at one's hand; instead, we aim to develop a method that enables self-improvement and is designed to be universally applicable.

# 3 METHOD: SELF-MOE

In this section, we describe Self-MoE, our proposed framework designed to build a compositional model in which specialized expert modules and a routing component are learned in a self-training manner to cooperate effectively. At a high level, Self-MoE decomposes the monolithic structure

of a base LLM into a dynamic mixture of self-specialized units, each equipped with distinct target capabilities. This section outlines the overall pipeline and architecture of Self-MoE, illustrated in Figure 2, which details both the self-specialization of individual target expert modules and their integration to form a compositional system, MiXSE (MiXture of Self-specialized Experts).

#### 3.1 BUILDING EXPERT MODULES THROUGH SELF-SPECIALIZATION

The first step of Self-MoE is creating specialized modules  $\{\Delta\Theta_i\}_{i=1}^n$  for each target expertise, while adhering to the desiderata discussed in Section 2. That is, the modules should be lightweight and self-improving. We employ the concept of self-specialization (Kang et al., 2024) where a base LLM is aligned with self-generated synthetic data for target specialization, resulting in lightweight LoRA (Hu et al., 2022) experts.

**Targeted Generation.** Self-specialization involves generating synthetic instruction-response data  $D_i = \{(\mathit{inst}_i^{(1)}, \mathit{resp}_i^{(1)}), (\mathit{inst}_i^{(2)}, \mathit{resp}_i^{(2)}), ...\}$  tailored to each target domain  $T_i$ . We ensure the data is both diverse and highly relevant to the specialized tasks/domains each module will address. The generation includes the following steps:

- (1) Seed Construction: First, given a target  $T_i$  identified, we prepare a small number of seed examples (e.g., 100) that capture essential characteristics and scenarios relevant to each target domain  $T_i$ . While we exploit existing datasets for the purpose of demonstration, we posit manual annotation for such a small number should be reasonable in real-world applications. These seeds serve as the foundational dataset from which synthetic variations are generated.
- (2) Instruction Brainstorming: Once the seed examples are established, the next step is to diversify the range of instructions (and corresponding input contexts) through a brainstorming process. Specifically, we prompt<sup>1</sup> a base LLM  $\Theta_0$  to create new instructions following sequences of seed instructions given in-context.
- (3) **Response Generation**: The final step involves generating corresponding responses for the newly created instructions. We use seed instruction-response pairs as in-context demonstrations to extract latent relevant knowledge from  $\Theta_0$ .

Self-Align with LoRA With each specialized synthetic data  $D_i$  in place, we now proceed with the self-alignment of  $\Theta_0$  to induce specialization, separately producing lightweight expert components  $\Delta\Theta_i$ . Note that  $D_i$  are self-generated by  $\Theta_0$  and used to specialize the same  $\Theta_0$  using an adapter module  $\Delta\Theta_i$ , resulting in an specialized model  $\Theta_{spec} = \Theta_0 + \Delta\Theta_i$ . Specifically, we utilize Low-Rank Adaptation (LoRA) (Hu et al., 2022), which integrates additional trainable parameters that are specific to each domain  $T_i$  while keeping  $\Theta_0$  intact. Within the corresponding  $\Theta$ , we define  $\theta$  as the weights at a certain layer where LoRA is attached. Let  $\theta_{spec} \in \mathbb{R}^{d \times k}$  be updated weights at a specific LoRA layer which can be decomposed as:

$$\theta_{spec} = \theta_0 + \Delta \theta_i$$
$$= \theta_0 + \theta_{B_i} \theta_{A_i}$$

where  $\theta_{B_i} \in \mathbb{R}^{d \times rank}$  and  $\theta_{A_i} \in \mathbb{R}^{rank \times k}$ , with  $rank \ll \min(d, k)$ . The forward pass becomes:

$$h = \theta_{spec} x = \theta_0 x + \theta_{B_i} \theta_{A_i} x$$

This applies to all LoRA layers, and only  $\Delta\Theta_i = \{\Delta\theta_i^{(1)}, \Delta\theta_i^{(2)}, ...\}$  is updated during training using  $D_i$ . As a whole, this process of self-specialization can be defined as producing an expert module  $\Delta\Theta_i$  for the *i*-th target along with the corresponding synthetic data  $D_i$  (Left in Figure 2):

$$f_{ss}:(\Theta_0,T_i)\to(\Delta\Theta_i,D_i)$$

We iterate this process for each target domain, focusing on knowledge, reasoning, math, and coding.

# 3.2 MIXTURE OF SELF-SPECIALIZED EXPERTS

After each expert module is individually specialized through the self-specialization process, they are integrated into a compound system  $\Theta_{comp}$ , MiXSE (MiXture of Self-specialized Experts). MiXSE

<span id="page-3-0"></span><sup>&</sup>lt;sup>1</sup>The prompts can be found in Table 9-12 in Appendix.

is designed to leverage the distinct capabilities of each module, orchestrating their cooperation to handle diverse tasks dynamically and efficiently. To achieve this benefit, a router module  $\theta_r$  is also incorporated, which analyzes each input token to dynamically route to the most appropriate expert module based on the task at hand.

Specifically, within each layer, the output h for each input x is calculated by combining the contributions of the selected expert modules  $\Delta\theta_i$ , weighted by their relevance as determined by the router:

$$h = \theta_0 x + \sum_{i=1}^n \alpha_i \Delta \theta_i x$$
$$= \theta_0 x + \sum_{i=1}^n \alpha_i \Delta \theta_{B_i} \theta_{A_i} x$$

where  $\alpha$  represents a set of weights computed by the router (i.e., a linear layer)  $\theta_r \in \mathbb{R}^{n \times k}$ .

$$\alpha = \text{top-k}(\text{softmax}(\theta_r x))$$

Note that we only take top-k probabilities and mask out the others to efficiently reduce computation. In essence, this also allows the pre-trained base weights  $\theta_0$  to be sufficiently able to contribute, mitigating potential issues of over-specialization such as forgetting or diminished generalizability. The router  $\theta_r$  is a linear layer, shared across all LoRA layers, and is trained using the aggregated self-generated data  $D = \{D_i\}_{i=1}^n$  to learn how to optimally select modules for a given task:

$$L(\theta_r) = -\mathbb{E}_{(inst, resp) \sim D}[log P_{\Theta_0}(resp \mid inst; \theta_r, \{\Delta \Theta_i\}_{i=1}^n)]$$

Here, we solely optimize the router to preserve the explicit semantic distinction of expert modules.

# 4 EXPERIMENTS AND RESULTS

**Datasets.** We evaluate Self-MoE across diverse domains categorized into knowledge, reasoning, math, and coding: MMLU (0- & 5-shot) (Hendrycks et al., 2021a), BBH (3-shot) (Suzgun et al., 2022), GSM8K (8-shot) (Cobbe et al., 2021), and HumanEval (0-shot) (Chen et al., 2021), respectively. For MMLU, we primarily employ the 0-shot setting unless otherwise specified, based on established observations (Dettmers et al., 2023; Lin et al., 2024) that tuning yields only marginal effects in the 5-shot setting for this task. To test generalization (Section 4.4), we additionally evaluate on MATH (4-shot) (Hendrycks et al., 2021b), MBPP (3-shot) (Austin et al., 2021), NaturalQuestions (5-shot) (Kwiatkowski et al., 2019), TriviaQA (5-shot) (Joshi et al., 2017), Hellaswag (0-shot) (Zellers et al., 2019), PIQA (0-shot) (Bisk et al., 2020), and TruthfulQA (0-shot) (Lin et al., 2022).

**Baselines.** To assess the effectiveness of Self-MoE, we compare performance against several baselines that use the same number of active parameters during inference:

- Four Self-Specialized Models (Kang et al., 2024): Trained on self-generated synthetic data for individual domains: knowledge, reasoning, math, and coding.
- Instance Merging (Multi-Task Tuning) (Chung et al., 2024): Leverages the aggregated synthetic data generated by self-specialization to train a model capable of handling multiple tasks.
- TIES (Yadav et al., 2023), DARE (Yu et al., 2024): Advanced weight merging methods integrating multiple expert strengths into a unified model.

We also contextualize these results with computationally intensive methods reported in the literature, despite indirect comparisons: BTM (Li et al., 2022), Sparse Upcycling (Komatsuzaki et al., 2023), BTX (Sukhbaatar et al., 2024), GLAN (Li et al., 2024a), Orca (Mitra et al., 2023), and Merlinite (Sudalairaj et al., 2024) in Appendix D.1.

**Implementation Details.** We adopt Gemma-7B (Team et al., 2024) as a base LLM for our main experiments, and additionally apply Self-MoE to various models, such as LLaMA-2 7B & 13B (Touvron et al., 2023), Mistral 7B (Jiang et al., 2023), and LLaMA-3 8B (AI@Meta, 2024) in Section 4.5. We use 100 seeds to generate 5K synthetic data for each domain, resulting in 20K data. Each LoRA module contributes less than 0.3% to the parameters of the base model, and the router's parameters are negligible, resulting in the added parameters of MiXSE amounting to only about 1%.

<span id="page-5-0"></span>Table 1: Main results. All models are built upon the same base LLM, Gemma-7B, taking self-improving approaches and having the same active parameters during inference. Corresponding aligned performances of self-specialization are underscored. Each column's best performance is highlighted in bold, while the gains achieved by our MiXSE over the base LLM are indicated.

| Method                                                                      | Active<br>Params                                 | Knowledge<br>(MMLU)          | Reasoning<br>(BBH)           | Math<br>(GSM8K)                     | Coding<br>(HumanEval)               | Avg.                         |
|-----------------------------------------------------------------------------|--------------------------------------------------|------------------------------|------------------------------|-------------------------------------|-------------------------------------|------------------------------|
| Base LLM                                                                    | 7B                                               | 58.4                         | 56.1                         | 42.5                                | 34.1                                | 47.8                         |
| Specialized LLM for Each Capability                                         |                                                  |                              |                              |                                     |                                     |                              |
| Knowledge Self-Spec. Reasoning Self-Spec. Math Self-Spec. Coding Self-Spec. | 7B + 0.3%<br>7B + 0.3%<br>7B + 0.3%<br>7B + 0.3% | 64.0<br>60.1<br>59.3<br>57.2 | 41.7<br>60.2<br>58.9<br>57.2 | 40.5<br>41.0<br><u>50.0</u><br>46.0 | 28.0<br>28.7<br>36.0<br><u>37.2</u> | 43.6<br>47.5<br>51.1<br>49.4 |
| Merging Methods                                                             |                                                  |                              |                              |                                     |                                     |                              |
| Instance Merging TIES Merging DARE Merging                                  | 7B + 0.3%<br>7B + 0.3%<br>7B + 0.3%              | 62.6<br>63.7<br>37.7         | 57.6<br>56.3<br>59.6         | <b>53.5</b> 38.5 45.0               | 36.0<br>32.9<br>34.8                | 52.4<br>47.9<br>44.3         |
| MiXSE (Ours)                                                                | 7B + 0.3%                                        | <b>65.6</b> ↑ 7.2            | <b>61.1</b> ↑ 5.0            | <b>52.5</b> ↑ 10.0                  | <b>37.8</b> ↑ 3.7                   | <b>54.3</b> ↑ 6.5            |

#### 4.1 Main Results

In Table 1, we showcase comparative benchmark results of various approaches across four specialized domains: knowledge, reasoning, math, and coding. All baselines use self-generated synthetic data based on the same Base LLM, Gemma-7B, and LoRA for tuning to ensure fair comparisons.

First, we confirm self-specialization markedly enhances target-specific expertise, compared to the base LLM. For instance, we can see substantial gains from corresponding specialized models (e.g., Knowledge Self-Spec. in the knowledge domain): 58.4 to 64.0 in knowledge, 56.1 to 60.2 in reasoning, and so on. However, this focused improvement sometimes comes at the cost of reduced performance in non-targeted areas, as evidenced by the drop in scores for the Knowledge Self-Spec. model in reasoning, math, and coding. This trade-off highlights the inherent limitation of overspecialization. In contrast, our MiXSE, demonstrates consistent improvements across all domains, due to its modular, compositional architecture that makes use of dynamic routing to leverage optimal experts. Surprisingly, it even outperforms all corresponding specialized models, indicating that it effectively synergizes the strengths of each specialization.

In comparison with other static merging methods like Instance Merging, TIES, and DARE, MiXSE stands out for its superior adaptability. While they attempt to combine the strengths of different specialization areas into a single model, they lack the dynamic flexibility that MiXSE offers. Notably, simple instance merging (i.e., multi-task tuning), though effective in enhancing the base LLM across domains, still falls short of achieving the superior average performance of 54.3 seen with MiXSE. This validates the advantages of dynamic expert integration in a compositional system.

## 4.2 ABLATION STUDY

Now that we have verified the effectiveness of MiXSE as a whole, we evaluate the impact of different configurations and components of the system, presented in Table 2. The configurations vary in terms of routing strategies and integration of experts, offering insights into the contributions of each element to the system's overall effectiveness.

We start by examining the Top-k routing strategy, which plays a crucial role in our model. Our findings show that both the Top-1 and Top-2 expert configurations deliver the best performance. This suggests that identifying and leveraging the most relevant expert for a given task is typically sufficient and most effective. On a side note, the similar performances of the different configurations may highlight the robustness of our method. Given the similar performances, we prefer the Top-1 expert setup for better efficiency.

Interestingly, the results also indicate a drop in performance when using All Experts. This can be attributed to that involving all experts regardless of their relevance can introduce noise and dilute

<span id="page-6-0"></span>Table 2: Analysis and ablation of the router in our MiXSE. Configurations vary to investigate the optimal number of experts used, to verify the possibility of self-learning for the router, and to see the importance of semantic distinctions among experts within the compositional system.

| Configuration                                         | Knowledge<br>(MMLU)   | Reasoning<br>(BBH)    | Math<br>(GSM8K)             | Coding<br>(HumanEval)       | Avg.              |
|-------------------------------------------------------|-----------------------|-----------------------|-----------------------------|-----------------------------|-------------------|
| Base LLM                                              | 58.4                  | 56.1                  | 42.5                        | 34.1                        | 47.8              |
| Top-k Routing                                         |                       |                       |                             |                             |                   |
| w/ Top-1 Expert<br>w/ Top-2 Experts<br>w/ All Experts | <b>65.6</b> 65.5 65.4 | <b>61.1</b> 60.9 58.9 | 52.5<br>52.5<br><b>54.0</b> | 37.8<br><b>38.4</b><br>33.5 | <b>54.3 53.</b> 0 |
| Random Routing                                        |                       |                       |                             |                             |                   |
| w/o Self-Optimized Router                             | 59.9                  | 58.5                  | 48.0                        | 36.6                        | 50.8              |
| Experts & Router Joint Training                       |                       |                       |                             |                             |                   |
| w/o Semantic Experts                                  | 64.5                  | 58.1                  | 46.0                        | 33.5                        | 50.5              |

the specific contributions of the most pertinent experts. Additionally, involving more experts than necessary can increase computational overhead.

Furthermore, employing Random Routing serves as a useful setup to highlight the effectiveness of strategic expert selection of our Self-Optimized Router, which is a key component of our MiXSE. We observe that the performance significantly decreases under this configuration, highlighting the router's role in dynamically tailoring the selection of experts according to the specific requirements of each task. The router's ability to discern and activate the most suitable experts based on the context is critical for optimizing performance. Notably, this ability is learned by relying on a very small amount of seed data.

Another interesting finding comes from the configuration where experts and the router are jointly trained, which means that the semantic distinctions among experts may be diluted. This setup substantially decreases performance relative to scenarios where the router and experts are optimized independently. This decline validates that semantic experts play a crucial role in enhancing the system's capability to handle tasks requiring specific expertise, while offering better interpretability (Section 4.3).

## <span id="page-6-1"></span>4.3 ROUTING ANALYSIS

Understanding how MiXSE allocates tasks to its various experts is crucial for gauging its interpretability. By analyzing the routing distributions across four distinct domains, we aim to see whether the system matches queries to the most suitable experts. Figure 3 presents the routing distributions used to solve each benchmark, where the weights are averaged across tokens and layers within individual tasks.

We first observe that the MiXSE's router effectively selects the correct expert for each corresponding target. This is evident from the impressive alignment between tasks and the experts chosen by the router; for example, the knowledge expert predominantly handles knowledge tasks, while the coding expert is routed coding tasks. This highlights the router's ability to learn and apply this routing automatically.

![](_page_6_Figure_8.jpeg)

<span id="page-6-2"></span>Figure 3: Routing analysis that shows routing distributions over four domains for each benchmark, averaging the weights across tokens within individual tasks.

cally and consistently, making the system's decisions interpretable and trustworthy.

Beyond the direct matching of tasks to domain-specific experts, the router also demonstrates its ability to exploit synergies between different areas of expertise. For instance, the reasoning expert is

<span id="page-7-2"></span>Table 3: Investigation on generalization and a forgetting issue of Self-MoE. Non-Target (In-Expertise) indicates where MiXSE does not directly specialize using seed data directly while relevant to targets. Non-Target (Out-of-Expertise) refers to irrelevant cases.

| Category           | Benchmark                     | Base<br>LLM | Instance<br>Merging | MiXSE |  |  |  |  |  |
|--------------------|-------------------------------|-------------|---------------------|-------|--|--|--|--|--|
| Target             |                               |             |                     |       |  |  |  |  |  |
| Academic Knowledge | MMLU                          | 58.4        | 62.6                | 65.6  |  |  |  |  |  |
| Reasoning          | BBH                           | 56.1        | 57.6                | 61.1  |  |  |  |  |  |
| Math               | GSM8K                         | 42.5        | 53.5                | 52.5  |  |  |  |  |  |
| Coding             | HumanEval                     | 34.1        | 36.0                | 37.8  |  |  |  |  |  |
| Target Av          | erage                         | 47.8        | 52.4                | 54.3  |  |  |  |  |  |
|                    | Non-Target (In-Expertise)     |             |                     |       |  |  |  |  |  |
| Math               | MATH                          | 20.7        | 15.3                | 21.4  |  |  |  |  |  |
| Coding             | MBPP                          | 37.8        | 37.6                | 39.6  |  |  |  |  |  |
| Λ                  | Non-Target (Out-of-Expertise) |             |                     |       |  |  |  |  |  |
| World Knowledge    | Natural Questions             | 24.2        | 22.3                | 24.5  |  |  |  |  |  |
| world Knowledge    | TriviaQA                      | 63.9        | 58.6                | 62.5  |  |  |  |  |  |
| Commonsense        | Hellaswag                     | 80.6        | 78.0                | 80.7  |  |  |  |  |  |
| Commonsense        | PIQA                          | 81.1        | 80.1                | 81.2  |  |  |  |  |  |
| Safety             | TruthfulQA                    | 44.7        | 42.2                | 44.3  |  |  |  |  |  |
| Non-Target         | 50.4                          | 47.7        | 50.6                |       |  |  |  |  |  |

frequently involved in tasks across the knowledge, math, and coding, reflecting the system's compositional use of expertise. This explains the reason for MiXSE's superior performances across all domains even beyond all specialized modules in Table 1.

#### <span id="page-7-0"></span>4.4 GENERALIZABILITY TEST

While Self-MoE has shown clear benefits in target benchmarks such as MMLU, BBH, GSM8K, and HumanEval, one may be curious about its generalizability to non-targets, or concerned with the potential issues of specialization such as forgetting. In Table 3, we conduct an investigation using non-targeted benchmarks that were not utilized in building MiXSE.

On MATH and MBPP benchmarks, which can be considered highly relevant to target benchmarks, GSM8K and HumanEval, we find our Self-MoE can still improve over the base LLM even though they were not directly targeted in our training regime. This finding supports the generalizability of the Self-MoE approach.

Concerning the potential side effect of forgetting, we extend our testing to include domains such as world knowledge, common sense, and safety, which are rarely associated with the targets directly. Our experiments show that overall, there are rarely meaningful performance drops when applying our Self-MoE. Only a minor drop is observed with MiXSE in TriviaQA, but this is substantially less than in the case of instance merging. This suggests our approach almost maintains existing knowledge for non-targets while significantly boosting target performances, unlike monolithic baselines.

## <span id="page-7-1"></span>4.5 APPLICABILITY TO OTHER BASE LLMS

Following the successful demonstration of our Self-MoE approach based on Gemma-7B, we now present Figure 4 where we apply Self-MoE to other base LLMs beyond Gemma-7B. We use diverse model variants including LLaMA-2 7B & 13B, Mistral 7B, and LLaMA-3 8B. Our findings suggest that our approach improves all models regardless of the model family, size, and level of base performance. This is significant as it might imply that one can take any monolithic

![](_page_7_Figure_9.jpeg)

<span id="page-7-3"></span>Figure 4: Results of Self-MoE w/ other LLMs.

model to enjoy a free upgrade to a compositional system that offers better effectiveness, flexibility, and interpretability.

## 4.6 IMPACT OF THE NUMBER OF SYNTHETIC DATA

Figure 5 illustrates the impact of scaling self-generated synthetic data for Self-MoE. As the data scales from 0 to 20K, our MiXSE model demonstrates substantial and consistent improvements over the base one in average performance across domains, suggesting the scalable potential of Self-MoE. Instance Merging, serving as a strong baseline, also benefits from increased data, but the gains progress at a slower rate, as evidenced by linear trendlines. This reflects the inefficiency of the static merging scheme, which, being monolithic, suffers from trade-offs in knowledge gains and forgetting.

![](_page_8_Figure_2.jpeg)

<span id="page-8-0"></span>Figure 5: Analysis with the varied sizes of self-generated synthetic data for Self-MoE.

#### 4.7 SCALING THE NUMBER OF EXPERTS

In Table 4, we present the results of MiXSE composed of varying numbers of experts, with experts added progressively one at a time in the order of knowledge, reasoning, math, and coding. The results indicate that starting with the knowledge expert, which initially exhibits a performance trade-off, subsequent additions of reasoning, math, and coding experts consistently enhance overall performance.

<span id="page-8-1"></span>Table 4: Scaling the number of experts. K: Knowledge expert. R: Reasoning expert. M: Math expert. C: Coding expert.

| # Experts    | Knowledge<br>(MMLU) | Reasoning<br>(BBH) | Math<br>(GSM8K) | Coding<br>(HumanEval) | Avg. |
|--------------|---------------------|--------------------|-----------------|-----------------------|------|
| 0 (Base LLM) | 58.4                | 56.1               | 42.5            | 34.1                  | 47.8 |
| 1 (K)        | 64.0                | 41.7               | 40.5            | 28.0                  | 43.6 |
| 2 (K+R)      | 65.8                | 58.0               | 43.0            | 32.3                  | 49.8 |
| 3 (K+R+M)    | 62.7                | 61.5               | 54.5            | 32.9                  | 52.9 |
| 4 (K+R+M+C)  | 65.6                | 61.1               | 52.5            | 37.8                  | 54.3 |

This highlights the compositional MiXSE's advantage of adaptability and modularity.

#### 4.8 DISCUSSION ON THE OVERHEAD OF SELF-MOE

One possible concern in adapting LLMs into compositional systems using Self-MoE is the potential introduction of overhead. Here, we discuss this aspect in detail, emphasizing that the additional overhead of Self-MoE is minimal while yielding significant performance gains. Essentially, the expert modules in Self-MoE are lightweight LoRA modules, contributing only about 1% additional parameters (total) for four experts, as detailed in Table 5 (Total Params). These experts are sparsely activated, which maintains low active parameters (7B + 0.3%) during inference, thus efficiently minimizing inference overhead. In contrast, traditional MoE models like Mixtral (Jiang et al., 2024) and BTX (Sukhbaatar et al., 2024) typically employ a feedforward network (FFN) layer for each expert, resulting in a significant proportional increase in total parameters as the number of experts grows, as indicated in Table 5, which demands much more memory for model loading. The design choice in Self-MoE leads to better scalability and resource efficiency, especially when the number of experts is scaled to incorporate numerous domains of expertise.

## 5 RELATED WORK

To offer a broader perspective, Table 5 presents a comprehensive summary of various models that, while relevant, are not directly comparable. For further discussions and a more detailed comparison, please refer to Appendix D.1.

Combination of Experts. There have been numerous efforts to combine the strengths of multiple models or modules. The Mixture of Experts (MoE) models such as Switch Transformer (Fedus et al., 2022), GLAM (Du et al., 2022), and Mixtral (Jiang et al., 2024) exemplify this, dynamically allocating tasks based on the expertise of each component for better efficiency and scalability. These models contrast with ours by not prioritizing lightweight experts, resulting in a larger model with more parameters. Unlike their experts implicitly learned during pre-training, Self-MoE explicitly creates semantic experts for targeted improvements.

<span id="page-9-0"></span>Table 5: Comprehensive summary of relevant models for references. Detailed discussions are provided in Appendix D.1.

| Method                          | Total<br>Params | Active<br>Params    | Compos-<br>itional | Semantic<br>Experts | Light-<br>weight | Data &<br>Resource<br>-Efficient | w/o<br>Teacher<br>& Labels |
|---------------------------------|-----------------|---------------------|--------------------|---------------------|------------------|----------------------------------|----------------------------|
| Base LLM                        |                 |                     |                    |                     |                  |                                  |                            |
| Gemma 7B                        | 7B              | 7B                  | ×                  | -                   | -                | _                                | -                          |
| LLaMA-2 70B                     | 70B             | 70B                 | ×                  | -                   | -                | -                                | -                          |
| Mixtral 8x7B                    | 47B             | 13B                 | ~                  | ×                   | ×                | -                                | -                          |
| Pre-training Methods            |                 |                     |                    |                     |                  |                                  |                            |
| Branch-Train-Merge (4x7B)       | <24B            | 11.1B               | ~                  | <b>~</b>            | ×                | ×                                | ~                          |
| Sparse Upcycling (4x7B)         | <24B            | 11.1B               | ~                  | ~                   | ×                | ×                                | ~                          |
| Branch-Train-Mix (4x7B)         | <24B            | 11.1B               | ~                  | ~                   | ×                | ×                                | ~                          |
| MoE w/ LoRA                     |                 |                     |                    |                     |                  |                                  |                            |
| PHATGOOSE                       | <4B             | >3B                 | ~                  | ~                   | ~                | ×                                | ×                          |
| MOLE                            | -               | -                   | ~                  | ~                   | ~                | ×                                | ×                          |
| Distillation from Larger Models |                 |                     |                    |                     |                  |                                  |                            |
| GLAN 7B (w/ GPT-4)              | 7B              | 7B                  | ×                  | -                   | -                | ×                                | ×                          |
| Orca-2 7B (w/ GPT-4)            | 7B              | 7B                  | ×                  | -                   | -                | ×                                | ×                          |
| Merlinite 7B (w/ Mixtral 8x7B)  | 7B              | 7B                  | ×                  | -                   | -                | ×                                | ×                          |
| Self-Improving                  |                 |                     |                    |                     |                  |                                  |                            |
| Ours                            | 7B + 1%         | $7\mathrm{B}+0.3\%$ | ~                  | ~                   | ~                | <b>~</b>                         | ~                          |

Another relevant area is merging, involving the weighted averaging of multiple models to form a single, aggregated model (Wortsman et al., 2022; Matena & Raffel, 2022; Ilharco et al., 2023; Jin et al., 2023). One of the leading methods, TIES (Yadav et al., 2023) tackles conflicts and parameter inconsistencies among models. DARE (Yu et al., 2024) further reduces the redundancy of parameters. However, these methods are fundamentally static in that they operate with fixed parameters once merged, which may lead to interference, lacking the dynamic flexibility that MiXSE offers.

There exist notable recent MoE models that similarly explore the utilization of semantic experts, albeit in distinct contexts (Gururangan et al., 2022; Wu et al., 2024; Muqeeth et al., 2024; Sukhbaatar et al., 2024). MOLE relies on human-labeled data, and PHATGOOSE assumes the availability of existing expert models trained by external creators and necessitates additional training for a router on the creators' side. DEMix and BTX rely on extensive pre-training, demanding significant resources, yet it as a pre-trained model holds the potential to complement our self-training approach. Unlike MOLE and PHATGOOSE, our Self-MoE framework creates experts and a router from scratch through self-improvement, while using minimal resources, as contrasted to DEMix and BTX.

**Self-Improvement and Specialization of LLMs.** The pursuit of enhancing the capabilities of LLMs often revolves around an instruction-tuning scheme, which can significantly boost crosstask generalizability (Ouyang et al., 2022; Su et al., 2022; Mishra et al., 2022; Wei et al., 2022). Due to the bottlenecks of expensive annotation costs which lead to limited scalability, the self-training concept (Luo, 2022) has gained attention from the community, where LLMs are aligned with automatically self-generated synthetic instructions (Wang et al., 2023; Sun et al., 2023; Li et al., 2024b). These are distinguished from distillation techniques (Hinton et al., 2015; Kang et al., 2023), which assume a stronger teacher model (Mitra et al., 2023; Li et al., 2024a; Sudalairaj et al., 2024), limiting their applicability.

With the growing need to adapt generalist models to specific domains, Kang et al. (2024) adopts the self-training for specialization, tackling that general instruction tuning is rarely effective in expert domains. While this work lays a foundation for enhancing specialized expertise with minimal resources, we recognize inherent trade-offs in a monolithic structure, such as performance compromises outside specialized domains. Conversely, our Self-MoE achieves uncompromising multiple expertise with a modular approach without extensive resources and adding many parameters.

## 6 Conclusion

In this study, we proposed Self-MoE to build compositional LLMs with self-specialized experts, MiXSE, to enhance targeted capabilities, adaptability, and interpretability without the reliance on ex-

tensive human-labeled data. Empirical evaluations across diverse domains with multiple base models demonstrated that MiXSE significantly enhances base LLM performance and overcomes specialization trade-offs. We believe this work offers a step towards modular, self-improving paradigms which can address the inherent limitations of monolithic models, providing a promising direction for future LLM research.

# REFERENCES

- <span id="page-10-6"></span>AI@Meta. Llama 3 model card. 2024. URL [https://github.com/meta-llama/](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) [llama3/blob/main/MODEL\\_CARD.md](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md).
- <span id="page-10-3"></span>Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021.
- <span id="page-10-7"></span>Loubna Ben Allal, Niklas Muennighoff, Logesh Kumar Umapathi, Ben Lipkin, and Leandro von Werra. A framework for the evaluation of code generation models. [https://github.com/](https://github.com/bigcode-project/bigcode-evaluation-harness) [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness), 2022.
- <span id="page-10-4"></span>Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(05):7432–7439, April 2020. ISSN 2159-5399. doi: 10.1609/aaai.v34i05.6239. URL <http://dx.doi.org/10.1609/AAAI.V34I05.6239>.
- <span id="page-10-0"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper\\_files/paper/2020/](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) [file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf).
- <span id="page-10-8"></span>Eric L. Buehler and Markus J. Buehler. X-lora: Mixture of low-rank adapter experts, a flexible framework for large language models with applications in protein mechanics and molecular design, 2024.
- <span id="page-10-2"></span>Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob Mc-Grew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021.
- <span id="page-10-1"></span>Daixuan Cheng, Shaohan Huang, and Furu Wei. Adapting large language models via reading comprehension. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=y886UXPEZ0>.
- <span id="page-10-5"></span>Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent

- Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instructionfinetuned language models. *Journal of Machine Learning Research*, 25(70):1–53, 2024. URL <http://jmlr.org/papers/v25/23-0870.html>.
- <span id="page-11-2"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.
- <span id="page-11-3"></span>Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL <https://openreview.net/forum?id=OUIFPHEgJU>.
- <span id="page-11-6"></span>Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P Bosma, Zongwei Zhou, Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. GLaM: Efficient scaling of language models with mixture-of-experts. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 5547–5569. PMLR, 17–23 Jul 2022. URL <https://proceedings.mlr.press/v162/du22c.html>.
- <span id="page-11-5"></span>William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022. URL <http://jmlr.org/papers/v23/21-0998.html>.
- <span id="page-11-0"></span>Shangbin Feng, Weijia Shi, Yuyang Bai, Vidhisha Balachandran, Tianxing He, and Yulia Tsvetkov. Knowledge card: Filling LLMs' knowledge gaps with plug-in specialized language models. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https:](https://openreview.net/forum?id=WbWtOYIzIK) [//openreview.net/forum?id=WbWtOYIzIK](https://openreview.net/forum?id=WbWtOYIzIK).
- <span id="page-11-9"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL [https://zenodo.org/records/](https://zenodo.org/records/10256836) [10256836](https://zenodo.org/records/10256836).
- <span id="page-11-7"></span>Suchin Gururangan, Mike Lewis, Ari Holtzman, Noah A. Smith, and Luke Zettlemoyer. DEMix layers: Disentangling domains for modular language modeling. In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz (eds.), *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp. 5557–5576, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.407. URL [https://aclanthology.org/](https://aclanthology.org/2022.naacl-main.407) [2022.naacl-main.407](https://aclanthology.org/2022.naacl-main.407).
- <span id="page-11-1"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In *International Conference on Learning Representations*, 2021a. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=d7KBjmI3GmQ) [d7KBjmI3GmQ](https://openreview.net/forum?id=d7KBjmI3GmQ).
- <span id="page-11-4"></span>Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *NeurIPS*, 2021b.
- <span id="page-11-8"></span>Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. In *NIPS Deep Learning and Representation Learning Workshop*, 2015. URL [http://arxiv.](http://arxiv.org/abs/1503.02531) [org/abs/1503.02531](http://arxiv.org/abs/1503.02531).

- <span id="page-12-5"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*, 2022. URL [https://openreview.net/forum?](https://openreview.net/forum?id=nZeVKeeFYf9) [id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9).
- <span id="page-12-6"></span>Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. Lorahub: Efficient cross-task generalization via dynamic lora composition, 2023.
- <span id="page-12-2"></span>Jianheng Huang, Leyang Cui, Ante Wang, Chengyi Yang, Xinting Liao, Linfeng Song, Junfeng Yao, and Jinsong Su. Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal, 2024.
- <span id="page-12-10"></span>Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=6t0Kwf8-jrj) [6t0Kwf8-jrj](https://openreview.net/forum?id=6t0Kwf8-jrj).
- <span id="page-12-0"></span>Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas ´ Wang, Timothee Lacroix, and William El Sayed. Mistral 7b, 2023. ´
- <span id="page-12-9"></span>Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lelio Renard Lavaud, Lucile Saulnier, Marie- ´ Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Theophile Gervet, Thibaut Lavril, Thomas Wang, Timoth ´ ee Lacroix, and William El Sayed. ´ Mixtral of experts, 2024.
- <span id="page-12-11"></span>Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. Dataless knowledge fusion by merging weights of language models. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=FCnohuR6AnM>.
- <span id="page-12-7"></span>Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan (eds.), *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL [https://aclanthology.org/](https://aclanthology.org/P17-1147) [P17-1147](https://aclanthology.org/P17-1147).
- <span id="page-12-3"></span>Junmo Kang, Wei Xu, and Alan Ritter. Distill or annotate? cost-efficient fine-tuning of compact models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 11100–11119, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.622. URL [https://aclanthology.org/2023.](https://aclanthology.org/2023.acl-long.622) [acl-long.622](https://aclanthology.org/2023.acl-long.622).
- <span id="page-12-4"></span>Junmo Kang, Hongyin Luo, Yada Zhu, Jacob Hansen, James Glass, David Cox, Alan Ritter, Rogerio Feris, and Leonid Karlinsky. Self-specialization: Uncovering latent expertise within large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Findings of the Association for Computational Linguistics ACL 2024*, pp. 2681–2706, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. URL <https://aclanthology.org/2024.findings-acl.157>.
- <span id="page-12-8"></span>Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=T5nUQDrM4u>.
- <span id="page-12-1"></span>Suhas Kotha, Jacob Mitchell Springer, and Aditi Raghunathan. Understanding catastrophic forgetting in language models via implicit inference. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=VrHiF2hsrm>.

- <span id="page-13-2"></span>Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7:452–466, 2019. doi: 10.1162/tacl a 00276. URL <https://aclanthology.org/Q19-1026>.
- <span id="page-13-5"></span>Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, Yuxian Gu, Xin Cheng, Xun Wang, Si-Qing Chen, Li Dong, Wei Lu, Zhifang Sui, Benyou Wang, Wai Lam, and Furu Wei. Synthetic data (almost) from scratch: Generalized instruction tuning for language models, 2024a.
- <span id="page-13-4"></span>Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models, 2022.
- <span id="page-13-7"></span>Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason E Weston, and Mike Lewis. Self-alignment with instruction backtranslation. In *The Twelfth International Conference on Learning Representations*, 2024b. URL [https://openreview.net/forum?](https://openreview.net/forum?id=1oijHJBRsT) [id=1oijHJBRsT](https://openreview.net/forum?id=1oijHJBRsT).
- <span id="page-13-8"></span>Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Alexander Cosgrove, Christopher D Manning, Christopher Re, Diana Acosta-Navas, Drew Arad Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue WANG, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri S. Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Andrew Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. Holistic evaluation of language models. *Transactions on Machine Learning Research*, 2023. ISSN 2835-8856. URL <https://openreview.net/forum?id=iO4LZibEqW>. Featured Certification, Expert Certification.
- <span id="page-13-3"></span>Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3214–3252, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.229. URL [https://aclanthology.org/2022.](https://aclanthology.org/2022.acl-long.229) [acl-long.229](https://aclanthology.org/2022.acl-long.229).
- <span id="page-13-1"></span>Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. RA-DIT: Retrieval-augmented dual instruction tuning. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=22OTbutug9) [22OTbutug9](https://openreview.net/forum?id=22OTbutug9).
- <span id="page-13-0"></span>Chen Ling, Xujiang Zhao, Jiaying Lu, Chengyuan Deng, Can Zheng, Junxiang Wang, Tanmoy Chowdhury, Yun Li, Hejie Cui, Xuchao Zhang, Tianjiao Zhao, Amit Panalkar, Dhagash Mehta, Stefano Pasquali, Wei Cheng, Haoyu Wang, Yanchi Liu, Zhengzhang Chen, Haifeng Chen, Chris White, Quanquan Gu, Jian Pei, Carl Yang, and Liang Zhao. Domain specialization as the key to make large language models disruptive: A comprehensive survey, 2023.
- <span id="page-13-6"></span>Hongyin Luo. Self-training for natural language processing. *Ph.D. thesis, Massachusetts Institute of Technology*, 2022.
- <span id="page-13-9"></span>Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods. [https://github.](https://github.com/huggingface/peft) [com/huggingface/peft](https://github.com/huggingface/peft), 2022.

- <span id="page-14-6"></span>Michael S Matena and Colin A Raffel. Merging models with fisher-weighted averaging. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 17703–17716. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/70c26937fbf3d4600b69a129031b66ec-Paper-Conference.pdf) [file/70c26937fbf3d4600b69a129031b66ec-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/70c26937fbf3d4600b69a129031b66ec-Paper-Conference.pdf).
- <span id="page-14-12"></span>Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. MetaICL: Learning to learn in context. In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz (eds.), *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp. 2791–2809, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main. 201. URL <https://aclanthology.org/2022.naacl-main.201>.
- <span id="page-14-9"></span>Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In *ACL*, 2022.
- <span id="page-14-4"></span>Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, and Ahmed Awadallah. Orca 2: Teaching small language models how to reason, 2023.
- <span id="page-14-1"></span>Mohammed Muqeeth, Haokun Liu, Yufan Liu, and Colin Raffel. Learning to route among specialized experts for zero-shot generalization, 2024.
- <span id="page-14-7"></span>Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35: 27730–27744, 2022.
- <span id="page-14-8"></span>Hongjin Su, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A Smith, Luke Zettlemoyer, Tao Yu, et al. One embedder, any task: Instruction-finetuned text embeddings. *arXiv preprint arXiv:2212.09741*, 2022.
- <span id="page-14-5"></span>Shivchander Sudalairaj, Abhishek Bhandwaldar, Aldo Pareja, Kai Xu, David D. Cox, and Akash Srivastava. Lab: Large-scale alignment for chatbots, 2024.
- <span id="page-14-3"></span>Sainbayar Sukhbaatar, Olga Golovneva, Vasu Sharma, Hu Xu, Xi Victoria Lin, Baptiste Roziere, ` Jacob Kahn, Daniel Li, Wen tau Yih, Jason Weston, and Xian Li. Branch-train-mix: Mixing expert llms into a mixture-of-experts llm, 2024.
- <span id="page-14-10"></span>Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, and Chuang Gan. Principle-driven self-alignment of language models from scratch with minimal human supervision. In *Advances in Neural Information Processing Systems*, 2023.
- <span id="page-14-2"></span>Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, ¨ Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging bigbench tasks and whether chain-of-thought can solve them. *arXiv preprint arXiv:2210.09261*, 2022.
- <span id="page-14-11"></span>Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. [https://github.com/tatsu-lab/stanford\\_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023.
- <span id="page-14-0"></span>Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Riviere, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, L ` eonard ´ Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Amelie H ´ eliou, Andrea Tacchetti, Anna Bulanova, An- ´ tonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Clement Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric ´ Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu,

Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikuła, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vlad Feinberg, Wojciech Stokowiec, Yu hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Warkentin, Ludovic Peran, Minh Giang, Clement Farabet, Oriol Vinyals, Jeff Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin ´ Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. Gemma: Open models based on gemini research and technology, 2024.

<span id="page-15-0"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

<span id="page-15-1"></span>Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, and Mi Zhang. Efficient large language models: A survey. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL <https://openreview.net/forum?id=bsCCJHbO8A>. Survey Certification.

<span id="page-15-6"></span>Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 13484– 13508, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/ v1/2023.acl-long.754. URL <https://aclanthology.org/2023.acl-long.754>.

<span id="page-15-5"></span>Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In *International Conference on Learning Representations*, 2022. URL [https://openreview.net/](https://openreview.net/forum?id=gEZrGCozdqR) [forum?id=gEZrGCozdqR](https://openreview.net/forum?id=gEZrGCozdqR).

<span id="page-15-4"></span>Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 23965–23998. PMLR, 17–23 Jul 2022. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v162/wortsman22a.html) [v162/wortsman22a.html](https://proceedings.mlr.press/v162/wortsman22a.html).

<span id="page-15-2"></span>Hongqiu Wu, Linfeng Liu, Hai Zhao, and Min Zhang. Empower nested Boolean logic via selfsupervised curriculum learning. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 13731– 13742, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.emnlp-main.847. URL <https://aclanthology.org/2023.emnlp-main.847>.

<span id="page-15-3"></span>Xun Wu, Shaohan Huang, and Furu Wei. Mixture of loRA experts. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=uWvKBCYh4S) [id=uWvKBCYh4S](https://openreview.net/forum?id=uWvKBCYh4S).

- <span id="page-16-4"></span>Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. TIES-merging: Resolving interference when merging models. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL <https://openreview.net/forum?id=xtaX3WyCj1>.
- <span id="page-16-5"></span>Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In *International Conference on Machine Learning*. PMLR, 2024.
- <span id="page-16-2"></span>Matei Zaharia, Omar Khattab, Lingjiao Chen, Jared Quincy Davis, Heather Miller, Chris Potts, James Zou, Michael Carbin, Jonathan Frankle, Naveen Rao, and Ali Ghodsi. The shift from models to compound ai systems. [https://bair.berkeley.edu/blog/2024/02/18/](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) [compound-ai-systems/](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/), 2024.
- <span id="page-16-3"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Llu´ıs Marquez ` (eds.), *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10. 18653/v1/P19-1472. URL <https://aclanthology.org/P19-1472>.
- <span id="page-16-0"></span>Biao Zhang, Zhongtao Liu, Colin Cherry, and Orhan Firat. When scaling meets LLM finetuning: The effect of data, model and finetuning method. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=5HCnKDeTws) [5HCnKDeTws](https://openreview.net/forum?id=5HCnKDeTws).
- <span id="page-16-1"></span>Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu, Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, and Mengnan Du. Explainability for large language models: A survey. *arXiv preprint arXiv:2309.01029*, 2023.

# A EXPERIMENT DETAILS

We provide each of our self-specialization prompts for knowledge, reasoning, math, and coding experts in Tables 9, 10, 11, and 12. We largely follow Kang et al. (2024)'s prompt structure to ensure quality, with additional domain-specific instructions that inform task-related information.

For our evaluation, we employ popular and widely accepted evaluation frameworks to pursue standard evaluation setups and protocols: HELM (Liang et al., 2023), LM Evaluation Harness (Gao et al., 2023), and BigCode Evaluation Harness (Ben Allal et al., 2022). We use Huggingface PEFT (Mangrulkar et al., 2022) and XLoRA (Buehler & Buehler, 2024) for the implementation of MoE compatible with LoRA.

Regarding seed instructions, we sampled 100 training instances from each of the MMLU, BBH, and GSM8K datasets, for knowledge, reasoning, and math domains, respectively. For coding, since the size of the HumanEval dataset is very small and thus the training set is not available, we took 100 samples from the MBPP training set and converted the task format to make them suit the HumanEval.

During instruction generation, we use three seed data, which are randomly sampled, as in-context examples, using a temperature of 1 and top-p of 0.98, whereas we use five seed data in-context for response generation with greedy decoding. For specialization, we use LoRA applied to all modules with a rank of 8 and alpha of 16, and train it using a learning rate of 3e-4, epochs of 3, and batch size of 32. We train each module and MiXSE using a standard Alpaca (Taori et al., 2023) prompt template on a single A100-80GB, which takes only a few hours.

## B LIMITATIONS

While our study demonstrates promising results for the Self-MoE, we recognize areas requiring further investigation in future work. Employing self-specialization Kang et al. (2024) to generate synthetic data within our framework may raise concerns about potential data contamination and noise. Nonetheless, findings from Kang et al. (2024), which conducted an n-gram overlap analysis between the self-specialization data and test data, confirmed no significant overlap, thus alleviating the concerns about contamination. Despite this, the need for continuous monitoring of potential biases from pre-training and the development of enhanced data validation and noise filtering strategies remain important, and may present interesting direction for future work. Moreover, due to computational constraints, we did not scale our model and data to their full potential. We also did not work on the optimization of the XLoRA, the MoE module we used, to focus purely on the research problem defined in this study. Future work should therefore concentrate on overcoming these limitations, which will enable better data quality and more extensive training to unveil the full potential of the Self-MoE framework.

<span id="page-17-0"></span>Table 6: Dataset statistics. Non-Target (In-Expertise) indicates where MiXSE does not directly specialize using seed data directly while relevant to targets. Non-Target (Out-of-Expertise) refers to irrelevant cases.

| Category           | Benchmark                     | # Examples |  |  |  |  |  |  |  |
|--------------------|-------------------------------|------------|--|--|--|--|--|--|--|
|                    | Target                        |            |  |  |  |  |  |  |  |
| Academic Knowledge | MMLU (57 Tasks)               | 14,079     |  |  |  |  |  |  |  |
| Reasoning          | BBH (27 Tasks)                | 6,511      |  |  |  |  |  |  |  |
| Math               | GSM8K                         | 8,790      |  |  |  |  |  |  |  |
| Coding             | HumanEval                     | 164        |  |  |  |  |  |  |  |
| Non-To             | Non-Target (In-Expertise)     |            |  |  |  |  |  |  |  |
| Math               | MATH                          | 12,500     |  |  |  |  |  |  |  |
| Coding             | MBPP                          | 257        |  |  |  |  |  |  |  |
| Non-Targ           | Non-Target (Out-of-Expertise) |            |  |  |  |  |  |  |  |
| W11171-1           | Natural Questions             | 3,610      |  |  |  |  |  |  |  |
| World Knowledge    | TriviaQA                      | 17,200     |  |  |  |  |  |  |  |
| Commonsense        | Hellaswag                     | 10,000     |  |  |  |  |  |  |  |
| Commonsense        | PIQA                          | 3,000      |  |  |  |  |  |  |  |
| Safety             | TruthfulQA                    | 817        |  |  |  |  |  |  |  |

<span id="page-18-1"></span>Table 7: Additional comparisons with other models for references. Results are extracted from each corresponding paper, except for pre-training methods where the numbers are all from BTX (Sukhbaatar et al., 2024).

| Method                                                                                                                                                 | Total<br>Params      | Active<br>Params        | Compos-<br>itional | Semantic<br>Experts | Light-<br>weight | Data & Resrc<br>-Efficient | w/o Teacher<br>& Labels | Knowledge<br>(MMLU 5-shot) | Reasoning<br>(BBH)   | Math<br>(GSM8K)      | Coding<br>(HumanEval) |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-------------------------|--------------------|---------------------|------------------|----------------------------|-------------------------|----------------------------|----------------------|----------------------|-----------------------|
| Base LLM                                                                                                                                               |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| Gemma 7B (Team et al., 2024)<br>LLaMA-2 70B (Touvron et al., 2023)<br>Mixtral 8x7B (Jiang et al., 2024)                                                | 7B<br>70B<br>47B     | 7B<br>70B<br>13B        | ×                  | ×                   | ×                | -<br>-<br>-                | -<br>-                  | 65.7<br>68.9<br>70.6       | 56.1<br>51.2<br>67.1 | 42.5<br>35.2<br>65.7 | 34.1<br>29.9<br>32.3  |
| Pre-training Methods                                                                                                                                   |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| Branch-Train-Merge (4x7B) (Li et al., 2022)<br>Sparse Upcycling (4x7B) (Komatsuzaki et al., 2023)<br>Branch-Train-Mix (4x7B) (Sukhbaatar et al., 2024) | <24B<br><24B<br><24B | 11.1B<br>11.1B<br>11.1B | * * *              | **                  | ×××              | ×<br>×                     | ž                       | 44.3<br>52.1<br>52.5       | -<br>-               | 27.7<br>40.1<br>37.1 | 30.6<br>26.2<br>28.7  |
| MoE w/ LoRA                                                                                                                                            |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| PHATGOOSE (Muqeeth et al., 2024)<br>MOLE (Wu et al., 2024)                                                                                             | <4B                  | >3B                     | ×                  | × ×                 | ž                | ×                          | ×                       | -                          | 35.6<br>42.2         | -                    | -                     |
| Distillation/Synthetic Data from Larger Models                                                                                                         |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| GLAN 7B (w/ GPT-4) (Li et al., 2024a)<br>Orca-2 7B (w/ GPT-4) (Mitra et al., 2023)<br>Merlinite 7B (w/ Mixtral 8x7B) (Sudalairaj et al., 2024)         | 7B<br>7B<br>7B       | 7B<br>7B<br>7B          | ×<br>×             | -                   | -                | ×<br>×                     | ×<br>×                  | 62.9<br>53.9<br>64.9       | 60.7<br>42.8         | 80.8<br>55.7<br>44.6 | 48.8<br>17.1          |
| Self-Improving                                                                                                                                         |                      |                         |                    |                     |                  |                            |                         |                            |                      |                      |                       |
| Ours                                                                                                                                                   | 7B + 1%              | 7B + 0.3% ✓             | ~                  | ~                   | ~                | ~                          | 66.2                    | 61.1                       | 52.5                 | 37.8                 |                       |

## C DATASET DESCRIPTIONS

The statistics for each dataset are provided in Table 6. The target datasets used are as follows:

- MMLU (Massive Multitask Language Understanding) (Hendrycks et al., 2021a): A collection of 57 academic knowledge tasks.
- BBH (BIG-Bench Hard (Suzgun et al., 2022): A set of 27 challenging reasoning tasks.
- **GSM8K** (Grade School Math 8K) (Cobbe et al., 2021): A diverse set of grade school math word problems.
- **HumanEval** (Chen et al., 2021): A hand-written evaluation set for python programming problems.

#### D ADDITIONAL RESULTS

#### <span id="page-18-0"></span>D.1 ADDITIONAL COMPARISON AND DISCUSSION

In Table 7, we present additional comparisons with various other models and methods to provide a broader perspective, though comparisons may not appear to be direct, due to factors involved such as parameters, resources, etc. We discuss some noteworthy points.

Notably, although MiXSE significantly improves upon its base model, Gemma 7B, it does not yet reach the performance levels of the more powerful Mixtral 8x7B. It is important to understand that Mixtral also utilizes an MoE (Mixture of Experts) architecture, but unlike MiXSE, it does not prioritize lightweight experts, leading to a much larger model with significantly more parameters. Moreover, while Mixtral's experts are implicitly built during pre-training, MiXSE explicitly creates semantic experts, allowing for targeted improvements and clearer interpretability. Importantly, our self-improving method can be potentially applied on top of any pre-trained model including Mixtral in principle.

Similarly, BTX (Branch-Train-MiX) uses a pre-training MoE strategy where parameter-heavy semantic experts are employed, yielding substantial enhancements over the base LLM. This approach highlights the effectiveness of using semantically rich experts to refine the model's capabilities. To make comparisons in terms of efficiency, our model uses fewer parameters (7B), compared to BTX (12B active with much more whole parameters) and requires only about 1 GPU day for training, compared to 900 GPU days for BTX. In essence, since BTX is also a pre-training method while specialized, we expect it to be complementary to our Self-MoE, as evidenced in previous work (Kang et al., 2024).

With a shared spirit, MOLE and PHATGOOSE build a MoE (Mixture of Experts) using LoRA, which is semantic and lightweight. However, there are significant differences in foundational assumptions: MOLE depends on human-labeled data, while PHATGOOSE requires access to pre-

<span id="page-19-0"></span>Table 8: Results of MiXSE using only seed data. Seed Only training shows only marginal improvements over the Base LLM in some benchmarks, validating that the effect of Self-MoE is not merely due to the use of seed data.

| Benchmark             | Base LLM | Seed Only | MiXSE |
|-----------------------|----------|-----------|-------|
| Knowledge<br>(MMLU)   | 58.3     | 57.4      | 65.6  |
| Reasoning<br>(BBH)    | 56.1     | 57.0      | 61.1  |
| Math<br>(GSM8K)       | 42.5     | 45.0      | 52.5  |
| Coding<br>(HumanEval) | 34.1     | 34.1      | 37.8  |
| Avg.                  | 47.8     | 48.4      | 54.3  |

trained expert models developed externally. In contrast, our Self-MoE framework independently constructs both experts and a router entirely from scratch, focusing on self-improvement without such dependencies. While their scenarios are considered reasonable in a certain context, we aim for broader applicability by minimizing assumptions on conditions.

Lastly, GLAN demonstrates outstanding performance across various domains. This is attributed to their reliance on distilling from the larger and stronger model, GPT-4, using a huge amount of data (e.g., 10 million). As outlined in our problem statement (Section [2\)](#page-2-1), we deliberately avoid assuming the availability of such advanced models to ensure the broader applicability of our method which self-improves from scratch. Consequently, while acknowledging each of their own value, it is crucial to recognize that direct comparisons may not be entirely appropriate, given the fundamental differences in resource assumptions and initial conditions.

## D.2 MIXSE USING ONLY SEED DATA

Table [8](#page-19-0) shows the results of the MiXSE when exploiting only seed data for training, clarifying the benefits derived from our methodological enhancements beyond the mere inclusion of seed data in training. While the Seed Only shows slight improvements over the Base LLM in some benchmarks, the significant enhancements of our MiXSE across all benchmarks confirm that the enhanced capabilities of Self-MoE are not merely due to the use of seed data. This further highlights the achievement of self-improvement with our method.

# D.3 VAILDITY OF COMPARATIVE RESULTS

In an effort to address the concern related to the sensitivity of in-context learning [\(Min et al., 2022\)](#page-14-12), we perform three runs with the different lists of few-shot samples where applicable. As a result, we see that the mean of the base LLM (Gemma-7B)'s average performance across domains is 47.9 with a standard deviation (SD) of 0.56, that of our MiXSE is 53.6 with an SD of 0.60, and that of instance merging is 51.6 with an SD of 0.87. A statistical analysis between MiXSE and instance merging yields a p-value of 0.03, confirming the significant difference.

Table 9: Prompts for knowledge-related instruction and response generation.

#### <span id="page-20-0"></span>**Instruction Brainstorming Prompt**

You are asked to come up with a set of task instructions about diverse domains across STEM, humanities, social sciences, and others. These task instructions will be given to a language model and we will evaluate the model for completing the instructions.

Here are the requirements:

- 1. The type of task should be multiple-choice question answering. That is, a question along with multiple options (A, B, C, D) should be provided.
- 2. The language used for the instruction/question also should be diverse.
- 3. A language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
- 4. The instructions should be in English.
- 5. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
- 6. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging.
- 7. Ensure diverse domains are covered for extensive expert-level knowledge. The subjects may include Abstract Algebra, Anatomy, Astronomy, Business Ethics, Clinical Knowledge, College-level Biology, Chemistry, Computer Science, Mathematics, Medicine, Physics, Computer Security, Conceptual Physics, Econometrics, Electrical Engineering, Elementary Mathematics, Formal Logic, Global Facts, High School-level Biology, Chemistry, Computer Science, European History, Geography, Gov't and Politics, Macroeconomics, Mathematics, Microeconomics, Physics, Psychology, Statistics, US History, World History, Human Aging, Human Sexuality, International Law, Jurisprudence, Logical Fallacies, Machine Learning, Management, Marketing, Medical Genetics, Miscellaneous, Moral Disputes, Moral Scenarios, Nutrition, Philosophy, Prehistory, Professional-level (Accounting, Law, Medicine, Psychology), Public Relations, Security Studies, Sociology, US Foreign Policy, Virology, World Religions, etc.

List of tasks:

#### **Response Generation**

You are a knowledgeable domain expert. Given an instruction and a question, generate the best answer to solve the given task about STEM, humanities, social sciences, and others.

Table 10: Prompts for reasoning-related instruction and response generation.

#### <span id="page-20-1"></span>**Instruction Brainstorming Prompt**

You are asked to come up with a set of task instructions focusing on challenging tasks that require multi-step reasoning. These task instructions will be given to a language model and we will evaluate the model for completing the instructions.

Here are the requirements:

- 1. The type of task should be question answering, requiring multi-step reasoning.
- 2. The language used for the instruction/question also should be diverse.
- 3. The generated problem should have a single correct answer.
- 4. The instructions should be in English.
- 5. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
- 6. You should generate an appropriate input question to the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging.
- 7. Ensure diverse topics and levels are covered for extensive expert-level reasoning. The tasks may be about boolean expression, causal judgement, date understanding, disambiguation of question, closing Dyck-n words, formal fallacies, geometric shapes, hyperbaton, logical deduction of objects, movie recommendation, multi-step arithmetic problem, navigation, object counting, table reasoning, reasoning about colored objects, selecting one that ruins the name in an input, salient translation error detection, sarcastic sentence classification, sports understanding, temporal sequences, tracking shuffled objects, web of lies, word sorting, etc.

List of tasks:

## **Response Generation**

You are a multi-step reasoning expert. Given an instruction and a challenging question, generate step-by-step reasoning and the answer.

Table 11: Prompts for math-related instruction and response generation.

#### <span id="page-21-1"></span>**Instruction Brainstorming Prompt**

You are asked to come up with a set of task instructions focusing on mathematical problems. These task instructions will be given to a language model and we will evaluate the model for completing the instructions.

Here are the requirements:

- 1. The type of task should be question answering, requiring multi-step reasoning.
- 2. The language used for the instruction/question also should be diverse.
- 3. The generated mathematical problem should have a solution.
- 4. The instructions should be in English.
- 5. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
- 6. You should generate an appropriate input question to the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging.
- 7. Ensure diverse topics and levels are covered for extensive expert-level reasoning. The subjects may include Algebra, Counting, Probability, Calculus, Statistics, Geometry, Linear Algebra, Number Theory and grade school math, etc.

List of tasks:

#### **Response Generation**

You are a math expert. Given an instruction and a mathematical question, generate step-by-step reasoning and the answer.

Table 12: Prompts for coding-related instruction and response generation.

#### <span id="page-21-0"></span>**Instruction Brainstorming Prompt**

You are asked to come up with a set of task instructions focusing on coding problems. These task instructions will be given to a language model and we will evaluate the model for completing the instructions.

Here are the requirements:

- 1. The type of task should be about coding problems, such as writing a python function given a specific instruction and test examples.
- 2. The language used for the instruction should be diverse, but the programming language should be python.
- 3. The generated problem should have a solution.
- 4. The instructions should be in English.
- 5. You should generate appropriate and correct test examples for the given problem.
- 6. Ensure diverse functions and levels are covered for extensive expert-level coding.

List of tasks:

#### **Response Generation**

You are a coding expert. Given an instruction and test cases, write a python function that passes the test cases.