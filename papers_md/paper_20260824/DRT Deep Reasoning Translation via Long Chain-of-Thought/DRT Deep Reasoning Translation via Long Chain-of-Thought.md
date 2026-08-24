# DRT: Deep Reasoning Translation via Long Chain-of-Thought

# Jiaan Wang, Fandong Meng\*, Yunlong Liang, Jie Zhou

Pattern Recognition Center, WeChat AI, Tencent Inc

{torchwang,fandongmeng,yunlonliang,withtomzhou}@tencent.com

# Abstract

Recently, O1-like models have emerged as representative examples, illustrating the effectiveness of long chain-of-thought (CoT) in reasoning tasks such as math and coding tasks. In this paper, we introduce DRT, an attempt to bring the success of long CoT to neural machine translation (MT). Specifically, in view of the literature books that might involve similes and metaphors, translating these texts to a target language is very difficult in practice due to cultural differences. In such cases, literal translation often fails to convey the intended meaning effectively. Even for professional human translators, considerable thought must be given to preserving semantics throughout the translation process. To simulate LLMs' long thought ability in MT, we first mine sentences containing similes or metaphors from existing literature books, and then develop a multi-agent framework to translate these sentences via long thought. In the multi-agent framework, a translator is used to iteratively translate the source sentence under the suggestions provided by an advisor. To ensure the effectiveness of the long thoughts, an evaluator is also employed to quantify the translation quality in each round. In this way, we collect tens of thousands of longthought MT data, which is used to train our DRT. Using Qwen2.5 and LLama-3.1 as the backbones, DRT models can learn the thought process during machine translation, and outperform vanilla LLMs as well as LLMs which are simply fine-tuning on the paired sentences without long thought, showing its effectiveness.[1](#page-0-0)

# 1 Introduction

Recently, the emergence of the O1-like LLMs shows great performance in reasoning tasks, *e.g.*, math and coding tasks [\(OpenAI,](#page-9-0) [2024b;](#page-9-0) [Qin et al.,](#page-9-1) [2024;](#page-9-1) [Huang et al.,](#page-8-0) [2024;](#page-8-0) [Zhang et al.,](#page-9-2) [2024;](#page-9-2) [Zhao](#page-9-3)

[et al.,](#page-9-3) [2024\)](#page-9-3). With the help of long thought, LLMs tend to explore, reflect and self-improve the reasoning processes to achieve more accurate answers.

In this paper, we explore technical routes to bring the success of long thought to MT. To this end, we introduce DRT, a product of our exploration, and we hope it could facilitate the research community. There are two key points in achieving this goal:

- i) A suitable translation scenario to employ long thought in MT: Not all scenarios require long chain-of-thought (CoT)[2](#page-0-1) during translation. For example, in simple expressions, literal translation can meet most needs, and translation via long CoT may be unnecessary. Inappropriate scenarios might cause the overthinking issue [\(Chen et al.,](#page-8-1) [2024\)](#page-8-1).
- ii) A method to synthesize MT data with long thought: Long thought SFT (supervised finetuning) data plays a vital role in simulating LLMs' long thought ability [\(Huang et al.,](#page-8-0) [2024\)](#page-8-0). Previous work pays much attention to how to synthesize long-thought data in math and coding tasks [\(Qin](#page-9-1) [et al.,](#page-9-1) [2024;](#page-9-1) [Huang et al.,](#page-8-0) [2024;](#page-8-0) [Zhao et al.,](#page-9-3) [2024\)](#page-9-3).

For i), inspired by [Van den Broeck](#page-9-4) [\(1981\)](#page-9-4), a possible scenario is translating sentences with similes or metaphors, where literal translation often fails to convey the intended semantics. Given that, we decide to mine such sentences from literature books. The mining process uses an advanced large language model (LLM) to first judge Q1: *whether each literature sentence has any similes or metaphors*. If has, the LLM will be asked to literally translate the sentence to a target language, and give a final judgment on Q2: *whether literal translation is effective for native speakers of the target language to comprehend.* If the answers of Q1 and Q2 are "yes" and "no", respectively, the corresponding literature sentences will be reserved, and regarded as "suitable to translate via long thought".

<span id="page-0-0"></span><sup>\*</sup> [Corresponding author.](#page-9-3)

<sup>1</sup>[The synthesized data and model checkpoints are released](#page-9-3) at [https://github.com/krystalan/DRT](#page-9-3).

<span id="page-0-1"></span><sup>&</sup>quot;long CoT" is equal to "long thought", and we alternatively use these two terms in this paper.

For ii), after collecting the literal sentences with similes or metaphors, the next question is how to synthesize long thought MT samples. Previous work typically utilizes Monte Carlo Tree Search (MCTS) [\(Qin et al.,](#page-9-1) [2024;](#page-9-1) [Zhao et al.,](#page-9-3) [2024;](#page-9-3) [Zhang](#page-9-2) [et al.,](#page-9-2) [2024\)](#page-9-2) or data distillation [\(Huang et al.,](#page-8-0) [2024\)](#page-8-0) (from existing O1-like models) to collect long thought SFT samples. Nevertheless, MCTS is typically used in math and coding tasks where multiple reasoning behaviors should be considered, and the method emphasizes complex reasoning that might not be efficient for machine translation. Besides, utilizing existing O1-like models for data distillation might (1) constrain the potential quality of the long-thought data; and (2) have a data gap in MT since current O1-like models are typically optimized toward math and coding tasks.

Therefore, we propose a multi-agent framework to synthesize MT data with long thought. In detail, there are three agents in the framework, *i.e.*, a translator, an advisor and an evaluator. The synthesis process is iterative, consisting of the following three steps during each iteration: (1) the translator generates a new translation conditioned on the previous step's translation and the corresponding refinement suggestions from the advisor; (2) the advisor evaluates the current translation and offers detailed feedback; (3) the evaluator assesses the current translation and gives an evaluation score using predefined scoring criteria. Once the translation score provided by the evaluator reaches a pre-defined threshold or the number of iterations reaches a maximum value, the iteration will stop. After that, the translation and suggestions in every step could form the long-thought MT samples. To improve the readability and fluency of the longthought data, we employ GPT-4o [\(OpenAI,](#page-9-5) [2024a\)](#page-9-5) to reformulate the long-thought content.

Based on the collected long-thought MT samples, we train our DRT-7B, DRT-8B and DRT-14B using the backbones of Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct [\(Dubey et al.,](#page-8-2) [2024\)](#page-8-2) and Qwen2.5-14B-Instruct [\(Yang et al.,](#page-9-6) [2024a\)](#page-9-6), respectively. Experimental results on literature translation verify their effectiveness. In particular, DRT-14B outperforms QwQ-32B-preview and DeepSeek-R1- Distill-Qwen-32B in terms of BLEU, CometKiwi, CometScore and GPT-4 evaluations. Moreover, human evaluation and case study show the strong translation performance of DRT models.

Our main contributions are concluded as follows:

• We propose DRT aiming at building LLMs with

- long-thought machine translation ability. To achieve this, we mine literature sentences with similes or metaphors, and collect MT samples with long-thought processes.
- To synthesize the long-thought MT samples, we propose a multi-agent framework that involves a translator, an advisor and an evaluator. These three agents collaborate in an iterative manner to produce long thoughts during MT. Lastly, GPT-4o is used to further improve the quality of the synthesized long-thought MT samples.
- Experimental results on literature translation verify the effectiveness of our DRT. With the help of long thought, LLMs can learn to think during the machine translation.

# 2 DRT Data

We focus on English-to-Chinese translation[3](#page-1-0) , and we introduce how to collect the long-thought MT samples via three steps in this section: (1) collecting English sentences that tend to require long thoughts during translation (§ [2.1\)](#page-1-1); (2) synthesizing the long-thought translation process for the collected sentences by a designed multi-agent framework (§ [2.2\)](#page-2-0); (3) improving the readability and fluency of the long-thought content to form the final long-thought MT samples (§ [2.3\)](#page-2-1). Next, we provide data statistics and data analyses of the collected data to give a deeper understanding (§ [2.4\)](#page-3-0). Finally, we discuss the data quality (§ [2.5\)](#page-4-0).

# <span id="page-1-1"></span>2.1 Literature Book Mining

Following [Kryscinski et al.](#page-8-3) [\(2022\)](#page-8-3), we leverage the literature books from the Project Gutenberg publicdomain book repository[4](#page-1-2) , where the books are typically more than fifty years old and their copyrights have expired. About 400 English books are used to mine sentences with similes or metaphors.

First, we extract all sentences from these books, and filter out too short or too long sentences, *i.e.*, less than 10 words or more than 100 words, resulting in 577.6K literature sentences. Second, for each sentence, we use Qwen2.5-72B-Instruct [\(Yang](#page-9-6) [et al.,](#page-9-6) [2024a\)](#page-9-6) to judge whether the sentence involves similes or metaphors, and discard the sentences that do not contain any ones. Third, for the remaining sentences, we let Qwen2.5-72B-Instruct literally translate them to Chinese, and then judge

<span id="page-1-0"></span><sup>3</sup>Although we focus on English-to-Chinese translation in this work, the methods we introduced can be trivially applied to other languages or translation directions.

<span id="page-1-2"></span><sup>4</sup><https://www.gutenberg.org/>

<span id="page-2-2"></span>> **[图片提取文字 (无描述)]:**
> Source Sentence Translation Suggestions The Last Conflict In the second week of September, Maggie was again sitting in her lonely room, battling with the old The translation is very good. It accurately shadowy enemies that were forever slain and rising again. conveys the meaning of the original text and maintains a natural flow in Chinese. The Word-Level Translation phrase 'old shadowy enemies' is translated as '老敌人', which might lose some of the eerie Keyword Translation and mysterious connotation of the original, Conflict-冲突; enemies-敌人; slain-被杀死的; rising-复活 but it is still understandable. Overall, the Translator translation is clear and effective Preliminary Translation Translation\ Advisor Refine Loon Sentence Translation Evaluation Score 在九月的第二周, 玛吉再次坐在她孤独的房间里, 与那 些永远被杀死又再次复活的老敌人斗争。 Evaluator Translator
![](_page_2_Figure_0.jpeg)

Figure 1: The illustration of the multi-agent framework to synthesize long-thought MT samples. (a) A translator iteratively produces translations under the suggestions provided by an advisor; (b) An advisor reviews the translation results and gives suggestions; (c) An evaluator assesses the translation results and gives an overall score to indicate the translation quality.

whether the translation satisfies native Chinese people. If the answer is negative, the corresponding sentence will be reserved, and regarded as "suitable to translate via long thought". For prompt details, please refer to Appendix A.1. Consequently, we collect 63K (out of 577.6K) literature sentences involving similes or metaphors whose literal translations have flaws, called *pre-collected sentences*.

### <span id="page-2-0"></span>2.2 Multi-Agent Framework

For each pre-collected sentence (denoted as s), we design a multi-agent framework to translate it via long thought. As shown in Figure 1, our multi-agent framework includes three agents: a translator, an advisor, and an evaluator, each of which use Qwen2.5-72B-Instruct as the backbone. The synthetic process is illustrated as follows:

- (1) Word-level Translation. The translator first identifies the keywords that lie in the sentence, and then provides their translations under the consideration of the context. The keywords are denoted as  $\mathcal{W}^{\text{src}} = \{w_1^{\text{src}}, w_2^{\text{src}}, ..., w_k^{\text{src}}\}, \text{ where } w_i^{\text{src}} \text{ indicates the } i\text{-th keyword in } s, \text{ and } k \text{ is the number of keywords.}$  The translation of keywords is denoted as  $\mathcal{W}^{\text{tgt}} = \{w_1^{\text{tgt}}, w_2^{\text{tgt}}, ..., w_k^{\text{tgt}}\}. \text{ This step enables the model to identify potential challenges in translating the entire sentence by breaking it down into sub-problems (i.e., word-level translation).}$
- (2) Preliminary Translation. The translator then provides a preliminary sentence translation  $(t^0)$  conditioned on both the source sentence (s) and its keyword bilingual pairs  $(\langle \mathcal{W}^{\rm src}, \mathcal{W}^{\rm tgt} \rangle)$ .
- (3) Translation Refine Loop. In the refine loop, three agents work together to refine the translation iteratively. In each iteration step k (start from k=1), the advisor first evaluates the translation in

the previous step, *i.e.*,  $t^{k-1}$ , and provides detailed feedback  $f^{k-1}$  for polishing it. Then, the evaluator gives an overall score of  $t^{k-1}$  conditioned on both pre-defined scoring criteria and  $f^{k-1}$ , and the score is denoted as  $s^{k-1}$ . In the last of the iteration step, the translator takes its previous translation  $t^{k-1}$ , the corresponding feedback  $f^{k-1}$  and overall score  $s^{k-1}$  into account to provide a new translation  $t^k$ . The translation refine loop will stop when the overall score reaches a pre-defined threshold or the number of iteration steps meets the maximum. For prompt details of the translator, advisor and evaluator, please refer to Appendix A.2.

## <span id="page-2-1"></span>2.3 Long Thought Reformulation

After the multi-agent collaboration, we obtain a long thought process:

$$\mathcal{P}(s): s \Rightarrow \langle \mathcal{W}^{\text{src}}, \mathcal{W}^{\text{tgt}} \rangle \Rightarrow \langle t^0, f^0, s^0 \rangle$$
$$\Rightarrow \langle t^1, f^1, s^1 \rangle \Rightarrow \dots \Rightarrow \langle t^m, f^m, s^m \rangle$$
(1)

where  $\mathcal{P}(s)$  denotes the multi-agent thought process for s, and m is the number of iteration steps. To emphasize the valid thought process, translations without score change will be removed. That is, if  $s^i$  is equal to  $s^{i-1}$  (i=1,2,...,m), we will discard  $\langle t^i, f^i, s^i \rangle$  in  $\mathcal{P}(s)$ , resulting in:

$$\mathcal{P}'(s): s \Rightarrow \langle \mathcal{W}^{\text{src}}, \mathcal{W}^{\text{tgt}} \rangle \Rightarrow \langle t^0, f^0, s^0 \rangle$$
  
 
$$\Rightarrow \langle t^{r_1}, f^{r_1}, s^{r_1} \rangle \Rightarrow \dots \Rightarrow \langle t^{r_n}, f^{r_n}, s^{r_n} \rangle$$
 (2)

where  $1 \le r_1 < r_2 < ... < r_n \le m$ , and n is the number of remaining steps. If n < 3, we will discard the whole sample, *i.e.*,  $\mathcal{P}(s)$ .

For the remaining samples, we follow Qin et al. (2024), and leverage GPT-40 (OpenAI, 2024a) to modify and polish  $\mathcal{P}'(s)$  into a self-reflection

<span id="page-3-1"></span>[Source Sentence] Passing by the tower with her husband on the path to the gate she could feel the vibrant air humming round them from the louvred belfry in the circle of sound, and it matched the highly-charged mental atmosphere in which she was living.

[Thought During Translation] I start by identifying and translating the key terms from the English sentence: 'tower' as '塔', 'husband' as '丈夫', 'gate' as '大门', 'vibrant' as '充满活力的', 'humming' as '嗡嗡作响', 'louvred' as '有百叶窗的', 'belfry' as '钟楼', 'circle' as '圆圈', 'sound' as '声音', 'highly-charged' as '高度紧张的', 'mental' as '精神的', and 'atmosphere' as '氛围'.

With these translations in mind, I attempt my first translation: 经过塔时,她和丈夫走在通往大门的路上,能感受到有百叶窗的钟楼在声音的圆圈中发出的充满活力的空气在他们周围嗡嗡作响,这与她所处的高度紧张的精神氛围相匹配。

I reflect on this attempt and realize that while it is generally accurate, the phrasing could be improved for better fluency and clarity. The segment 有百叶窗的钟楼在声音的圆圈中发出的充满活力的空气' feels cumbersome and could be simplified. Additionally, '高度紧张的精神氛围' might not fully capture the nuance of 'highly-charged mental atmosphere.'

I revise my translation: 经过塔时,她和丈夫走在通往大门的路上,感受到从有百叶窗的钟楼传来的充满活力的空气在他们周围轻轻回响,这与她所处的高度活跃的精神氛围相契合。

This revision feels more fluent and natural, effectively conveying the original meaning. The phrase '轻轻回啊' improves the flow and captures the sense of the air humming around them. The term '高度活跃的精神氛围' seems a better fit for 'highly-charged mental atmosphere,' making the translation more accurate and easier to understand for a Chinese reader.

I make another revision: 经过塔时,她和丈夫走在通往大门的路上,感受到从有百叶窗的钟楼传来的充满活力的气息在他们周围轻轻回荡,这与她内心的高度活跃氛围相契合。

This version is very smooth and natural, effectively conveying the original meaning. The use of '轻轻回荡' is particularly apt, capturing the sense of the air humming around them. The phrase '高度活跃的精神氛围' accurately reflects the 'highly-charged mental atmosphere,' making the translation both precise and fluent. There are no significant issues that would hinder understanding for a Chinese reader.

Finally, I decide on the final translation: 经过塔时,她和丈夫走在通往大门的路上,感受到从有百叶窗的钟楼传来的充满活力的气息在他们周围轻轻回荡,这与她内心的高度活跃氛围相映成趣。

Figure 2: An example of long thought synthesized by the designed multi-agent framework and GPT-40 reformulation.

<span id="page-3-2"></span>

|                                                              | # Sample | Query | Thought | Output |
|--------------------------------------------------------------|----------|-------|---------|--------|
| o1-journey                                                   | 327      | 41.53 | 486.05  | 3.41   |
| Marco-O1 CoT data                                            | 10,000   | 52.73 | 673.98  | 52.73  |
| DRT data (training) DRT data (validation) DRT data (testing) | 19,264   | 37.25 | 527.64  | 44.67  |
|                                                              | 1,000    | 37.43 | 531.36  | 44.98  |
|                                                              | 2,000    | 37.19 | 525.44  | 44.70  |

Table 1: The number of samples and average tokenlevel length of query, thought and output. "Query" and "Output" in DRT data mean the source sentences and the translated outputs, respectively.

description (the used prompt is provided in Appendix A.3). Finally, we obtain 22,264 MT samples with long thought. Figure 2 gives an example sample to illustrate the synthetic results.

It is also worth noting that during the GPT-40 reformulation, we specify the translation with the highest score  $s^{r_j}$  as the final translation. Thus, the final translation is not necessarily the last one during refinement, *i.e.*,  $t^{r_n}$ .

### <span id="page-3-0"></span>2.4 Data Statistics and Data Analyses

We split the collected 22,264 samples into training, validation and testing sets with 19,264, 1,000 and 2,000 samples, respectively. Table 1 shows the data statistics of DRT data and previous O1-like data. For Marco-O1 CoT data (Zhao et al., 2024), since it is not fully released, we use its demo data to calculate the data statistics.<sup>5</sup> As we can see, the average number of tokens in our synthesized

<span id="page-3-4"></span>> **[图片提取文字 (无描述)]:**
> 16301(73.22%) 16000 14000 12000 the number of samples 10000 8000 6000 4430(19.9%) 4000 2000 1174(5.27%) 350(1.57%) 8(0.04%) 1(0.0%) 0 3 8 the number of refinement steps
![](_page_3_Figure_17.jpeg)

Figure 3: The distribution of the number of refinement steps in DRT data.

thought reaches 500+ tokens, showing the long thought process in our data.

Refine Loop Analyses. Figure 3 shows the number of refinement steps in the DRT data, which ranges from 3 to 8 steps. We can find that most samples (73.22%) involve 3 refinement steps, while only one sample involves 8 steps. Furthermore, to provide a deeper understanding of the refinement process, we calculate the average edit distance before and after each refinement step. Specifically, the first three refinement steps cause 21.44, 13.16 and 10.90 character-level edit distance. This observation is consistent with intuition. As the refinement progresses, the magnitude of the modification gradually decreases. To further understand the improvement brought by the translation refine loop, we calculate the average overall scores (provided

<span id="page-3-3"></span><sup>5</sup>https://github.com/AIDC-AI/Marco-o1

<span id="page-4-1"></span>> **[图片提取文字 (无描述)]:**
> 92.37 91.94 89.18 89.52 89.52 88.56 88.46 84.72 85.03 84,78 0,80 75.0 the i-th Refinement Step the i-th Refinement Step the i-th Refinement Step the i-th Refinement Step (a) (c)
![](_page_4_Figure_0.jpeg)

Figure 4: Trends in average scores (provided by the evaluator agent) over the refinement steps. The trends for samples with three, four, five, and six refinement steps are illustrated in (a), (b), (c), and (d), respectively.

<span id="page-4-2"></span>

| Metric                                 | ACC. (%) |  |  |
|----------------------------------------|----------|--|--|
| CometKiwi                              | 56.0     |  |  |
| Evaluator Agent (Qwen2.5-72B-Instruct) | 92.5     |  |  |
| Evaluator Agent (GPT-40)               | 93.5     |  |  |

Table 2: Accuracy of automatic metrics for translation quality estimation (ACC.: accuracy).

by the evaluator agent) along with each refinement step. As shown in Figure 4, as the number of refinement steps increases, the average score generally increases, demonstrating that the refine loop could iteratively increase the quality of translations.

### <span id="page-4-0"></span>2.5 Quality Analyses

The Effectiveness of the Evaluator Agent. Previous work has shown that the state-of-the-art LLMs can be used as evaluators for various text generation tasks (Kocmi and Federmann, 2023; Wang et al., 2023; Li et al., 2024). To figure out the effectiveness of our evaluator agent, we randomly select 200 source sentences from DRT data, and for each of them, we further select its two translations as well as scores (provided by the evaluator agent) during refinement. We next employ human annotators to compare the two translations of each source sentence, and judge which translation is better, or two translations are similar in quality (annotation details can be found in Appendix B). After obtaining the quality labels, we calculate the accuracy of the evaluator agent according to its evaluation score. For comparison, we also calculate the accuracy of CometKiwi (Rei et al., 2022) and GPT-40 evaluator agent. As shown in Table 2, our evaluator agent achieves a high accuracy (92.5%), demonstrating its effectiveness in evaluating literature translation quality. Besides, the widely-used CometKiwi metric only achieves 56.0% accuracy. Thought CometKiwi is powerful in the general domain (e.g., news) (Kocmi and Federmann, 2023), its effectiveness in the literature domain is limited and unreliable, which is also pointed out by Karpinska and Iyyer (2023). Furthermore, the GPT-40 evaluator agent slightly outperforms the origin evaluator agent (with Qwen2.5-72B-Instruct backbone). Considering the tradeoff between cost and effectiveness, we finally decide to use Qwen2.5-72B-Instruct as our evaluator agent.

Translation Quality. Based on the effectiveness of the evaluator agent and the observation that evaluation scores of final translations typically reach 90.0 (c.f., Figure 4), we can ensure a high level of translation quality in the constructed data. According to the pre-defined scoring criteria of the evaluator agent (c.f., Appendix A.2), a score of 90.0 indicates excellent translations.

## 3 Experiments

### 3.1 Experimental Setups

Metrics. Following previous work, we adopt "BLEU" (Papineni et al., 2002), "CometKiwi" and "CometScore" (Rei et al., 2022) to evaluate the model translations. Among them, BLEU evaluates n-grams overlap between model translations and references, while CometScore evaluates the semantic similarity of model translations against references. CometKiwi uses a language model to judge whether a model translation conveys the semantics of the source sentence.

As pointed out by Karpinska and Iyyer (2023), BLEU and COMET may be ineffective for evaluating literature translation. Meanwhile, recent studies also show the strong ability of LLMs in NLP evaluation (Li et al., 2024). Therefore, we use evaluators implemented using GPT-40 in reference-based and reference-free styles, which we refer to as "GRB" and "GRF", respectively. The evaluation prompts borrow from Kocmi and Federmann (2023), and are illustrated in Appendix C. Further-

<span id="page-5-0"></span>

| Model                                    | reference-free     |                    |                    | reference-based    |                    |                    |  |
|------------------------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--|
| Model                                    | GEA                | GRF                | CometKiwi          | GRB                | BLEU               | CometScore         |  |
| Vanilla LLMs                             |                    |                    |                    |                    |                    |                    |  |
| Llama-3.1-8B-Instruct                    | 59.58              | 79.25              | 70.14              | 73.30              | 18.55              | 74.58              |  |
| Qwen2.5-7B-Instruct                      | 66.21              | 81.53              | 70.36              | 77.92              | 27.02              | 76.78              |  |
| Qwen2.5-14B-Instruct                     | 70.86              | 84.74              | 72.01              | 80.85              | 30.23              | 78.84              |  |
| Marco-o1-7B                              | 64.24              | 82.41              | 71.62              | 77.50              | 29.48              | 77.41              |  |
| QwQ-32B-preview                          | 75.50              | 86.31              | 71.48              | 83.08              | 27.46              | 78.68              |  |
| DeepSeek-R1-Distill-Llama-8B             | 56.89              | 76.31              | 67.13              | 69.49              | 15.83              | 71.82              |  |
| DeepSeek-R1-Distill-Qwen-7B              | 43.66              | 65.16              | 63.49              | 58.13              | 10.99              | 69.21              |  |
| DeepSeek-R1-Distill-Qwen-14B             | 70.64              | 83.92              | 71.01              | 80.29              | 25.55              | 77.66              |  |
| DeepSeek-R1-Distill-Qwen-32B             | 71.88              | 84.78              | 71.93              | 81.59              | 29.36              | 78.93              |  |
| SFT LLMs (w/o CoT)                       |                    |                    |                    |                    |                    |                    |  |
| Llama-3.1-8B-SFT                         | 69.33              | 84.10              | 70.25              | 80.18              | 30.03              | 78.26              |  |
| Qwen2.5-7B-SFT                           | 72.29              | 85.06              | 71.03              | 81.72              | 35.44              | 80.10              |  |
| Qwen2.5-14B-SFT                          | 74.53              | 85.66              | <u>72.08</u>       | <u>83.08</u>       | 37.63              | 80.82              |  |
| DRT                                      |                    |                    |                    |                    |                    |                    |  |
| DRT-8B (Backbone: Llama-3.1-8B-Instruct) | 69.65 <sup>†</sup> | 84.49 <sup>‡</sup> | 70.85 <sup>†</sup> | $80.80^{\dagger}$  | 32.67 <sup>†</sup> | 78.81 <sup>†</sup> |  |
| DRT-7B (Backbone: Qwen2.5-7B-Instruct)   | $75.05^{\dagger}$  | 85.57 <sup>‡</sup> | $71.78^{\dagger}$  | $82.38^{\dagger}$  | 35.54              | $80.19^{\ddagger}$ |  |
| DRT-14B (Backbone: Qwen2.5-14B-Instruct) | $77.41^{\dagger}$  | $87.19^{\dagger}$  | 72.11              | $83.20^{\ddagger}$ | <u>36.46</u>       | <u>80.64</u>       |  |

Table 3: Experimental results on literature translation. The **bold** and the <u>underline</u> denote the best and second-best performances, respectively. " $\dagger$ " and " $\ddagger$ " denote statistically significant better than the corresponding SFT LLMs (w/o CoT) with t-test p < 0.01 and 0.05, respectively.

more, as demonstrated in § 2.4, the GPT-40 evaluator agent achieves great accuracy in literature translation. We also leverage it as the evaluation metric in experiments, which is referred to as "*GEA*". Since GRB, GRF and GEA need the API costs, we randomly select 400 samples to conduct evaluation.

**Backbones.** We adopt the following three LLMs as the backbones of our DRT: Llama-3.1-8B-Instruct (Dubey et al., 2024), Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct (Yang et al., 2024b). All model checkpoints are publicly available.

For evaluation toolkits and the implementation details of all models, please refer to Appendix D.

#### 3.2 Comparison Models

Vanilla LLMs. We leverage vanilla Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct (Yang et al., 2024a) as the comparison models. Besides, six O1-like LLMs are also conducted as baselines: Marco-o1-7B (Zhao et al., 2024), QwQ-32B-preview (Qwen, 2024), DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-32B (Guo et al., 2025).

SFT LLMs (w/o CoT). We also fine-tune LLMs with only paired sentences of DRT training data (without thought). This setting allows LLMs to learn the mapping from source literature sentences to the corresponding Chinese translations directly. We

denote the fine-tuned LLMs as Llama-3.1-8B-SFT, Qwen2.5-7B-SFT and Qwen2.5-14B-SFT, serving as strong baselines in the experiments.

#### 3.3 Main Results

Table 3 shows the experimental results, we analyze the performance from the following aspects:

SFT LLMs (w/o CoT) vs. Vanilla LLMs. After instruction tuning on the paired sentences of our training data, SFT LLMs (w/o CoT) significantly outperform the corresponding vanilla LLMs. For example, Llama-3.1-8B-SFT outperforms Llama-3.1-8B-Instruct by 9.75 GEA, 4.85 GRF and 6.88 GRB. Qwen2.5-7B-SFT outperforms Qwen2.5-7B-Instruct by 6.08 GEA, 3.53 GRF and 3.80 GRB. This finding demonstrates the effectiveness of our multi-agent framework and the quality of the synthesized translation. Please also note that the final translations are synthesized by Qwen2.5-72B-Instruct, indicating that we can leverage off-theshelf open-source LLMs to collect high-quality literation translation data. And the data could help smaller LLMs (such as 7B and 14B ones) to boost their literature translation skills.

**DRT vs. Vanilla LLMs.** After fine-tuning on the long-thought MT training data, our DRT-series LLMs also significantly outperform the corresponding vanilla backbones. Particularly, DRT-14B outperforms QwQ-32B-preview and DeepSeek-R1-

<span id="page-6-0"></span>

| Model                | Flu.   | Sem.   | Lit.   |  |
|----------------------|--------|--------|--------|--|
| Qwen2.5-14B-Instruct | -0.353 | -0.363 | -0.442 |  |
| QwQ-32B-Preview      | -0.063 | 0.022  | -0.007 |  |
| Qwen2.5-14B-SFT      | 0.103  | 0.108  | 0.087  |  |
| DRT-14B              | 0.313  | 0.233  | 0.362  |  |

Table 4: Human evaluation results in terms of fluency, semantic accuracy and literary quality.

Distill-Qwen-32B in terms of all metrics, showing its effectiveness in literature MT.

DRT vs. SFT LLMs (w/o CoT). Using Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct as backbones, LLMs tuned with long thought achieve better performance than those tuned without long thought in terms of all metrics. For example, DRT-7B outperforms Qwen2.5-7B-SFT by 2.76 GEA, 0.51 GRF, 0.75 CometKiwi, 0.66 GRB, 0.10 BLEU and 0.09 CometScore. When using Qwen2.5-14B-Instruct as the backbone, we find that DRT-14B outperforms Qwen2.5-14B-SFT in terms of GEA, GRF, CometKiwi and GRB, but underperforms in terms of BLEU and CometScore. In detail, BLEU and CometScore evaluate the translations from the perspective of similarity between model translations and golden references. We conjecture that the higher BLEU and CometScore performance of Qwen2.5-14B-SFT is due to the model's ability to quickly learn domain-specific translations through tuning without long thoughts, allowing it to adapt to the literature translation more straightforwardly. However, training without long thoughts might lead the model to a sub-optimal solution, like learning shortcuts. When adopting evaluation metrics that are not significantly dependent on the golden references (i.e., GEA, GRF, CometKiwi and GRB), DRT-14B shows its superior performance. Note that although GRB is a reference-based metric, it does not assess the model translations simply based on how similar they are to the golden references.

**DRT vs. Commercial LLMs.** To give a deeper understanding of our DRT models' performance, we further compare DRT models with GPT-40 (OpenAI, 2024a) and o1-preview (OpenAI, 2024b). The experimental results and corresponding analyses are provided in Appendix E.

#### 3.4 Human Evaluation

We conduct human evaluation to further evaluate the performance of DRT-14B and strong baselines (Qwen2.5-14B-Instruct, QwQ-32B-Preview and Qwen2.5-14B-SFT). We randomly select 200 sam-

<span id="page-6-1"></span>> **[图片提取文字 (无描述)]:**
> Nama-3.1.88-587 0.71 0.53 Owen25-148-5FT) 0.94 DRT-8B 8.44 DRT-14B 7.27 13.06 10 12 14 Time Cost (s/case)
![](_page_6_Figure_7.jpeg)

Figure 5: Time cost during inference on the testing set.

ples from our test set, and employ three human evaluators with high levels of fluency in English and Chinese to assess the generated translations from three aspects: fluency (Flu.), semantic accuracy (Sem.) and literary quality (Lit.). Following the Best-Worst Scaling method (Kiritchenko and Mohammad, 2017), evaluators are asked to select the best and the worst generated translation on each aspect. The result scores are calculated based on the percentage of times each model is selected as best minus the times it is selected as worst. Thus, the final scores should range from -1 (worst) to 1 (best). As shown in Table 4, DRT-14B significantly outperforms these strong baselines, especially in the aspect of literary quality. These results demonstrate the superiority of our DRT models. The Fleiss' Kappa scores (Fleiss, 1971) of Flu., Sem. and Lit. are 0.75, 0.69 and 0.85, respectively, indicating a good inter-agreement among evaluators.

### 3.5 Inference Time Analysis

During evaluating LLMs' literature translation performance on our test set, we leverage vLLM to accelerate the model generation. A single NVIDIA A100 GPU (40G) is used to deploy each LLM. As shown in Figure 5, the average time costs of DRT models are significantly higher than LLMs (w/o CoT). This is because DRT models should first generate the long thought and then provide the final translation, thus needing more inference time ( $\times 11.9 \sim 13.9$ ). This also indicates that the O1-like LLMs may not be applicable to some scenarios with high real-time requirements.

### 3.6 Case Study

Table 5 provides some literature translation cases of Qwen2.5-14B-Instruct, QwQ-32B-Preview, Qwen2.5-14B-SFT and DRT-14B. With the help of long thought, the translations of DRT-14B align

<span id="page-7-0"></span>

| Source Sentence                                                                                                                                                                                                                                                                                 | Qwen2.5-14B-Instruct                                                       | QwQ-32B-Preview                                                                                                      | Qwen2.5-14B-SFT                                                                             | DRT-14B                                                                      |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| And if their words had taken<br>another turnif he himself had<br>been less fastidious about intrud-<br>ing on another man's secretsit<br>was cruel to think how thin a film<br>had shut out rescue from all this<br>guilt and misery.                                                           | 而如果他们的言辞转向了<br>另一个方向如果他自己对侵扰他人秘密不那么介意                                      | 而且, <mark>假如他们的谈话走向不同</mark> 假如他自己不对干涉他人私事如此谨慎那时他会意识到,只要一层薄尊的屏障就阻隔了救援,使得所有这些罪恶和痛苦无法解脱,这真是残忍。                         | 如 果他 们 的 言 辞 稍 有 不同,若他不那么介意侵扰别人的隐私,想到那层薄纱如何隔绝了救援,真是残忍。                                      | 若他们的言辞稍作转变若他本人不那么介意侵入他人的心扉想到这层薄薄的隔阂竟将所有内疚与痛苦隔离于救援之外,实为残忍。                    |
| This cold officer upon a mon-<br>ument, who dropped epithets<br>unconcernedly down, would be<br>finer as a dead man, he thought.                                                                                                                                                                | 他认为,这个站在纪念碑<br>上的冷漠官员,若死了会更<br>好,他不带任何感情地抛下<br>了一些称呼。                      | 这个冷冰冰的官员站在纪念碑上,毫不在意地抛下一些称号,他想,如果作为一个死人会更出色。                                                                          | 这位冷酷的官员站在纪念碑上,随意地抛出贬义词,他觉得,作为死人会更显尊贵。                                                       | 纪念碑上的冷淡官员,漫不<br>经心地吟咏那些修饰语,他<br>心想,若化为亡者,或许更<br>显尊贵。                         |
| I see on your cheek two tears which I know are hot as two sparks, and salt as two crystals of the sea.                                                                                                                                                                                          | 我看见你脸颊上有两滴泪<br>水,我知道它们像两簇火花<br>一样滚烫,像两颗海水结晶<br>一样咸。                        | 我见你面颊两颗泪,温热如<br>火花,咸涩如海晶。                                                                                            | 我见你脸颊上两颗泪珠,热<br>如火花,咸似海盐。                                                                   | 我看见你脸颊上的两滴眼泪,它们既如两簇火花般炽热,又似两粒海盐般咸涩。                                          |
| The night wore out, and, as he stood upon the bridge listening to the water as it splashed the river-walls of the Island of Paris, where the picturesque confusion of houses and cathedral shone bright in the light of the moon, the day came coldly, looking like a dead face out of the sky. | 夜深了,当他站在桥上,听着水花拍打巴黎岛河岸的声音,月光下房屋与教堂错落有致的景象闪耀着光芒时,冷冰冰的白天来临了,仿佛是从天空中露出的一张死人脸。 | 夜晚消逝了,当他站在桥<br>上,聆听水流拍打着巴黎与<br>大美丽的房屋,那里的房屋而又<br>繁堂构成了一幅混乱而又<br>哪的图景,在月光下闪耀酷<br>明亮的光芒,而白天则冷酷<br>地到来,仿佛是天空中一张<br>死人脸。 | 夜幕渐渐消散,他站在桥上,聆听水花轻拍巴黎岛的河岸。月光下,岛上房屋与河岸。月光下,岛上房屋与大教堂交织成一幅迷人的画卷,显得格外明亮。清晨悄然而至,宛如从天际浮现的一张冷峻的脸庞。 | 夜色渐渐消逝,他站在桥上,倾听河水轻拍巴黎城岛的河岸。月光下,房屋看迷大教堂精蓉交织,闪耀着迷人的光辉。白日冷冰冰地降临,宛如从天际浮现的一张死寂的脸。 |

Table 5: Case Studies of literature translation. Green indicates good translations, while red indicates bad ones.

more closely with the conventions of the Chinese language and exhibit a greater literary quality. In addition to DRT-14B, some translation snippets of other LLMs can also show a great performance (marked in green). This indicates that vanilla LLMs might have the capability to translate literature, and long thought could further activate this capability.

## 4 Related Work

O1-like LLMs. Recently, O1-like LLMs have shown great performance in reasoning tasks, especially math and coding tasks. After the emergency of OpenAI O1 model (OpenAI, 2024b), many efforts are given in reproducing OpenAI O1. For example, Qin et al. (2024) propose journey learning, a training paradigm, to encourage LLMs to learn not just shortcuts, but the complete exploration process. Huang et al. (2024) explore the data distillation from existing O1-like models, and show the effectiveness of data distillation. Zhang et al. (2024) leverage Monte Carlo Tree Search (MCTS) to synthesize reasoning-enhanced code data, and train O1-Coder. Marco-o1 (Zhao et al., 2024) is proposed to deal with open-ended text generation. More recently, DeepSeek-R1 (Guo et al., 2025) and Kimi K1.5 (Team et al., 2025) are proposed, and show their promising reasoning ability.

**Literature Translation.** Different from translating standard MT corpora (*e.g.*, news articles), translating literature books is more difficult since it often requires equivalence beyond the word level (Thai et al., 2022). Besides, it is also difficult to evaluate literature translation using automatic metrics,

and previous literature translation work typically relies on human evaluation (Fonteyne et al., 2020; Karpinska and Iyyer, 2023). Due to its difficulty, early work is limited to small-scale attempts (Genzel et al., 2010; Jones and Irvine, 2013; Besacier and Schwartz, 2015; Toral et al., 2018). Recently, Karpinska and Iyyer (2023) utilize LLMs to perform literature translation, and show that discourse-level LLM translators achieve better performances compared with sentence-level approaches. Thai et al. (2022) introduce Par3 to benchmark LLMs' literature translation capability from non-English languages to English.

#### 5 Conclusion

In this paper, we introduce DRT, an attempt to bring the success of long-thought reasoning to neural machine translation (MT). Specifically, we synthesize the machine translation long-thought samples by a designed multi-agent framework and GPT-40 reformulation. To collect the source sentences that are suitable for translation via long thought, we mine sentences with similes or metaphors from existing literature books. To synthesize the long thought machine translation process for these sentences, a translator, an advisor and an evaluator collaborate to translate the source sentence iteratively. Based on the synthesized data, we train DRT models. Extensive experiments on literature translation demonstrate the effectiveness of DRT models in terms of automatic evaluation. Case study and human evaluation further verify their superiority.

# Limitations

While we show the effectiveness of long thought in MT, there are some limitations worth noting: (1) We focus on English-to-Chinese translation in this work, and future work could extend the data and the method to other translation directions. (2) There is still a lack of accurate automatic evaluation metrics for literary translation. Previous literature translation work typically relies on human evaluation [\(Fonteyne et al.,](#page-8-9) [2020;](#page-8-9) [Karpinska](#page-8-5) [and Iyyer,](#page-8-5) [2023\)](#page-8-5), and points out that BLEU and Comet might not be suitable for evaluating literature translation [\(Karpinska and Iyyer,](#page-8-5) [2023\)](#page-8-5). This is because literary translations carry the responsibility of both semantic and critical interpretation, as they must address the challenge of achieving equivalence that often extends beyond the level of individual words [\(Thai et al.,](#page-9-14) [2022\)](#page-9-14).

# Ethical Considerations

We discuss the main ethical considerations of DRT models as follows: (1) Copyright. We mine literature sentences from 400 English books provided by the Project Gutenberg public-domain book repository[6](#page-8-13) , where the books are typically more than fifty years old and their copyrights have expired. The book data also has been extracted and released by [Kryscinski et al.](#page-8-3) [\(2022\)](#page-8-3). Therefore, we can construct DRT data based on these books, and further release our synthesized data. (2) Licenses. We will release our model checkpoints and synthesized data under CC-BY-NC-SA 4.0 license.

# References

- <span id="page-8-12"></span>Laurent Besacier and Lane Schwartz. 2015. Automated translation of a literary work: a pilot study. In *Fourth Workshop on Computational Linguistics for Literature-co-located with NAACL 2015*.
- <span id="page-8-1"></span>Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. 2024. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. *arXiv preprint arXiv:2412.21187*.
- <span id="page-8-2"></span>Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*.

- <span id="page-8-8"></span>Joseph L Fleiss. 1971. Measuring nominal scale agreement among many raters. *Psychological bulletin*, 76(5):378.
- <span id="page-8-9"></span>Margot Fonteyne, Arda Tezcan, and Lieve Macken. 2020. [Literary machine translation under the mag](https://aclanthology.org/2020.lrec-1.468)[nifying glass: Assessing the quality of an NMT](https://aclanthology.org/2020.lrec-1.468)[translated detective novel on document level.](https://aclanthology.org/2020.lrec-1.468) In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 3790–3798, Marseille, France. European Language Resources Association.
- <span id="page-8-10"></span>Dmitriy Genzel, Jakob Uszkoreit, and Franz Josef Och. 2010. "poetic" statistical machine translation: rhyme and meter. In *Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing*, pages 158–166.
- <span id="page-8-6"></span>Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*.
- <span id="page-8-0"></span>Zhen Huang, Haoyang Zou, Xuefeng Li, Yixiu Liu, Yuxiang Zheng, Ethan Chern, Shijie Xia, Yiwei Qin, Weizhe Yuan, and Pengfei Liu. 2024. O1 replication journey–part 2: Surpassing o1-preview through simple distillation, big progress or bitter lesson? *arXiv preprint arXiv:2411.16489*.
- <span id="page-8-11"></span>Ruth Jones and Ann Irvine. 2013. The (un) faithful machine translator. In *Proceedings of the 7th Workshop on Language Technology for Cultural Heritage, Social Sciences, and Humanities*, pages 96–101.
- <span id="page-8-5"></span>Marzena Karpinska and Mohit Iyyer. 2023. [Large lan](https://doi.org/10.18653/v1/2023.wmt-1.41)[guage models effectively leverage document-level](https://doi.org/10.18653/v1/2023.wmt-1.41) [context for literary translation, but critical errors per](https://doi.org/10.18653/v1/2023.wmt-1.41)[sist.](https://doi.org/10.18653/v1/2023.wmt-1.41) In *Proceedings of the Eighth Conference on Machine Translation*, pages 419–451, Singapore. Association for Computational Linguistics.
- <span id="page-8-7"></span>Svetlana Kiritchenko and Saif Mohammad. 2017. [Best](https://doi.org/10.18653/v1/P17-2074)[worst scaling more reliable than rating scales: A case](https://doi.org/10.18653/v1/P17-2074) [study on sentiment intensity annotation.](https://doi.org/10.18653/v1/P17-2074) In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 465–470, Vancouver, Canada. Association for Computational Linguistics.
- <span id="page-8-4"></span>Tom Kocmi and Christian Federmann. 2023. Large language models are state-of-the-art evaluators of translation quality. In *24th Annual Conference of the European Association for Machine Translation*, page 193.
- <span id="page-8-3"></span>Wojciech Kryscinski, Nazneen Rajani, Divyansh Agarwal, Caiming Xiong, and Dragomir Radev. 2022. [BOOKSUM: A collection of datasets for long-form](https://doi.org/10.18653/v1/2022.findings-emnlp.488) [narrative summarization.](https://doi.org/10.18653/v1/2022.findings-emnlp.488) In *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 6536–6558, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

<span id="page-8-13"></span><sup>6</sup><https://www.gutenberg.org/>

- <span id="page-9-18"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*.
- <span id="page-9-8"></span>Zhen Li, Xiaohan Xu, Tao Shen, Can Xu, Jia-Chen Gu, Yuxuan Lai, Chongyang Tao, and Shuai Ma. 2024. [Leveraging large language models for NLG evalua](https://doi.org/10.18653/v1/2024.emnlp-main.896)[tion: Advances and challenges.](https://doi.org/10.18653/v1/2024.emnlp-main.896) In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 16028–16045, Miami, Florida, USA. Association for Computational Linguistics.
- <span id="page-9-5"></span>OpenAI. 2024a. Gpt-4o system card. *arXiv preprint arXiv:2410.21276*.
- <span id="page-9-0"></span>OpenAI. 2024b. Learning to reason with large language models. [https://openai.com/index/](https://openai.com/index/learning-to-reason-with-llms/) [learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/).
- <span id="page-9-10"></span>Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [Bleu: a method for automatic evalu](https://doi.org/10.3115/1073083.1073135)[ation of machine translation.](https://doi.org/10.3115/1073083.1073135) In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.
- <span id="page-9-1"></span>Yiwei Qin, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, Yixin Ye, Weizhe Yuan, Hector Liu, Yuanzhi Li, et al. 2024. O1 replication journey: A strategic progress report–part 1. *arXiv preprint arXiv:2410.18982*.
- <span id="page-9-12"></span>Team Qwen. 2024. Qwq: Reflect deeply on the boundaries of the unknown. *Hugging Face*.
- <span id="page-9-17"></span>Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 3505–3506.
- <span id="page-9-9"></span>Ricardo Rei, Marcos Treviso, Nuno M. Guerreiro, Chrysoula Zerva, Ana C Farinha, Christine Maroti, José G. C. de Souza, Taisiya Glushkova, Duarte Alves, Luisa Coheur, Alon Lavie, and André F. T. Martins. 2022. [CometKiwi: IST-unbabel 2022 sub](https://aclanthology.org/2022.wmt-1.60)[mission for the quality estimation shared task.](https://aclanthology.org/2022.wmt-1.60) In *Proceedings of the Seventh Conference on Machine Translation (WMT)*, pages 634–645, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.
- <span id="page-9-13"></span>Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. *arXiv preprint arXiv:2501.12599*.

- <span id="page-9-14"></span>Katherine Thai, Marzena Karpinska, Kalpesh Krishna, Bill Ray, Moira Inghilleri, John Wieting, and Mohit Iyyer. 2022. [Exploring document-level literary ma](https://doi.org/10.18653/v1/2022.emnlp-main.672)[chine translation with parallel paragraphs from world](https://doi.org/10.18653/v1/2022.emnlp-main.672) [literature.](https://doi.org/10.18653/v1/2022.emnlp-main.672) In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 9882–9902, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
- <span id="page-9-15"></span>Antonio Toral, Martijn Wieling, and Andy Way. 2018. Post-editing effort of a novel with statistical and neural machine translation. *Frontiers in Digital Humanities*, 5:9.
- <span id="page-9-4"></span>Raymond Van den Broeck. 1981. The limits of translatability exemplified by metaphor translation. *Poetics today*, 2(4):73–87.
- <span id="page-9-7"></span>Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. 2023. [Is ChatGPT a good NLG evalua](https://doi.org/10.18653/v1/2023.newsum-1.1)[tor? a preliminary study.](https://doi.org/10.18653/v1/2023.newsum-1.1) In *Proceedings of the 4th New Frontiers in Summarization Workshop*, pages 1–11, Singapore. Association for Computational Linguistics.
- <span id="page-9-6"></span>An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024a. Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*.
- <span id="page-9-11"></span>An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024b. Qwen2. 5 technical report. *arXiv preprint arXiv:2412.15115*.
- <span id="page-9-2"></span>Yuxiang Zhang, Shangxi Wu, Yuqi Yang, Jiangming Shu, Jinlin Xiao, Chao Kong, and Jitao Sang. 2024. o1-coder: an o1 replication for coding. *arXiv preprint arXiv:2412.00154*.
- <span id="page-9-3"></span>Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. 2024. Marco-o1: Towards open reasoning models for open-ended solutions. *arXiv preprint arXiv:2411.14405*.
- <span id="page-9-16"></span>Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. 2024. [LlamaFactory: Unified](https://doi.org/10.18653/v1/2024.acl-demos.38) [efficient fine-tuning of 100+ language models.](https://doi.org/10.18653/v1/2024.acl-demos.38) In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)*, pages 400–410, Bangkok, Thailand. Association for Computational Linguistics.

# A Prompt in Data Synthesis

### <span id="page-10-0"></span>A.1 Prompts in Literature Book Mining

SYSTEM PROMPT:

You are assigned to translate an English literary work into Chinese. The text may include descriptions or expressions that embody English cultural nuances, which may not resonate with Chinese language habits. In such instances, a literal translation may not be appropriate; instead, these sentences should be paraphrased to convey their intended meaning effectively.

USER PROMPT:

The English sentence is provided as follows: <english sentence> {sentence}

</english sentence>

Please begin by assessing whether the English sentence contains any metaphors or similes. If there are none, respond with "no metaphors and no similes."

If the English sentence does contain metaphors or similes, provide a literal translation of them, and then evaluate whether the literal translation is appropriate and easy for Chinese natives to understand.

If it is suitable, format your response as follows (two lines):

"your literal translation for metaphors/similes here (in Chinese)"

"suitable"

If it is unsuitable, please provide the reason for the unsuitability. Format your response as follows (three lines):

"your literal translation for metaphors/similes here (in Chinese)"

"unsuitable"

"reason for unsuitability here (in Chinese)"

### <span id="page-10-1"></span>A.2 Prompts in Multi-Agent Framework

## Translator Agent (Word-level translation)

Given an English sentence, identify the important words (usually nouns, verbs, technical terms, and named entities that require special attention in translation) and translate them into Chinese. Output the translations in JSON format, for example:

{"EnglishWord1": "ChineseTranslation", "English-Word2": "ChineseTranslation"}

The Chinese translations can be a single translation or multiple options as deemed appropriate.

## Translator Agent (Preliminary translation)

SYSTEM PROMPT:

Given an English sentence and a JSON object containing potential translations of important keywords, produce a Chinese literal translation of the entire sentence. Please directly output the Chinese translation without any descriptions.

USER PROMPT:

<English Sentence> {sentence} </English Sentence> <Potential Keyword Translation> {keyword translation}

</Potential Keyword Translation>

## Translator Agent (Refinement translation)

In the refine loop, the translator agent receives the feedback of the previous translation, and then provides a new translation. The prompt is a multiturn dialogue between the translator and advisor, where the system prompt is the same as the preliminary translation.

### Advisor Agent

Please rate the Chinese translation of the following English text and provide your comments and suggestions.

## Evaluator Agent

SYSTEM PROMPT:

Please evaluate the following Chinese translation of an English text. Rate the translation on a scale of 0 to 100, where:

- 10 points: Poor translation; the text is somewhat understandable but contains significant errors and awkward phrasing that greatly hinder comprehension for a Chinese reader.
- 30 points: Fair translation; the text conveys the basic meaning but lacks fluency and contains several awkward phrases or inaccuracies, making it challenging for a Chinese reader to fully grasp the intended message.
- 50 points: Good translation; the text is mostly fluent and conveys the original meaning well, but may have minor awkwardness or slight inaccuracies that could confuse a Chinese reader.
- 70 points: Very good translation; the text is smooth and natural, effectively conveying the intended meaning, but may still have minor issues that could slightly affect understanding for a Chinese reader.
- 90 points: Excellent translation; the text is fluent and natural, conveying the original meaning clearly and effectively, with no significant issues that would hinder understanding for a Chinese reader.

Please provide the reason first, followed by a score. Format your evaluation in the JSON structure below: {"reason": "reason for the score", "score": int}

## <span id="page-11-0"></span>A.3 Prompts in Thought Reformulation

A student is engaged in the task of translating an English sentence into Chinese.

The English sentence is as follows: <English Sentence> {sentence} </English Sentence>

This student constantly thinks about and optimizes his translation. The whole process is shown as follows:

<Translation Process> {translation process} </Translation Process>

Please polish the whole translation process into a long first-person self-reflection description (use the present tense).

The self-reflection should begin with selecting the keywords from the English sentence, translating the keywords, and then attempt to translate the whole sentence, and then think about whether the translation is good or not, and iteratively make translation attempts. Finally, make a final translation decision.

Output the self-reflection description directly without any additional descriptions or explanations. Each line in the self-reflection description can be regarded as a reasoning step to the translation.

# <span id="page-11-1"></span>B Details of Human Annotation

In Section [2.4,](#page-3-0) we employ human annotation to provide the quality comparison between two translations for a source sentence. Specifically, we employ three Chinese master students with high levels of fluency in both English and Chinese as our human annotators. For each sample, we give the source sentence and its two translation (without the scores provided by our evaluator agent) to all three annotators, and every annotator should provide one of the following judgments: (1) the first translation is better than the second one; (2) the second translation is better than the first one; (3) two translation are similar in quality. During annotation, we encourage the annotators to give differentiated judgments, *i.e.*, judgment (1) or judgment (2). If three annotators give the same judgment for a sample, the judgment will be regarded as the final label. Otherwise, its label will be decided by a group meeting of all three annotators and a senior data scientist.

# <span id="page-11-2"></span>C GPT-4o Evaluator

For GRB and GRF, we prompt GPT-4o (2024-08- 06 version) as the MT evaluator in the reference-

based and reference-free manners, respectively. The corresponding prompts borrow from [Kocmi](#page-8-4) [and Federmann](#page-8-4) [\(2023\)](#page-8-4), and make some adaptions to literature translation.

## GRB Prompt:

Score the following translation from English to Chinese with respect to the human reference on a continuous scale from 0 to 100, where score of zero means "no meaning preserved" and score of one hundred means "perfect preservation of meaning, with faithfulness, expressiveness, and elegance".

English source: {src}

Chinese human reference: {ref} Chinese translation: {hyp}

Score:

## GRF Prompt:

Score the following translation from English to Chinese on a continuous scale from 0 to 100, where score of zero means "no meaning preserved" and score of one hundred means "perfect preservation of meaning, with faithfulness, expressiveness, and elegance".

English source: {src} Chinese translation: {hyp}

Score:

# <span id="page-11-3"></span>D Implementation Details.

Automatic Evaluation. To calculate CometKiwi and CometScore, we leverage the official codes[7](#page-11-4) and the official models[8](#page-11-5) . To calculate the BLEU score, we use the *sacrebleu* toolkit[9](#page-11-6) to calculate the corpus-level BLEU.

Training Details. Llama-Factory [\(Zheng et al.,](#page-9-16) [2024\)](#page-9-16) is used to instruct-tune LLMs. All LLMs are tuned on 8×NVIDIA A100 GPUs (40G) with 1e-5 learning rate and 8 (8×1) batch size. We use the DeepSpeed ZeRO-3 optimization [\(Rasley et al.,](#page-9-17) [2020\)](#page-9-17). Following [Qin et al.](#page-9-1) [\(2024\)](#page-9-1), we set the number of training epochs to 3, and the training process costs 70 GPU hours and 124 GPU hours for 7B and 14B models, respectively.

Inference Details. When evaluating model performance on the test set, we use vLLM toolkit [\(Kwon](#page-9-18) [et al.,](#page-9-18) [2023\)](#page-9-18) to accelerate the model generation. We

<span id="page-11-5"></span><span id="page-11-4"></span><sup>7</sup><https://github.com/Unbabel/COMET>

<sup>8</sup>[https://huggingface.co/Unbabel/](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) [wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) and [https://huggingface.](https://huggingface.co/Unbabel/wmt22-comet-da) [co/Unbabel/wmt22-comet-da](https://huggingface.co/Unbabel/wmt22-comet-da)

<span id="page-11-6"></span><sup>9</sup><https://github.com/mjpost/sacrebleu>

<span id="page-12-2"></span>

| Model                                    | reference-free |              |           | reference-based |              |              |  |
|------------------------------------------|----------------|--------------|-----------|-----------------|--------------|--------------|--|
| 112001                                   | GEA            | GRF          | CometKiwi | GRB             | BLEU         | CometScore   |  |
| Commercial LLMs                          |                |              |           |                 |              |              |  |
| GPT-40                                   | 71.88          | 85.57        | 73.01     | 82.78           | 34.51        | 79.41        |  |
| o1-preview                               | 78.01          | <u>87.11</u> | 73.70     | 83.86           | 30.65        | 80.12        |  |
|                                          |                | DRT          |           |                 |              |              |  |
| DRT-8B (Backbone: Llama-3.1-8B-Instruct) | 69.65          | 84.49        | 70.85     | 80.80           | 32.67        | 78.81        |  |
| DRT-7B (Backbone: Qwen2.5-7B-Instruct)   | 75.05          | 85.57        | 71.78     | 82.38           | <u>35.54</u> | <u>80.19</u> |  |
| DRT-14B (Backbone: Qwen2.5-14B-Instruct) | <u>77.41</u>   | 87.19        | 72.11     | <u>83.20</u>    | 36.46        | 80.64        |  |

Table 6: Experimental results of comparing DRT with commercial LLMs. The **bold** and the <u>underline</u> denote the best and second-best performances, respectively.

use the sampling decoding strategy with 0.1 temperature, and set the repetition penalty to 1.05. For DeepSeek-R1 series (DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B and DeepSeek-R1-Distill-Qwen-32B), we follow the instruction <sup>10</sup> to enforce them to avoid blank thinking. All experimental results listed in this paper are the average of 3 runs.

# <span id="page-12-0"></span>E Comparison with Commercial LLMs

As shown in Table 6, DRT-14B achieves competitive results with o1-preview, showing its superiority. Additionally, we observe that o1-preview significantly outperforms GPT-40 in terms of GEA. This finding highlights the effectiveness of long thought in machine translation. When applied to appropriate translation contexts, long thought can further enhance the authenticity of translations.

<span id="page-12-1"></span><sup>10</sup>https://github.com/deepseek-ai/
DeepSeek-R1