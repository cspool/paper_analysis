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

