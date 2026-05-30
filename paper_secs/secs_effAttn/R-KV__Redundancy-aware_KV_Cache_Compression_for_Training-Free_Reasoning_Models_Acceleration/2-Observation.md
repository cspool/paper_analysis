# 2 Observation

## <span id="page-1-0"></span>2.1 Redundancy in Reasoning Models

As noted in [\[2\]](#page-10-1), reasoning models often generate a detailed chain of thoughts and multiple reflection steps, resulting in significantly longer responses than standard models. [Figure 2](#page-2-1) shows that both reasoning models (i.e., DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Qwen-14B) generate more than 8× longer generation output compared to the ground truth on two popular math reasoning datasets. However, not all of the additional tokens contribute meaningful content, as much of the decoded context is dominated by repetition. [Figure 2](#page-2-1)

<span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 2: Comparison of generation length and average 1-/2-gram frequency for reasoning models and ground truth of MATH-500 [8] and AIME 2024 [9]. Reasoning models generate substantially longer responses with  $8-14 \times$  more tokens, and show higher word repetition with  $5-7 \times$  higher frequency.

also shows that the average frequency of 1- to 2-grams is consistently higher in the generation output of reasoning models than in ground truth, indicating greater repetitions in the generated outputs of reasoning models.

## <span id="page-2-0"></span>2.2 Failure of Existing KV Compression Methods to Handle Redundancy

Most existing KV cache compression methods prioritize token selection based primarily on tokens' contextual importance, typically measured through attention scores between key and query tokens [3, 4]. While this approach effectively retains critical context, it fails to account for redundancy—particularly problematic in reasoning models. In such models, we find that repetitive content often receives disproportionately high attention scores, as it closely mirrors previously generated repetitive text. As a result, redundant tokens are excessively retained, unnecessarily inflating the KV cache size without providing additional meaningful new information. In Figure 3, we visualize the cached tokens (inside red boxes) selected by a popular attention-based KV cache method (i.e., SnapKV), showing many repetitions related to self-reflection and conclusion to the final answer.

```
You are given a math problem. Problem: In Mr. Roper's class of 30 students,
 [Question and Instruction - 102 words]
First, the problem says that there are 30 students in total in the class. Out of
[Think - 203 words]
[Reflection for 13 times and 581 words in total]
 But wait, the ... So, 10% of 30 is 3. So 3 students are leaving early.
 Think - 36 words]
But in the initial problem...So
[Think - 42 words]
    wait, the ...30 is 3. So 3
 But in the initial problem, the... So 3
             .. early?" So, 10% of 30 is 3. So 3 students are leaving early.
                    So. 10% of 30 is 3. So
But wait, the user wrote: ...10% of 30 is 3, So, 3 students are leaving early.
                   of 30 is 3. Therefore, 3 students are leaving early.
I think that's all. The calculation is straightforward: 10% of 30 is 3.
```

Figure 3: KV selected by SnapKV. SnapKV suffers from redundancy in reasoning models. Black tokens are not selected by SnapKV; brighter colors reflect higher attention scores. Blue tokens are omitted output.

