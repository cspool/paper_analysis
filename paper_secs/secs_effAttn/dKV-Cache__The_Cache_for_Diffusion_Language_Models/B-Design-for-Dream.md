# B Design for Dream

Dream has a different caching strategy with LLaDA. The main reason for this is that Dream is adapted from pre-trained autoregressive models, which would make the position of output align with the probability of the next token, instead of the current token in the traditional setting of masked diffusion models. This would make a difference in the caching strategy and we investigate between different designs of those caching strategies:

## <span id="page-13-0"></span>**Algorithm 1** Pseudo code for dKV-Cache-Decode. We take step t as an example.

**Require:** Sequence  $\mathbf{x}_{c(t)}^{1:L}$  at step t (Simplied as  $\mathbf{x}$ ), position index of masked tokens  $\mathcal{M}_t$ , cached Key  $\mathbf{K}_{t-1}^{\mathcal{I}\setminus\mathcal{M}_{t-1}}$  and Value  $\mathbf{V}_{t-1}^{\mathcal{I}\setminus\mathcal{M}_{t-1}}$ 

- 1:  $\mathbf{x}' \leftarrow \mathbf{x}[\mathcal{M}_{t-1}]$  $\triangleright \mathcal{M}_{t-1}$ : t-1 for one-step shift
- 2:  $\mathbf{PE}' \leftarrow [\mathbf{PE}[\mathcal{I} \setminus \mathcal{M}_{t-1}]; \mathbf{PE}[\mathcal{M}_{t-1}]] \triangleright \text{Positional embeddings: cached on left, uncached on right}]$
- $\, \triangleright \, \mathcal{T} \colon \text{Calculation in Transformer to get } Q, \, K \text{ and } V$
- 3:  $\mathbf{Q}_{t}^{\mathcal{M}_{t}}, \mathbf{K}_{t}^{\mathcal{M}_{t}}, \mathbf{V}_{t}^{\mathcal{M}_{t}} \leftarrow \mathcal{T}(\mathbf{x}')$   $\triangleright \mathcal{T}$ : Calculation in Transet:  $\mathbf{K}_{t}^{\mathcal{I}} \leftarrow \operatorname{Concat}\left(\mathbf{K}_{t-1}^{\mathcal{I} \setminus \mathcal{M}_{t-1}}, \mathbf{K}_{t}^{\mathcal{M}_{t-1}}\right), \mathbf{V}_{t}^{\mathcal{I}} \leftarrow \operatorname{Concat}\left(\mathbf{V}_{t-1}^{\mathcal{I} \setminus \mathcal{M}_{t-1}}, \mathbf{V}_{t}^{\mathcal{M}_{t-1}}\right)$
- 5:  $\mathbf{K}_{t}^{\mathcal{I} \setminus \mathcal{M}_{t}} \leftarrow \text{Reorder}(\mathbf{K}_{t}^{\mathcal{I}}, I'), \mathbf{V}_{t}^{\mathcal{I} \setminus \mathcal{M}_{t}} \leftarrow \text{Reorder}(\mathbf{V}_{t}^{\mathcal{I}}, I')$  $\triangleright \mathcal{I}'$ : The index of  $\mathcal{I} \setminus \mathcal{M}_t$  in the  $\left[\mathbf{x}[\mathcal{I}\setminus\mathcal{M}_{t-1}]\;;\;\mathbf{x}[\mathcal{M}_{t-1}]\right]$
- 6:  $p' \leftarrow \mathcal{A}(\mathbf{Q}_t^{\mathcal{M}_t}, \mathbf{K}_t^{\mathcal{I}}, \mathbf{V}_t^{\mathcal{I}})$ 7:  $p \leftarrow \operatorname{Scatter}(p', \mathcal{M}_{t-1})$

▶ Put the token logits back to the original position

8: **Return** p,  $\mathbf{K}_t^{\mathcal{I} \setminus \mathcal{M}_t}$ ,  $\mathbf{V}_t^{\mathcal{I} \setminus \mathcal{M}_t}$ 

<span id="page-13-1"></span>![](_page_13_Figure_11.jpeg)

Figure 6: Three variants for the caching strategy for diffusion language models adapted from autoregressive language models, which would have shifted output position.

- Un-Shift, Figure 6(a): We cache the unshifted token representations. Specifically, for the t-th token, we store its key and value as  $\mathbf{K}^t$  and  $\mathbf{V}^t$  at position t.
- Right-Shift, Figure 6(b): Given that the hidden state is highly sensitive to changes in input, we also explore a right-shifted variant. Here, for the t-th token, we cache  $K^{t+1}$  and  $V^{t+1}$ .
- Un&Right-Sift, Figure 6(c): We introduce a stricter variant where caching is conditioned on both input stability and decoding completion. For the t-th token, we cache its features only after its input is fixed and it has been decoded.

The one-step shift is still used here. For example, in the right-shift variant, the t-th token is fed into the model at position t+1 in the next step, and we cache its output  $\mathbf{K}^{t+1}$  and  $\mathbf{V}^{t+1}$  then. The results are shown in Table 4, where Un&right-shift would have the best performance, and the right shift would largely harm the model performance. However, we use the Un-Shift in our main experiment, since Un&right-shift is incompatible with the above concat\_reorder.

<span id="page-13-2"></span>Table 4: Comparison between different types of caching strategy for Dream-Base-7B.

|       | Un-Shift | Right-Shift | Un&Right Shift |
|-------|----------|-------------|----------------|
| MMLU  | 71.78    | 64.60       | 71.73          |
| GSM8K | 76.34    | 32.68       | 77.71          |

## **Evaluation Details**

#### C.1 For LLaDA

We re-implemented the evaluation of LLaDA on those reported datasets. We generate and extract the final answer instead of comparing the log prob in the multiple-choice question. Thus, the result of MMLU and GPQA is lower than reported since the model sometimes cannot generate the answer in the given format or does not generate the answer. We show the configuration of each experiment in Table 5

<span id="page-14-0"></span>

| Remasking | Base (random / confidence) Configuration | Few-Steps<br>(random)<br>Steps T | dKV-Cache-Greedy<br>(random)<br>Cache Interval | dKV-Cache-Greedy<br>(random)<br>Window Size | Half-Steps<br>(confidence)<br>Steps T | dKV-Cache-Decode<br>(confidence)<br>Cache Interval |
|-----------|------------------------------------------|----------------------------------|------------------------------------------------|---------------------------------------------|---------------------------------------|----------------------------------------------------|
| MMLU      | L=32, T=32, B=16                         | T=20                             | 2                                              | 4                                           | T=16                                  | 8                                                  |
| GSM8K     | L=256, T=256, B=32                       | T=160                            | 2                                              | 4                                           | T=128                                 | 8                                                  |
| Math500   | L=256, T=256, B=64                       | T=160                            | 2                                              | 4                                           | T=128                                 | 8                                                  |
| GPQA      | L=128, T=128, B=64                       | T=80                             | 2                                              | 4                                           | T=64                                  | 8                                                  |

2

2

4

2

T=256

T=256

8

8

Table 5: Configurations of experiments on LLaDA-Instruct.

#### C.2 For Dream

HumanEval

**MBPP** 

We follow the original evaluation pipeline for Dream<sup>4</sup> and we also adopt two datasets, MMLU and GPQA, to generate the answer instead of comparing the probabilities. We follow all the hyperparameters set in the evaluation script, including the temperature, the remasking strategy, top\_p and the number of few-shot in in-context learning.

## D Impact of batch size on speed

|| L=512, T=512, B=32

L=512, T=512, B=32

T = 320

T = 320

![](_page_14_Figure_6.jpeg)

Figure 7: Impact of batch size on decoding speed. Evaluated on LLaDA with a single NVIDIA H20; prefill length fixed at 100 tokens.

Our inference pipeline relies heavily on indexing operations, gathers and scatters, that generate a stream of small, non-contiguous memory accesses. At a batch size of one, these uncoalesced reads make the inference memory-bound. As a result, the GPU's compute units sit idle waiting for data. In some cases with the batch size equal to 1, inference with caching can actually underperform the unaccelerated baseline. By contrast, scaling up the batch size can solve this problem and deliver far greater, more stable speed-up ratios over the baseline.

## E Case Study

We provide the case study of our method in Table 6 and Table 7 and Table 8.

<span id="page-14-1"></span><sup>&</sup>lt;sup>4</sup>https://github.com/HKUNLP/Dream/blob/main/eval/eval\_dream\_gen.sh

<span id="page-15-0"></span>Table 6: Case study on LLaDA-8B-Instruct. Question: Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?

| Model                                                 | Generate Text (L=128, T=128, B=64)                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LLaDA-8B<br>Instruct<br>(low<br>confidence)           | 1. Calculate the distance Lily runs in the first 4 hours:<br>- Speed: 12 kilometers per hour<br>- Time: 4 hours<br>- Distance: 12 km/h * 4 h = 48 km<br>2. Calculate the distance Lily runs in the next 4 hours:<br>- Speed: 6 kilometers per hour<br>- Time: 4 hours<br>- Distance: 6 km/h * 4 h = 24 km<br>3. Add the distances together:<br>- Total distance: \boxed{72} km            |
| dKV-Cache<br>Decode,<br>Refresh = 4                   | 1. Calculate the distance Lily runs in the first 4 hours:<br>- Distance = Speed × Time = 12 km/h × 4 h = 48 km<br>2. Calculate the distance Lily runs in the next 4 hours:<br>- Distance = Speed × Time = 6 km/h × 4 h = 24 km<br>3. Add the distances from both periods to find the total distance Lily<br>runs in 8 hours:<br>- Total Distance = 48 km + 24 km = 72 km<br>\boxed{72 km} |
| dKV-Cache<br>Decode,<br>Refresh=16                    | 1. Calculate the distance Lily runs in the first 4 hours:<br>12 km/hour * 4 hours = 48 km<br>2. Calculate the distance Lily runs in the next 4 hours:<br>6 km/hour * 4 hours = 24 km<br>3. Add the distances from the first 4 hours and the next 4 hours to find<br>the total distance she runs:<br>48 km + 24 km = 72 km<br>Therefore, Lily can run \boxed{72} kilometers in 8 hours.    |
| dKV-Cache<br>Greedy,<br>Refresh = 2                   | To determine how many kilometers Lily can run in 8 hours, we need to<br>break down her running schedule and sum the distances.<br>First, calculate the distance she runs at 12 kilometers per hour for 4<br>hours:<br>12 ∗ 4 = 48<br>After 4 hours, she runs 6 kilometers per hour for the remaining 4 hours:<br>6 ∗ 4 = 24<br>Now, add the distances:<br>48 + 24 = 72<br>\boxed{72}      |
| dKV-Cache<br>Greedy,<br>Refresh<br>=<br>4<br>(random) | For the first 4 hours Lily runs at 12 kilometers per hour, so she runs 12 *<br>4 = 48 kilometers. 8 hours - 4 hours are left after the first 4 (8 - 4 = 4<br>hours).<br>For the next 4 hours, she runs at 6 kilometers per hour, so she runs 6 * 4<br>= 24 kilometers.<br>Adding the two distances together, Lily can run 48 + 24 = 72 kilometers<br>in 8 hours.<br>Answer: \boxed{72}    |

<span id="page-16-0"></span>Table 7: Case study on Dream-Instruct-7B. Request: Write a story that ends with "Finally, Joey and Rachel get married."

| Model                                | Generate Text (L=256, T=256)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dream<br>Instruct-7B<br>(Entropy)    | Once upon a time, there was a young man named Joey who lived in a small<br>town in the middle of nowhere. He was a kind and gentle soul who loved to sing<br>and play the guitar. One day, he met a beautiful girl named Rachel who was<br>also a talented singer and musician. They fell in love and decided to get married.<br>However, there was a problem. Joey's parents were against the marriage because<br>they didn't know Rachel well enough. They thought that she was not a good<br>match for their son. Rachel, on the other hand, was not worried about Joey's<br>parents' opinion. She knew that she loved Joey and that he was the one for her.<br>She decided to try and win over Joey's parents by showing them how much she<br>cared about him. She started by cooking for them and inviting them to her house<br>for dinner. She also took them on a trip to the mountains where they could enjoy<br>the scenery and the fresh air. After a while, Joey's parents started to see Rachel<br>in a different light. They realized that she was a good person and that she loved<br>Joey very much. They decided to give their blessing to the marriage. Finally,<br>Joey and Rachel got married and lived happily ever after.                                                              |
| dKV-Cache<br>Decode,<br>Refresh = 16 | Once upon a time, in a small town called New York, there lived a young man<br>named Joey and a young woman named Rachel. They were both very happy<br>with their lives, but they were also very different from each other. Joey was a<br>very outgoing person who loved to be around people, while Rachel was a very<br>introverted person who loved to be alone. One day, Joey and Rachel met at a<br>coffee shop and fell in love. They spent hours talking and laughing together, and<br>they realized that they were perfect for each other. They decided to be together<br>and build their lives together. As the years went by, Joey and Rachel grew closer<br>and closer. They shared their dreams and their fears, and they supported each<br>other through thick and thin. They were a perfect match, and they knew that<br>they were meant to be together. Finally, Joey and Rachel got married. They had<br>a beautiful wedding with all their friends and family in attendance. They were<br>so happy to be together, and they knew that they would always be together.                                                                                                                                                                                                                         |
| dKV-Cache<br>Prefill                 | Once upon a time, in a small town called New York, there lived a young man<br>named Joey. Joey was a kind and caring person who always tried to help others.<br>He had a beautiful girlfriend named Rachel, who was also kind and caring.<br>They were in love with each other and wanted to spend their lives together. One<br>day, Joey and Rachel decided to get married. They planned their wedding for a<br>long time, and everyone in the town was excited to attend. On the day of the<br>wedding, Joey and Rachel exchanged vows and promised to love each other<br>forever. After the wedding, Joey and Rachel moved into a beautiful house in the<br>countryside. They were happy and content with their lives, and they had two<br>children, a boy and a girl. The children grew up to be kind and caring people,<br>just like their parents. As the years went by, Joey and Rachel continued to love<br>each other deeply. They had many ups and downs, but they always found a way<br>to overcome their challenges and stay together. Finally, after many years of<br>marriage, Joey and Rachel decided to retire and live in a small cottage in the<br>countryside. They were happy and content with their lives, and they knew that<br>they had found the love of their lives in each other. |

<span id="page-17-0"></span>Table 8: Case study on Dream-Instruct-7B. Question: Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?

| Model                                | Generate Text (L=128, T=128)                                                                                                                                                                                                                                                                                        |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dream<br>Instruct-7B<br>(Entropy)    | In the first 4 hours, Lily runs 12 kilometers per hour, so she runs a total of 12 *<br>4 = 48 kilometers. In the next 4 hours, Lily runs 6 kilometers per hour, so she<br>runs a total of 6 * 4 = 24 kilometers. Therefore, in 8 hours, Lily can run a total<br>of 48 + 24 = 72 kilometers. #### 72                 |
| dKV-Cache<br>Decode,<br>Refresh = 4  | In the first 4 hours, Lily can run 12 kilometers per hour, so she can run a total of<br>12 * 4 = 48 kilometers. In the next 4 hours, Lily can run 6 kilometers per hour,<br>so she can run a total of 6 * 4 = 24 kilometers. Therefore, in 8 hours, Lily can<br>run a total of 48 + 24 = 72 kilometers. #### 72     |
| dKV-Cache<br>Decode,<br>Refresh = 16 | In the first 4 hours, Lily runs 12 kilometers per hour, so she runs a total of 12 *<br>4 = 48 kilometers. In the next 4 hours, Lily runs 6 kilometers per hour, so she<br>runs a total of 6 * 4 = 24 kilometers. Therefore, in 8 hours, Lily can run a total<br>of 48 + 24 = 72 kilometers. The answer is: 72       |
| dKV-Cache<br>Prefill,<br>No refresh  | In the first 4 hours, Lily runs 12 kilometers per hour, so she runs a total of<br>12 × 4 = 48 kilometers. In the next 4 hours, she runs 6 kilometers per hour, so<br>she runs a total of 6 × 4 = 24 kilometers. Therefore, in 8 hours, Lily can run a<br>total of 48 + 24 = \boxed{72} kilometers.The answer is: 72 |