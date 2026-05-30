# <span id="page-15-1"></span>K Visualization of Instruction-Relevant Focus Across Middle Layers

<span id="page-15-6"></span>![](_page_15_Picture_11.jpeg)

Figure 8: The Most Instruction-Relevant Region Highlighted in Red Boxes.

Given the user instruction "*What kind of apple is this?*" and the image in [Fig. 8,](#page-15-6) we observe that the last token in the middle layers consistently focuses on the most instruction-relevant region (see [Tab. 16\)](#page-16-1).

<span id="page-16-1"></span>

| Layers | Top 10 Visual Tokens Indices                     |
|--------|--------------------------------------------------|
| 22     | 107, 108, 129, 130, 60, 222, 155, 255, 512, 162  |
| 21     | 107, 108, 129, 130, 60, 222, 155, 255, 512, 162  |
| 20     | 107, 108, 60, 162, 161, 222, 163, 61, 399, 255   |
| 19     | 108, 107, 60, 222, 255, 387, 399, 61, 207, 299   |
| 18     | 108, 222, 107, 207, 60, 502, 155, 88, 355, 399   |
| 17     | 107, 222, 108, 155, 60, 512, 130, 156, 255, 129  |
| 16     | 107, 108, 222, 155, 60, 156, 131, 355, 109, 340  |
| 15     | 107, 108, 222, 60, 61, 255, 88, 163, 399, 155    |
| 14     | 222, 107, 355, 108, 340, 159, 574, 255, 398, 131 |
| 13     | 222, 107, 355, 108, 340, 398, 574, 255, 60, 155  |
| 12     | 222, 355, 340, 398, 270, 155, 574, 107, 272, 207 |
| 11     | 222, 355, 340, 574, 575, 398, 108, 107, 155, 156 |
| 10     | 222, 575, 355, 574, 340, 398, 207, 571, 272, 108 |

Table 16: Top 10 most attended vision tokens from the last input token at each layer. Green indicates the most critical visual tokens, while red marks the visual attention sink tokens.

## L Layer-wise Cross-Attention Masking on MobileVLM 3B

Compared to LLaVA-v1.5 7B, MobileVLM v2 3B has a broader range of shallow layers and fewer deep layers. This suggests that smaller models may require more computations on task recognition.

![](_page_16_Figure_5.jpeg)

Figure 9: Impact of masking layer ranges from shallowto-deep and deep-to-shallow, showing a clear reduction in cross-modal fusion in both shallow and deep layers.

## <span id="page-16-0"></span>M FLOPs Analysis on LLaVA-1.5 7B

Our proposed method greatly reduces visionrelated self-attention, cross-attention and FFN, leading to an overall FLOPs reduction of > 60%. Here is a detailed analysis:

The total computation in MLLMs primarily consists of two components: attention computation and

feed-forward network (FFN) computation. Among these, attention computation scales quadratically with sequence length, making it the primary computational bottleneck—especially in models like Qwen2-VL, which can generate up to 12,000 visual tokens. For instance, in LLaVA-1.5 7B, the FLOPs for attention computation can be expressed as 2n <sup>2</sup>d. The reduction ratio for visual attention computation is given by:

$$R = 1 - \frac{L'2 * 2(n'_v)^2 d + L'(n'_v n_t) d}{32 * (2(n_v^2 d + n_v n_t))}$$

where the L ′ the number of cross-modal interaction layers, n ′ v represents the number of retained visual tokens. If the input sequence consists of 650 tokens (576 visual tokens and 74 text tokens), our approach eliminates attention computation in shallow and deep layers, retaining only a few critical tokens for cross-modal fusion. This results in a 99% reduction at maximum in attention computation.

FLOPs Calculation. In LLaMA 2 7B [\(Touvron](#page-10-21) [et al.,](#page-10-21) [2023\)](#page-10-21), the primary flops include FFN and self-attention. The flops for FFN is 3ndm, where n is the number of input tokens, d is the hidden state size, and m is the intermediate size of the FFN. Hence, the FLOPs overall calculation for visual tokens follows:

$$\sum_{i=0}^{L_{\text{middle}}} \left( 4n'_v d^2 + 2n'^2_v d + 3n'_v dm \right) + \sum_{i=0}^{L_{\text{shallow}}} \left( 4n_v d^2 + 3n_v dm \right)$$

This optimization leads to an overall visual FLOPs reduction of 62.8% under the given setting (576 visual tokens and 74 text tokens), significantly enhancing efficiency while maintaining performance. Given that the efficiency gain scales with longer textual or visual inputs, our pruning framework offers much greater benefits for longer text instructions or when multiple images are provided.

Additionally, following our actionable guidelines for optimizing MLLMs, the visual computation overhead within shallow layers in FFN should be able to be further reduced through training.

#### N Failure Case Analysis

In this section, we present an analysis on failure cases in GQA, where our pruned model produced

- 1,125 mismatched answers compared to the vanilla LLaVA-v1.5 7B over 12,000 samples.
  - 234 answers were correct in our model but incorrect in the vanilla model.
  - 325 answers were incorrect in our model but correct in the vanilla model.

Upon closer inspection, we found that misclassifications were often related to variations in word choice rather than fundamental misunderstandings. Below are some examples:

- N.1 "Which kind of vehicle is in front of the flag?\nAnswer the question using a single word or phrase."
- Ground Truth Answer: "van"
- Vanilla Model: "truck"
- Ours: "van"
- N.2 "What is sitting in front of the table that looks yellow and black?\nAnswer the question using a single word or phrase."
- Ground Truth Answer: "luggage"
- Vanilla Model: "backpack"
- Ours: "suitcase"
- N.3 "What is in front of the poster?\nAnswer the question using a single word or phrase."
- Ground Truth Answer: "monitor"
- Vanilla Model: "monitor"
- Ours: "computer"