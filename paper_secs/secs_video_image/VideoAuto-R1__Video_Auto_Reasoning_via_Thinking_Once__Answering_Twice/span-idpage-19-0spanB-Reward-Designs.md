# <span id="page-19-0"></span>B Reward Designs

To complement the reward description in the main paper, we provide the details below. Our overall reward is defined as a weighted sum of the task reward and the format reward.

Task Reward. We consider three task types for computing task rewards: QA, temporal grounding, and grounding QA.

• Question Answering. For math problems, we use math-verify to compare the prediction with the ground truth; otherwise we compare normalized strings (e.g., case-folded, whitespace stripped). This yields a binary reward

$$R_{\text{QA}}(o_i) \in \{0, 1\}.$$

• Temporal Grounding. Let the ground-truth segments be G = {[s<sup>j</sup> , e<sup>j</sup> ]}<sup>j</sup> and the predicted segments be <sup>G</sup><sup>b</sup> <sup>=</sup> {[sˆk, <sup>e</sup>ˆk]}<sup>k</sup> (either set may contain one or multiple segments). We compute the temporal IoU and take the best matching pair with the largest tIoU. If no valid segment can be parsed, we assign RTG(oi) = 0.

$$R_{\mathrm{TG}}(o_i) = \max_{[\hat{s}, \hat{e}] \in \widehat{\mathcal{G}}, [s, e] \in \mathcal{G}} \mathrm{tIoU}([\hat{s}, \hat{e}], [s, e]) \in [0, 1],$$

• Grounding QA. We parse the textual answer and the predicted segments from the model output, compute RQA(oi) and RTG(oi) as above, and sum them:

$$R_{\text{GQA}}(o_i) = R_{\text{QA}}(o_i) + R_{\text{TG}}(o_i) \in [0, 2].$$

Format Reward. In addition to task correctness, we use a binary format reward Rfmt(oi) ∈ {0, 1} enforced via strict regex checks. For VideoAuto-R1, we require exactly two \\boxed{...} answers, and in between one <think>...</think> block, with no additional text before, between, or after.

Analysis of the Dual-Answer Reward Design. In Section 4.2 of the main paper, we introduce the dual-answer reward design used during training. The key components of this design are the weight coefficients w<sup>1</sup> and w<sup>2</sup>

assigned to the initial and reviewed answers, respectively, as well as the fallback bonus weight α. Table [12](#page-20-2) summarizes the effects of different choices for these coefficients.

First, when w<sup>1</sup> = w2, the model assigns identical rewards to two distinct cases: (i) the first answer is correct but the second is wrong, and (ii) the first answer is wrong but the second is correct. However, our intention is to prioritize the correctness of the reviewed answer, since users who permit step-by-step reasoning with a sufficient compute budget expect the final answer to be reliable. Therefore, equal weighting fails to distinguish these two scenarios. By choosing w<sup>1</sup> <w<sup>2</sup> (e.g., 0.9: 1.1), the total reward becomes 0.9 for a "correct → wrong" pattern, but 1.1 for "wrong → correct", thereby encouraging the model to produce accurate reviewed answers during RL.

<span id="page-20-2"></span>Table 12 Effects of Dual-Answer Reward Coefficients.

| First<br>Answer | Second<br>Answer | w1<br>= 1,<br>w2<br>= 1,<br>α= 0 | w1<br>= 0.9,<br>w2<br>= 1.1,<br>α= 0 | w1<br>= 0.9,<br>w2<br>= 1.1,<br>α= 0.3 |
|-----------------|------------------|----------------------------------|--------------------------------------|----------------------------------------|
| ✗               | ✗                | 0                                | 0                                    | 0                                      |
| Let's analyze   | ✗                | 0                                | 0                                    | 0                                      |
| ✓               | ✗                | 1                                | 0.9                                  | 0.9                                    |
| ✗               | ✓                | 1                                | 1.1                                  | 1.1                                    |
| Let's analyze   | ✓                | 1                                | 1.1                                  | 1.4                                    |
| ✓               | ✓                | 2                                | 2                                    | 2                                      |

Second, even with w<sup>1</sup> < w2, the model still assigns the same reward when the first output is an incorrect guess or a fallback string "Let's analyze the problem step-by-step." The fallback string is not a wrong prediction; rather, it is an explicit and honest signal that the model identifies the task as difficult and intentionally defers reasoning to the next stage. Such behavior should be incentivized. By introducing the fallback bonus α, as shown in the last column of Table [12,](#page-20-2) the model is able to clearly differentiate between an incorrect guess and a fallback indicator.

<span id="page-20-0"></span>Finally, when both the initial and reviewed answers are correct, the model receives the highest possible reward, which aligns with our design goal.

