# F Visual decoding potential

Visual decoding task for a new dataset NSD [2] has also been evaluated for MindEye2 [26] as the baseline and BrainMoE. We pre-trained two MindEye2s as the specific experts for long-term (novel trials) and short-term memory (easy/hard trials), respectively. Since visual decoding is a generative task, output contains much higher dimensions (256×1664 vs. class number 2 to 7) than downstream tasks focused in the main text. Therefore, we skipped our cognition adapter by weighted summing the diffusion prior of two experts with the BrainMoE routing probabilities. Both baseline and BrainMoE are pretrained with subjects 2-7 and finetuned with subject 1 on the entire 40 sessions. The final train and test losses, cosine similarity, and Mean Squared Error (MSE) during finetuning are listed in Table 9. Given the evidence that the performance of BrainMoE is better than the single expert MindEye2, there is potential for BrainMoE to expand to visual decoding.

<span id="page-21-2"></span>Table 9: Visual decoding performance.

|            | MindEye2 | BrainMoE |
|------------|----------|----------|
| Train loss | 9.639    | 7.994    |
| Test loss  | 11.142   | 9.405    |
| Cos. Sim.  | 0.778    | 0.840    |
| MSE        | 0.301    | 0.261    |

