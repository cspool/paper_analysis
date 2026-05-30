# <span id="page-33-0"></span>*D.3.1. Instructional Formatting for Cross-Modal Decision Making*

To facilitate this decision-making process, we structure the input as a comprehensive instruction that forces the model to weigh different sources of evidence. The model is presented with all modalities and explicitly prompted to declare its generation strategy.

```
Task: Perform reference-augmented correction for a given speech input.
Objective: Evaluate the quality of an internally generated hypothesis against external candidates.
First, select a generation strategy by producing a control token. Then, generate the final,
corrected text.
- If the internal hypothesis is deemed superior and well-aligned with the audio, select <accept_internal>.
- If the external candidates provide necessary corrections, select <integrate_reference>.
–-
Acoustic Evidence: [AUDIO]
External Candidate Transcriptions:
1. {h_1}
2. {h_2}
3. {h_3}
4. {h_4}
5. {h_5}
Internal Hypothesis:
{T_internal}
–-
Output:
```

The model is then trained to generate the complete target string, beginning with either <accept\_internal> or <integrate\_reference>, followed by the corrected and finalized text. We expand the model's vocabulary with these two special tokens to serve as explicit control signals.

## <span id="page-33-1"></span>*D.3.2. Supervision for Decision-Aware Fine-Tuning*

Supervision for this decision-aware fine-tuning is derived by comparing the internal hypothesis (internal) against the ground truth (gt) and the external references (ℋ). The decision label is determined as follows:

- **<accept\_internal>:** This label is assigned when the word error rate (WER) of internal is below a predefined threshold or when the external references in ℋ offer no improvement or introduce hallucinations. This teaches the model to trust its own cross-modal alignment between audio and text when its confidence is high.
- **<integrate\_reference>:** This label is assigned when internal contains correctable errors and at least one hypothesis in ℋ provides information that reduces the WER relative to gt. This trains the model to identify valuable external information and integrate it, effectively re-aligning its understanding based on supplementary textual evidence.

The final training target is the concatenation of the assigned decision token and the ground-truth transcript gt.

