# Lessons on Visual Perception.

- *(1) Qwen2.5VL-32B* Direct input augmentation with visual annotations achieves the highest overall performance, demonstrating the clear advantage of providing explicit visual understanding. Prompting the model to generate its own visual interpretation is moderately effective but generally less robust than supplying pre-processed annotations. Notably, on complex datasets such as OlympiaBench, structured prompting can be detrimental: when the model's intrinsic perceptual ability is insufficient, requiring it to articulate its perception often produces hallucinated or inaccurate observations, introducing bias and degrading reasoning performance.
- *(2) Qwen2.5VL-7B* Similar to the 32B model, explicit visual annotation improves performance. However, the 7B model shows greater sensitivity to perception prompting, with structured prompts consistently reducing performance, particularly on more challenging tasks. This suggests that smaller models are more prone to self-induced perceptual errors when required to generate their own interpretations.

### Lessons on Textual Perception.

*(1)Qwen2.5-VL-32B.* Under both explicit and implicit visual perception settings, incorporating additional textual perception yields only marginal performance gains. Moreover, the improvements are inconsistent across different benchmarks.

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

Figure 1: Overview of the proposed two-stage training pipeline VTPerception-R1. Stage 1 performs supervised fine-tuning with perception-grounded annotations, where explicit visual and textual notes are integrated into the reasoning process to strengthen multimodal perception. Stage 2 applies reinforcement learning with perception-aware rewards, further refining the Description → Think → Answer reasoning pipeline for improved consistency and interpretability.

*(2)Qwen2.5-VL-7B.* Similar to the 32B variant, introducing textual perception on top of either explicit or implicit visual prompts results in limited performance improvement. However, the gains are notably more stable across tasks.

Overall. Perception prompting's impact strongly depends on model scale: larger models leverage perceptual signals more effectively, while smaller models often struggle without explicit guidance. Supplying robust perceptual information boosts performance, confirming that strong perception is critical for advanced reasoning. Visual perception remains a major frontier for improvement across models, whereas textual perception offers particularly high gains for smaller-scale systems.

