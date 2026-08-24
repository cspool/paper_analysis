# Limitations

Although LightThinker has shown remarkable advancements in memory optimization and inference speed enhancement, certain limitations warrant careful consideration:

- 1. The number of cache tokens is fixed during training and must remain consistent during inference. The generalization capability of these token representations is uncertain. For instance, whether representations trained with 3 tokens can extrapolate to scenarios requiring more tokens during inference.
- 2. The design of the segmentation function is relatively simplistic, relying on rule-based methods. Future work could investigate more advanced segmentation strategies.
- 3. The performance of LightThinker on tasks such as novel generation, code generation, and multi-turn dialogue remains unassessed.

