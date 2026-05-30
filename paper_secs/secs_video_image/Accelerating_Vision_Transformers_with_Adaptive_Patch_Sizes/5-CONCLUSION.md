# 5 CONCLUSION

We presented Adaptive Patch Transformer (APT), a method to accelerate ViTs that uses larger patches in simpler areas and smaller patches in more complex ones. It significantly improves training and inference speeds, especially for larger models and higher resolutions. APT can be applied to any pretrained ViT backbone and converges in 1 epoch or less, enabling users to quickly train their models to be faster on a wide range of vision tasks. Our results suggest that APT will benefit the broader vision community by reducing the compute budget required to train state-of-the-art models.

Limitations. Although APT provides significant speedups, it still relies on a hand-crafted heuristic to determine patch sizes, which may not always align with downstream users' preferences and could likely be improved. Additionally, APT relies on an empirically-tuned threshold hyperparameter, which can add friction to adoption on downstream tasks. Finally, while APT works for image understanding tasks, it currently does not support image generation, which operates with extremely high-resolution images and large models, making it an ideal application for our work. Future work will be required to overcome these limitations, and we hope that APT can inspire further research on efficient ViTs.

## ACKNOWLEDGMENTS

RC is supported by the NSF Graduate Research Fellowship (GRFP). JK is supported by an IITP grant from the Korean government (MSIT) under the AI Excellence Global Innovative Leader Education Program (RS-2022-00143911).

