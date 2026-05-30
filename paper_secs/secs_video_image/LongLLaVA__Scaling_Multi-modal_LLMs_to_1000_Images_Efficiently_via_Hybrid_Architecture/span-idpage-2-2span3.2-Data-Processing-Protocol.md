# <span id="page-2-2"></span>3.2 Data Processing Protocol

To ensure the model can effectively distinguish temporal from spatial dependencies in multi-image inputs and perform robustly across diverse tasks, we have meticulously designed and differentiated special tokens for various scenarios. As illustrated in Figure [2,](#page-3-2) these tokens are tailored to represent the complex relationships between images in varying contexts, thereby enhancing the model's adaptability to a wide range of tasks.

<span id="page-2-4"></span><span id="page-2-3"></span><sup>1</sup> openai/clip-vit-base-patch32

<sup>2</sup>We chose Expert-0 due to minimal performance differences, detailed in Appendix [C.](#page-14-1)

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 1: **Architecture of LongLLaVA.** The LongLLaVA model is capable of (1) accommodating a variety of multimodal inputs and efficiently processing image tokens via 2D token compression; (2) uniformly managing the preprocessed inputs within its hybrid LLM architecture.

