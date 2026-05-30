# 7 Conclusion

We present LMFusion, a framework designed to equip LLMs with multimodal generative capabilities. By using Llama-3 for text generation and integrating parallel transformer modules for image diffusion, LMFusion efficiently reuses compute invested in pretrained LLMs.

LMFusion's modular design enables independent developments of language and vision modules, de-risking the complexities associated with a large-scale, joint-modality pretraining. While LMFusion is currently built upon text-only LLMs, it can benefit further from existing visual understanding LLMs [\(Liu et al.,](#page-12-15) [2023;](#page-12-15) [Dai](#page-11-17) [et al.,](#page-11-17) [2023;](#page-11-17) [Liu et al.,](#page-12-16) [2024b;](#page-12-16) [Zhu et al.,](#page-14-8) [2024\)](#page-14-8), inheriting the strong multimodal understanding ability while enabling generating interleaved text and visual content.

<span id="page-10-0"></span><sup>5</sup><huggingface.co/datasets/xai-org/RealworldQA>

<span id="page-10-1"></span><sup>6</sup>Concurrent to our work, [Liu et al.](#page-12-4) [\(2024a\)](#page-12-4) tackles multimodal generation via a joint attention mechanism between a DiT structure [\(Peebles and Xie,](#page-12-17) [2023\)](#page-12-17) for images and a frozen Llama-3 [\(Dubey et al.,](#page-11-2) [2024\)](#page-11-2) for texts.

