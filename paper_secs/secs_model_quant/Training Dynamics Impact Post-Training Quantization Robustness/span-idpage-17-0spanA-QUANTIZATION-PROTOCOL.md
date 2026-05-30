# <span id="page-17-0"></span>A QUANTIZATION PROTOCOL

**Alternative quantization methods.** Our results are centered around GPTQ Frantar et al. (2023) a popular and accessible quantization method that works off-the-shelf for new models with minimal engineering overhead. To assess whether the phenomena we observe are specific to GPTQ or reflect broader trends in PTQ, we replicate Figure 4 with LLM.int8() Dettmers et al. (2022) and AWQ Lin et al. (2024). As shown in Figure 10, we observe a consistent association between learning rate driven training dynamics and quantization error.

<span id="page-17-3"></span>![](_page_17_Figure_3.jpeg)

**Figure 10: Quantization error on different 4-bit quantization backends.** We replicate results from Section 4.1, training a 160M-parameter transformer with different quantization backends, and recover similar trends in quantization error during both the constant and cooldown phases of the learning rate schedule.

**Quantization details.** For each model, we quantize the linear layers following the default settings of GPTQModel (ModelCloud.ai & qubitium@modelcloud.ai, 2024) and HuggingFace's internal quantization backend. For GPTQ, we follow common practice (Wolf et al., 2020) and use C4 (Raffel et al., 2023) as the calibration dataset, with a group size of 128. For AWQ (Lin et al., 2024), we use Kwon et al. (2023).Finally, for LLM.int8() Dettmers et al. (2022) we follow HuggingFace Wolf et al. (2020) implementation.

