# 6 CONCLUSION AND LIMITATIONS

We conducted a comprehensive study on different prompt compression methods for LLMs across various tasks. Our results demonstrated that (Long)LLMLingua and LLMLingua-2 generally give the best performance, particularly at higher compression ratios. All methods appeared to increase hallucinations, primarily due to information loss. Additionally, current methods showed varied effectiveness in multimodal tasks, suggesting the need for further optimization. Finally, we analyzed the words that can be omitted during compression. Our study provided a broader understanding of prompt compression, assisting future research in prompt engineering strategies.

Limitations. In this empirical study, we focused on the prompt compression techniques only, conducting experiments with three (M)LLMs: GPT-3.5-turbo, GPT-4o-mini, and Claude-3-Haiku. In terms of the compression methods for open-source models, there are approaches on modifying internal states or KV cache information for compressing or trimming [\(Liu et al., 2023b;](#page-11-7) [Zhang et al.,](#page-13-5) [2023;](#page-13-5) [Xiao et al., 2024;](#page-13-6) [Ge et al., 2024\)](#page-10-6). We leave the further study to our future work.

### ACKNOWLEDGMENTS

This research is supported by SMP-IDATA Open Youth Fund, Guangzhou-HKUST(GZ) Joint Funding Program (Grant No.2023A03J0008), the Guangzhou Municipal Science and Technology Project (No. 2025A04J4070), and Education Bureau of Guangzhou Municipality.

