# A ADDITIONAL DISCUSSIONS

#### A.1 EXPERT ARCHITECTURE

Experts in MoE models are typically identical to the standard Feed-Forward Networks (FFNs) used in dense models. Recently, efforts have been made to improve the expert architecture. Deepseek-MoE [\(Dai et al.,](#page-10-3) [2024\)](#page-10-3) and XMoE [\(Yang et al.,](#page-13-5) [2024\)](#page-13-5) split the FFN in the dense model into smaller FFNs, reducing the size of each expert while increasing the number of activated experts. PEER [\(He,](#page-11-9) [2024\)](#page-11-9) and MH-MoE [\(Wu et al.,](#page-13-13) [2024\)](#page-13-13) go further by not only reducing the size of experts but also splitting input tokens into smaller units. Although these methods have made some progress, the structure of experts in existing MoE models remains largely based on FFNs, with little exploration of non-FFN or non-parametric experts. To the best of our knowledge, we are the first to propose zero-computation experts for the heterogeneous MoE architecture.

### A.2 LIMITATIONS AND FUTURE WORK

In this section, we delineate the limitations of our work and outline avenues for future research.

Heterogeneous MoE++ Between Different Layers. MoE++ implements heterogeneous experts within a single MoE layer. Additionally, as shown in Appendix [D,](#page-17-0) we observe that expert assignment patterns vary more significantly in the shallow and final layers across different tasks, compared to the middle layers. This suggests that the model adapts to tasks primarily through these layers. Future work could explore designing heterogeneous MoE++ configurations across different layers to further enhance the model's adaptability to a wide range of tasks.

Combining MoE++ with Other Modules. The current MoE++ method serves as a replacement for the FFN layer in Transformers. Future work could explore integrating other modules, such as combining the attention layer with our MoE++ method.

The Vulnerabilities of Large Language Models. The focus of our work is to build advanced and efficient mixture-of-experts Large Language Models (LLMs), and as a consequence, also inherit the vulnerabilities common to LLMs.

- Hallucination. Hallucinations in LLMs remain a significant unresolved challenge. These illusory responses can lead to unsupported claims during open-ended conversations, and addressing this issue could greatly accelerate progress in the field. For a deeper analysis of common weaknesses in large LLMs, please refer to [Brown et al.](#page-10-0) [\(2020\)](#page-10-0); [Rae et al.](#page-12-15) [\(2021\)](#page-12-15).
- Long sequence processing. Transformer-based language models often struggle with generalization when faced with test sequences that are significantly longer than those seen during training. This limitation is especially pronounced in multi-turn conversations, where the model may lose track of the previous context, leading to incorrect responses.
- Prompt sensitivity. In-context learning has shown troubling sensitivity to various aspects of demonstrations, such as prompt formats [\(Zhao et al.,](#page-13-14) [2021\)](#page-13-14). Notably, variations in prompt formats can lead to completely contradictory outputs. Addressing this issue could significantly accelerate progress in the field.

More Modalities. Language represents just one facet of communication. Visual and audio information serves to augment and enhance our comprehension of the world [\(Jin et al.,](#page-11-5) [2024;](#page-11-5) [2023;](#page-11-16) [2022\)](#page-11-17). Future work can explore alternative modalities, such as visual and audio inputs. The incorporation of multiple modalities holds the promise of broadening the spectrum of tasks that the model can address, and it has the potential to enhance their performance by leveraging synergies among these various modalities [\(Jin et al.,](#page-11-5) [2024\)](#page-11-5).

More Parameters. Due to computational constraints, the maximum number of MoE++ model parameters in our experiments is limited to 7B. However, our MoE++ method is highly generalizable and can be scaled to larger models in future research.

