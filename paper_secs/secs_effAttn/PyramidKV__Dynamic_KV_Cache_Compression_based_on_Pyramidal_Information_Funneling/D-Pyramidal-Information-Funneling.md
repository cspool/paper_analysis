# **D Pyramidal Information Funneling**

[Figure 5](#page-12-0) and [Figure 6](#page-13-0) shows the attention patterns of one QA example over six different layers (i.e., 0, 6, 12, 18, 24, and 30) for Mistral-7B-Instruct model and Mixtral-8x7B-Instruct Mixture-of-Experts model. [Figure 5](#page-12-0) and [Figure 6](#page-13-0) demonstrate that the Pyramidal Information Funneling phenomenon is also evident in both the Mistral model and Mixtral model . The results reveal that, akin to Llama-like models, Mistral exhibit a progressively narrowing attention focus across layers. This supports the universality of the Pyramidal Information Funneling phenomenon across diverse model families. We hope this addresses your concern and underscores the generalizability of our findings.

Our analysis uniquely examines attention metrics across all transformer layers, from 0 to 30, leading to the discovery of a key phenomenon we term Pyramidal Information Funneling.

[Lee et al.](#page-10-14) [\(2024\)](#page-10-14) conducted a limited investigation into attention patterns, focusing only on the lower layer (layer 0) and a single upper layer (layer 18). While [Lee et al.](#page-10-14) [\(2024\)](#page-10-14) noted that attention becomes more skewed in upper layers, it did not provide a fine-grained observation of attention patterns across all layers. In contrast, our study reveals several novel findings:

- **Localized Attention**: We observe that attention progressively narrows its focus, targeting specific components within the input sequence.
- **Massive Attention Mechanism**: In the upper layers, attention heavily concentrates on a small set of critical tokens. Notably, these tokens are not limited to the leading positions, as observed in [Lee et al.](#page-10-14) [\(2024\)](#page-10-14), but also appear at regular intervals across the sequence. The discrepancy arises from differences in input settings, with [Lee](#page-10-14) [et al.](#page-10-14) [\(2024\)](#page-10-14) identifying massive attention only at the initial tokens.

These insights motivated us to propose a token-selection method based on the highest attention scores in the upper layers, rather than solely relying on tokens from earlier positions.

To the best of our knowledge, [Chen et al.](#page-9-11) [\(2024b\)](#page-9-11) has not analyzed attention patterns across transformer layers.

Therefore, although [Lee et al.](#page-10-14) [\(2024\)](#page-10-14) and [Chen et al.](#page-9-11) [\(2024b\)](#page-9-11) are considered contemporaneous with our work, making a comparison unnecessary, the perspective of our observation is considered novel compared with [Lee et al.](#page-10-14) [\(2024\)](#page-10-14) and [Chen et al.](#page-9-11) [\(2024b\)](#page-9-11). Moreover, although [Lee et al.](#page-10-14) [\(2024\)](#page-10-14) also observed attention patterns, the method we proposed based

![](_page_15_Picture_1.jpeg)

Figure 7: Illustration of PyramidKV. At the lower level of the transformer, the PyramidKV selects more keys and values based on the exhibited average attention pattern. Fewer keys and values at the higher level are selected based on the massive activation pattern, where we observe that attention scores are concentrated over local regions.

<span id="page-15-0"></span>on our observations is significantly different from [Lee et al.](#page-10-14) [\(2024\)](#page-10-14), further highlighting the novelty of our work.

