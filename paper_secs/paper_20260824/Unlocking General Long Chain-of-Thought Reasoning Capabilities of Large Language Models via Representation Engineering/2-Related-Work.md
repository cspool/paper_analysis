# 2 Related Work

Our work is related to the following two research directions.

Large Language Model Reasoning. Recently, improving the reasoning capabilities of LLMs has become a critical challenge. Prior approaches, such as test-time search [\(Zhang et al.,](#page-11-6) [2024;](#page-11-6) [Tang et al.,](#page-10-5) [2024b;](#page-10-5) [Guan et al.,](#page-9-5) [2025;](#page-9-5) [Cheng et al.,](#page-9-6) [2025\)](#page-9-6), distillation [\(Yu et al.,](#page-11-7) [2024;](#page-11-7) [Min et al.,](#page-10-6) [2024\)](#page-10-6), and reinforcement learning [\(Guo et al.,](#page-9-0) [2025\)](#page-9-0), enable LLMs to engage in deliberate thinking [\(Tang et al.,](#page-10-7) [2024a;](#page-10-7) [Cheng et al.,](#page-9-1) [2024a;](#page-9-1) [Wang et al.,](#page-11-8) [2025a\)](#page-11-8). Despite their remarkable success, the underlying mechanisms of LLM reasoning remain unclear. Some studies [\(Christ et al.,](#page-9-7) [2024;](#page-9-7) [Rai and Yao,](#page-10-8) [2024\)](#page-10-8) analyze this by localizing specific neurons, but they only focus on isolated neuron connections, neglecting the cooperative activity of multiple neurons. Other work [\(Hu et al.,](#page-10-9) [2024;](#page-10-9) [Højer et al.,](#page-9-8) [2025\)](#page-9-8) addresses this via representation engineering to better control neuron collaboration. However, these studies are often limited to short-form CoT, struggling

to fully unlock the reasoning potential of LLMs. In this paper, we focus on exploring the mechanism of long CoT reasoning through representation engineering.

Representation Engineering. Representation engineering (Zou et al., 2023) treats internal representations as the fundamental unit, focusing on analyzing and manipulating them within neural networks. As a well-established technique, it has been applied in various areas such as personality modeling (Cao et al., 2024), instruction following (Stolfo et al., 2025), hallucination alleviation (Li et al., 2023a; Arditi et al., 2024; Li et al., 2024), and safety improvement (Liu et al., 2024). While prior work focuses on simple concepts like sentiment (Hollinsworth et al., 2024) and style (von Rütte et al., 2024; Scalena et al., 2024), our work aims to address the more complex challenge: understanding and unlocking general long CoT reasoning capabilities of LLMs.

## 3 Empirical Analysis

In this section, we first introduce the background of representation engineering and then use it to conduct an empirical analysis of long CoT reasoning.

### <span id="page-2-0"></span>3.1 Background: Representation Engineering

The Hopfieldian view (Hopfield, 1982) explains cognition and behavior as emerging from transformations or movements within neural populations in response to external stimuli. Building upon this perspective, representation engineering (Zou et al., 2023) is proposed, which is a widely used approach for the mechanism interpretability of LLMs. It treats representations as the fundamental unit of various mechanisms in LLMs for analysis. This approach primarily encompasses two components: representation extraction and control. We will detail them in the following part.

**Representation Extraction.** It focuses on identifying high-level concepts or functions encoded in LLMs. For a typical Transformer (Vaswani et al., 2017) model, the outputs of multi-head attention (MHA), multi-layer perception (MLP), and hidden states can all be considered as representations, with each connected through the residual stream. At a given layer l and token position t, the hidden state  $h_l^t$  is computed recursively as follows:

$$h_l^t = h_{l-1}^t + a_l^t + m_l^t, (1)$$

where  $a_l^t$  and  $m_l^t$  represent the outputs from MHA and MLP, respectively. Here, we follow Zou et al. (2023) to extract representations from the hidden states at the final token position due to the sequential nature of language modeling.

Representation Control. It aims to steer model behaviors with extracted representations. This process typically first establishes a representation controller to modulate extracted representations. Then, the controller will inject the representations of target behaviors into the representations of LLMs. Here, we follow Hendel et al. (2023) to utilize a linear module as the representation controller and select a specific layer for representation injection. Such a method can achieve fine-grained control of model behaviors while preserving efficiency.

### <span id="page-2-1"></span>3.2 Analysis of Long CoT Representations

In this part, we first describe how to extract long CoT representations and then conduct an empirical analysis about them.

Extraction of Representations. To extract representations, first, we prompt an LLM to collect its vanilla CoTs  $s_i$  and long CoTs  $l_i$  for a set of questions  $x_i \in \mathcal{X}$ . Then, we concatenate each problem with the corresponding CoT and input this into the LLM for encoding. As stated in Section 3.1, the hidden states of the layer L at the final token position are extracted as the representations, which can be represented as follows:

$$R_L(s_i) = h_L^{-1}(x_i; s_i) \ R_L(l_i) = h_L^{-1}(x_i; l_i),$$
 (2)

where  $h_L^{-1}(s)$  denotes the hidden states of the string s at the last token position and layer L, and; denotes string concatenation. After performing the above operation, we can obtain a set of representations for vanilla and long CoTs.

Analysis of General Representations. To analyze the characteristics of vanilla and long CoTs, we visualize their representations to compare their distributions. Specifically, we employ a dimensionality reduction approach (*i.e.*, t-SNE (van der Maaten and Hinton, 2008)) to map representations obtained from the above part onto a 2D plane. As illustrated in Figure 2a (more figures in Appendix C), the representations of various long CoTs are concentrated in a specific area of the whole space. In addition, their distribution areas are clearly distinct from those of vanilla CoTs. Taken together, the two pieces of evidence suggest that LLMs do encode

long CoT reasoning as a separate general capability in their parameter spaces. Moreover, we find that the separation between these two types of CoTs is the most pronounced in the middle layers of the model, while less clear in the early and final layers. This phenomenon may be attributed to the fact that middle layers integrate information from early layers and are more informative (Skean et al., 2025), playing a critical role in capturing high-level concepts (*e.g.*, CoT reasoning) (See Appendix A).

#### Analysis of Domain-Specific Representations. In

this part, we further examine the characteristics of vanilla and long CoTs in specific domains. Specifically, we collect representations in mathematical and other domains (i.e., physics, chemistry, and biology) and visualize them following the previous part. As shown in Figure 2b (more figures in Appendix D), different domains share similar contrastive representations between long and vanilla CoTs, which further demonstrates the transferability of long CoT reasoning. In addition, the representations of mathematical domains are relatively concentrated, while those of other domains (e.g., physics) are more dispersed. This may be due to the fact that mathematical problems focus on logical reasoning patterns, while problems in other domains also require domain-specific information. That is, domain-specific CoT data plays an important role for the elicitation of long CoT reasoning within these domains (Dong et al., 2025b).

