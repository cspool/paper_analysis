# <span id="page-13-0"></span>A. Additional Discussions

### A.1. Why is MoH Superior to Vanilla Multi-Head Attention?

We demonstrate that MoH is superior to vanilla multi-head attention from both theoretical and experimental perspectives.

Specifically, MoH not only improves efficiency and model performance but also helps different attention heads to specialize better compared to multi-head attention.

From the theoretical perspective, in standard multi-head attention, all heads use the same data, which can cause them to learn similar features. Many studies have pointed out that there are redundant heads in multi-head attention. Given a minibatch of data D, the gradient of each attention head in multi-head attention can be written as Ex∈D[ ∂L(x) ∂h<sup>i</sup> ].

In contrast, in MoH, routed heads are trained only on smaller subsets of data specifically assigned to them. In MoH's routing mechanism, the data is divided into h − h<sup>s</sup> subsets {D1, D2, ..., Dh−h<sup>s</sup> }, with each subset corresponding to a routed head. Besides, the routing score for each attention head acts as an adaptive adjustment to the learning rate, enabling the attention heads in MoH to specialize more effectively. Given a minibatch of data D and the router G(∗), the gradient of each routed head in MoH can be written as Ex∈D<sup>i</sup> [G(x)<sup>i</sup> ∂L(x) ∂h<sup>i</sup> ]. The gradient of each shared head in MoH can be written as Ex∈D[G(x)<sup>i</sup> ∂L(x) ∂h<sup>i</sup> ]. As shown in Tab. [A,](#page-3-0) the routing mechanism and adaptive weights in MoH enable attention heads to specialize more effectively compared to standard multi-head attention.

Table A. Comparisons between the multi-head attention and our proposed mixture-of-head attention.

| Methods              | #Head Type  | #Data     | #Weight (learning rate) | #Gradient                            |
|----------------------|-------------|-----------|-------------------------|--------------------------------------|
| Multi-Head Attention | -           | D         | 1                       | Ex∈D[<br>∂L(x)<br>]<br>∂hi           |
| MoH                  | routed head | Di<br>∈ D | G(x)i                   | Ex∈Di<br>∂L(x)<br>[G(x)i<br>]<br>∂hi |
| MoH                  | shared head | D         | G(x)i                   | Ex∈D[G(x)i<br>∂L(x)<br>]<br>∂hi      |

From the experimental perspective, we calculated the similarity of attention patterns and output features of different attention heads (include routed heads and shared heads). As shown in Tab. [B,](#page-4-0) the similarity of attention patterns and output features among attention heads in MoH is lower than in standard multi-head attention, indicating reduced redundancy and greater differentiation among the attention heads in MoH.

Table B. The similarity of attention patterns and output features among attention heads. Given a pair of attention score matrices A and A ′ , we calculate the similarity of attention patterns as 1 − 1 2 E[||A − A ′ ||1]. Since attention scores form a probability distribution for each query, the similarity is always between 0 to 1.

| Methods                     |                  | Similarity of Attention Patterns | Cosine Similarity of Output Features |                  |  |
|-----------------------------|------------------|----------------------------------|--------------------------------------|------------------|--|
|                             | ViT              | LLM                              | ViT                                  | LLM              |  |
| Multi-Head Attention<br>MoH | 0.5159<br>0.3978 | 0.4795<br>0.4333                 | 0.0411<br>0.0165                     | 0.2550<br>0.2042 |  |

#### A.2. Limitations and Future Work

In this section, we delineate the limitations of our work and outline avenues for future research.

Heterogeneous Attention Heads. We find that different attention heads operate in parallel within the attention mechanism, suggesting that different heads can have varying hidden sizes. Future work could explore the use of heterogeneous attention heads based on our MoH framework.

Lower Activation Rate. Currently, MoH outperforms multi-head attention by utilizing only 50%∼90% of the attention

heads. However, this is still a relatively high proportion. Future work could aim to further optimize MoH, reducing head activation to less than 50%.

Multimodal Inputs. Effectively processing information from multiple modalities in the attention mechanism remains an open question. Recent work [\(Wan et al.,](#page-12-23) [2024\)](#page-12-23) has shown that visual and textual tokens exhibit distinct attention patterns in multi-head attention. Future work could explore the attention patterns of MoH with different modal inputs, for example within multimodal large language models [\(Jin et al.,](#page-10-20) [2024b;](#page-10-20) [Lin et al.,](#page-10-21) [2023;](#page-10-21) [2024;](#page-10-22) [Liu et al.,](#page-11-25) [2024;](#page-11-25) [Jin et al.,](#page-10-23) [2023;](#page-10-23) [2024a\)](#page-10-24).

More Downstream Tasks. We evaluate our proposed MoH across various popular model frameworks, including ViT for image classification, DiT for class-conditional image generation, and LLMs for language tasks. Future work can explore the application of MoH in more downstream tasks, such as audio tasks and multimodal tasks.

More Parameters. Due to computational constraints, the maximum number of MoH model parameters in our experiments is limited to 8B (MoH-LLaMA3-8B). However, our MoH method is highly generalizable and can be scaled to larger models in future research.

