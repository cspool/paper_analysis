# 1 INTRODUCTION

Attention mechanisms have made significant contributions to the field of artificial intelligence, advancing various modalities such as language, vision, audio, video, graphs, and even time series [\(Achiam et al.,](#page-8-0) [2023;](#page-8-0) [Team,](#page-12-0) [2023\)](#page-12-0). The Transformer [\(Vaswani,](#page-12-1) [2017\)](#page-12-1), known for its ability to capture long-range dependencies, has become a foundational architecture in this space. However, traditional Transformers encounter computational challenges due to their quadratic time complexity, O(n 2 ), with respect to sequence length n, making it difficult to scale to long sequences. To overcome this limitation, several linear sequence modeling methods have been proposed, including linear attention [\(Katharopoulos et al.,](#page-10-0) [2020;](#page-10-0) [Qin et al.,](#page-11-0) [2023a;](#page-11-0) [Li et al.,](#page-10-1) [2025\)](#page-10-1), state space modeling [\(Gu & Dao,](#page-10-2) [2024;](#page-10-2) [Dao & Gu,](#page-9-0) [2024\)](#page-9-0), and linear RNNs [\(Peng et al.,](#page-11-1) [2024;](#page-11-1) [Qin et al.,](#page-11-2) [2024d\)](#page-11-2), which offer O(n) training complexity and O(1) inference complexity. These approaches often reduce the input sequence to a fixed-size hidden space, collapsing the information into a single "memory state". While these methods enhance efficiency, they face two main challenges: limited memory capacity and memory interference. When new information overwrites the single fixed-size memory state, previously stored representations may degrade, which negatively impacts its long-term memory performance on recall-intensive tasks.

We argue that the strong performance of Transformer models on recall-intensive tasks arises from their ability to avoid memory interference by maintaining independent key-value caches for each

<sup>\*</sup> Interns at Shanghai AI Laboratory; B Corresponding Authors: Weigao Sun (sunweigao@outlook.com) and Yu Cheng (chengyu@cse.cuhk.edu.hk); † Project Lead.

token, thus offering virtually unlimited memory capacity. In contrast, linear sequence modeling relies on extreme compression, consolidating all the input information into a single fixed-size memory state (Katharopoulos et al., 2020; Dao & Gu, 2024). This approach results in limited memory capacity and inherently leads to memory interference issues.

Interestingly, the human brain has developed mechanisms that enable large memory capacity while reducing memory interference. Neuroscience studies show that in the hippocampus, theta oscillations (4~8 Hz) and gamma oscillations (30~100 Hz) work together to support a neural coding mechanism for multi-item memory (Buzsáki, 2002; Lisman & Jensen, 2013). Specifically, each theta cycle is subdivided into multiple gamma subcycles, and within each gamma subcycle, a distinct group of neurons is activated following the "E%-max" mechanism (de Almeida et al., 2009). This sequential activation temporally separates different memory items, thus preventing interference.

Inspired by these biological insights, we propose a new architecture called **Mixture-of-Memories** (**MoM**), which aims to strike a balance between the explicit token representations in Transformers and the extreme compression found in earlier linear sequence modeling methods. MoM employs multiple independent memory states, with a router network that directs input tokens to specific memory states. The input sequence is divided into a predefined number of subsequences (phase-specific neural assemblies), which are processed in parallel and fed into the corresponding memory projections (dentate microcircuits) to generate key-value pairs. As the linear sequence modeling layer processes each subsequence using an RNN-like update mechanism, it produces multiple memory states that capture different aspects of the input sequence. The final output is computed as a weighted sum of these memories, which we refer to as the mixture-of-memories. This approach expands memory capacity and eliminates memory interference, enabling MoM to significantly outperform existing linear sequence models that rely on a single fixed-size memory state.

Our contributions can be summarized as follows:

- We present MoM, an architecture that incorporates multiple independent memory states, significantly enhancing memory capacity and eliminating memory interference, while retaining the efficiency benefits of linear-time training and constant-memory inference.
- Distinct with existing gating mechanisms, MoM is a new paradigm to reduce memory interference by separating the memory states. The overall design is broadly compatible with diverse linear sequence modeling methods, making it a straightforward and effective approach to boost task performance.
- Through empirical evaluation, we show that MoM outperforms strong linear sequence modeling baselines across a variety of language tasks, particularly on recall-intensive tasks. MoM even achieves performance on par with Transformer models, a feat that current linear sequence modeling methods struggle to match.

#### 2 Preliminary

For notations in this work, we use bold lower-case letters for row vectors (e.g.,  $q_t, k_t$ ), bold upper-case letters for matrices (e.g., Q, K) and the identical letters represent a row in the matrix, e.g.,  $q_t$  is the t-th row of Q.

## LINEAR ATTENTION

To reduce the time complexity of Transformer attention, various optimization techniques have been proposed. Linear Transformers (Katharopoulos et al., 2020) replace the softmax attention mechanism with dot-product of feature maps  $\phi(\cdot)$ :

$$o_t = \frac{\sum_{i=1}^n \phi(q_t) \phi(k_i)^T v_i}{\sum_{i=1}^n \phi(q_t) \phi(k_i)^T},$$
(1)

where  $q_t, k_t, v_t \in \mathbb{R}^d$ . While the presence of the denominator may lead to numerical instability (Qin et al., 2024b) and the feature map can utilize an identity function, which we omit for simplicity. In perspective of memory, the formulation can also be written in a recurrent format:

$$M_t = M_{t-1} + k_t^T v_t, \quad o_t = q_t M_t.$$
 (2)

This indicates that linear attention can function as a linear recurrent layer with a matrix-valued hidden state M which we refer to as memory sate and the output is generated by querying the memory state M. This represents the ultimate compression of sequence information, condensing the entire sequence into a single memory state.

Building on the foundational concepts of linear attention and memory perspective, some recent advancements have focused on optimizing memory structure, including gated updates [\(Yang et al.,](#page-12-2) [2023;](#page-12-2) [Qin et al.,](#page-11-4) [2024e](#page-11-4)[;d\)](#page-11-2) and memory capacity expansion [\(Peng et al.,](#page-11-1) [2024;](#page-11-1) [Qin et al.,](#page-11-2) [2024d\)](#page-11-2).

