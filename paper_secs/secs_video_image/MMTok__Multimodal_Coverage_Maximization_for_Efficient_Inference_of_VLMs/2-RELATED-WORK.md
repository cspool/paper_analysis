# 2 RELATED WORK

VLMs, such as LLaVA [\(Liu et al.,](#page-11-8) [2023\)](#page-11-8), InstructBLIP [\(Dai et al.,](#page-10-6) [2023\)](#page-10-6), and Qwen [\(Bai et al.,](#page-10-2) [2025\)](#page-10-2), have become a cornerstone for multimodal understanding by integrating large-scale vision encoders

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

Figure 2: **Overview of MMTok framework.** Our method optimizes two maximum coverage problems simultaneously to leverage text-vision and vision-vision similarity for vision token selections.

(e.g., CLIP-ViT (Radford et al., 2021b)) with pre-trained language models. These models achieve strong performance by representing images as sequences of visual tokens, but their inference cost grows quadratically with token count, highlighting the need for more efficient processing.

Many vision token selection methods have been proposed recently, but most of them rely only on unimodal information for pruning (Yang et al., 2025a; Shang et al., 2024; Chen et al., 2024a; Zhang et al., 2024; Alvar et al., 2025). For example, VisionZip (Yang et al., 2025a) and FastV (Chen et al., 2024a) prune tokens using pre-trained attention signals, either ranking by [CLS] token attention (VisionZip) or discarding low-attention vision tokens in deeper layers (FastV). Besides ranking, DivPrune (Alvar et al., 2025) uses a diversity-based criterion but only has vision tokens to maximize the intra-set diversity. These methods rely on vision information and may miss query-related semantics (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019). SparseVLM (Zhang et al., 2024) instead uses text-to-vision attention for scoring, yet ignores the information from the whole image. To mitigate the gap between existing unimodal algorithms and target multimodal tasks, we propose a coverage-based criterion to leverage both vision and text information sufficiently to select vision tokens effectively.

#### 3 THE PROPOSED METHOD

To leverage the power of pre-trained models, many existing VLMs adopt a pre-trained vision encoder to extract vision tokens from images and then concatenate them with text tokens as input for the pre-trained LLMs. Although the simple architecture demonstrates promising performance, the inference efficiency can be challenging. Concretely, given an image, a pre-defined number of vision tokens will be obtained as  $\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}$ . Even for a small  $336\times336$  image, n is 576 with the ViT-L-336px from CLIP (Radford et al., 2021a), which is much larger than that of the text tokens from the text query (Liu et al., 2023). The large n will significantly slow down the inference of LLMs, which relies on the self-attention operations, and the complexity is quadratic to the total number of tokens.

To accelerate the inference of VLMs, we propose to select an informative subset of vision tokens  $\{v_s\}_{s\in\mathcal{S}}$  to reduce the number of input tokens for LLM in VLM, where  $\mathcal{N}=\{1,\ldots,n\}$ ,  $\mathcal{S}\subseteq\mathcal{N}$ , and  $|\mathcal{S}|\ll n$ . Figure 2 illustrates the framework of our method, and we describe it as follows.

#### 3.1 VISION TOKEN SELECTION BY COVERAGE MAXIMIZATION

Unlike most of the existing work, we apply coverage as the main criterion for token selection. Given a similarity matrix  $M \in \mathbb{R}^{m,n}$  defined between target tokens and source tokens, where m denotes

the number of target tokens and n is the number of source tokens, a subset S will be selected to maximize the similarity between the target and selected tokens as

<span id="page-3-0"></span>
$$f(\mathcal{S}; M) = \frac{1}{m} \sum_{i=1}^{m} \max M_{i,\mathcal{S}}; \quad \mathcal{S}^* = \arg \max_{\mathcal{S}} f(\mathcal{S}; M)$$
 (1)

a.k.a. covering the target tokens by an appropriate subset of source tokens. We first find that Eq. [1](#page-3-0) is a popular submodular function [\(Leskovec et al.,](#page-10-9) [2007\)](#page-10-9).

Proposition 1. *[\(Leskovec et al.,](#page-10-9) [2007\)](#page-10-9) For all subsets* A ⊆ B ⊆ N *and* s ∈ N \ B*,*

$$f(\mathcal{A} \cup \{s\}) - f(\mathcal{A}) \ge f(\mathcal{B} \cup \{s\}) - f(\mathcal{B})$$

Maximizing submodular functions in general is NP-hard [\(Khuller et al.,](#page-10-5) [1999\)](#page-10-5), but a simple greedy algorithm can achieve a good approximation.

<span id="page-3-1"></span>Proposition 2. *[\(Nemhauser et al.,](#page-11-7) [1978\)](#page-11-7) Let* S *denote the subset obtained by the greedy algorithm, then we have*

$$f(S) \ge (1 - 1/e) \max_{A:|A| = |S|} f(A)$$

We elaborate on how to apply the coverage function for token selections in the following subsections.

#### 3.1.1 MAXIMUM TEXT-VISION COVERAGE

First, we consider covering the semantics from text tokens with source vision tokens, which aims to find the vision tokens related to the text input (e.g., query). Let {t1, . . . , tm} denote the text tokens from the query. A similarity matrix between text and vision tokens can be obtained as

$$M_{i,j}^{tv} = \mathbf{t}_i^{\top} \mathbf{v}_j$$

where Mtv ∈ R <sup>m</sup>×<sup>n</sup> and ∀i, j, ∥ti∥<sup>2</sup> = ∥vj∥<sup>2</sup> = 1. To align the semantic similarity between text and vision, we adopt the vision tokens after the projection layer (i.e., those concatenated with text tokens as input for LLMs). After obtaining the appropriate similarity matrix, a subset of vision tokens can be selected to maximize the similarity between all text tokens and selected vision tokens for coverage as

$$S' = \arg \max_{S} f(S; M^{tv})$$

According to Proposition [2,](#page-3-1) a greedy algorithm as summarized in Alg. [1](#page-3-2) can approximate the optimal solution. It should be noted that the proposed Alg. [1](#page-3-2) contains only simple operations (e.g., argmax, matrix multiplication, etc.) and thus is efficient for implementation.

### <span id="page-3-2"></span>Algorithm 1 A Greedy Algorithm to Cover Text Input with Vision Tokens

<span id="page-3-3"></span>Algorithm 2 MMToK: A Greedy Algorithm for Multimodal Coverage

```
1: Input: Similarity Matrix Mtv
                                , k
 2: Initialize S = ∅
 3: for i = 1, · · · , k do
 4: for s ∈ N \ S do
 5: Compute g(s) = f(S ∪ s; Mtv)
 6: end for
 7: Obtain si = arg maxs g(s)
 8: S = S ∪ si
 9: end for
10: return S
                                            10: return S
```

```
1: Input: Similarity Matrices Mtv′
                                  , Mvv′
                                        , k
2: Initialize S = ∅
3: for i = 1, · · · , k do
4: for s ∈ N \ S do
5: Compute g(s) = f(S ∪ s; Mtv′
                                      , Mvv′
                                            )
6: end for
7: Obtain si = arg maxs g(s)
8: S = S ∪ si
9: end for
```

