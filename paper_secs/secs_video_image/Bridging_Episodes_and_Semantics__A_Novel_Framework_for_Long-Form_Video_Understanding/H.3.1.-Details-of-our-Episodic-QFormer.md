# H.3.1. Details of our Episodic QFormer

The Episodic Q-Former, as visualized in Figure [7,](#page-11-3) extends the original QFormer architecture by inserting the Episodic COmpressor (ECO) described in Section [4.2.](#page-3-0) It begins with a set of initial queries that undergo a self-attention process, enhancing internal query representations. These queries then interact with episodic visual features through crossattention, allowing the incorporation of contextual visual information. The resulting enhanced queries are fed into our ECO module alongside existing query episodes, which represent previously processed queries grouped into episodes. ECO iteratively updates the query episodes, adding the new queries to the existing episodes. This Episodic QFormer allows the model to better handle long sequences or repeated queries by maintaining richer contextual knowledge across iterations.

To mitigate *temporal confusion* during merging, we apply positional encoding (PE) to frame features before ECO. This effectively discourages out-of-order merges by embedding temporal locality directly into similarity calculations. As an ablation, removing PE reduces MovieChat-1k accuracy from 78.6 to 77.3 on MovieChat-1k, indicating its effectiveness in preserving temporal coherence despite compression. Other studies such as Transformer-XL [\[8\]](#page-8-20) and Compressive Transformer [\[28\]](#page-9-19), also report performance drops when positional biases are removed from their compression modules.

ECO implicitly captures event frequency: frequent events naturally occur across multiple frames and thus have higher likelihoods of being retained or merged into reinforced prototypes within the memory bank. This selfreinforcing mechanism ensures high-importance (and often high-frequency) events remain well-represented. Explicit event frequency tracking is an idea worth exploring, however, we believe it would be more computationally intensive

<span id="page-11-2"></span><span id="page-11-1"></span>

| Dataset             | Max Epochs | LR   | Batch | Frames (N) | Episodes | Keep Ratio |
|---------------------|------------|------|-------|------------|----------|------------|
| MovieChat-1k (G)    | 1          | 1e-4 | 32    | 100        | 20       | 0.2        |
| MovieChat-1k (B)    | 1          | 1e-4 | 32    | 40         | 10       | 0.5        |
| LVU                 | 20         | 1e-4 | 32    | 100        | 20       | 0.2        |
| COIN                | 20         | 1e-4 | 32    | 100        | 20       | 0.2        |
| Breakfast           | 20         | 1e-4 | 32    | 100        | 20       | 0.2        |
| VideoMME (LongVA)   | -          | -    | 1     | 128        | 32       | 0.125      |
| VideoMME (Llava-OV) | -          | -    | 1     | 128        | 32       | 0.125      |

Table 9. Hyperparameters used for different datasets.

<span id="page-11-3"></span>![](_page_11_Picture_2.jpeg)

Figure 7. **Illustration of our Episodic QFormer:** We insert our ECO in the original QFormer to effectively and efficiently compute and aggregate queries across long video sequences. It returns query episodes representing the whole video.

and may force important but infrequent representations out of memory.

#### H.3.2. Details of SeTR

We design SeTR as an efficient tool to retrieve semantic information from a long video. Given tokens extracted from a long video sequence, we use a stride of size k, to form a group of  $\frac{N}{k}$  frames representing the number of semantics we want to extract. We then compress the remaining  $N-\frac{N}{k}$  frames into extracted  $\frac{N}{k}$  frames to obtain the semantic representations. SeTR is illustrated in Figure 8.

![](_page_11_Picture_7.jpeg)

Figure 8. **Illustration of SeTR:** Our Semantics reTRiever uses a stride of k split the videos into groups X of N/k frames and Y of  $N-\frac{N}{k}$  frames, then merge each frame from Y to its most semantically similar in X.

### <span id="page-11-0"></span>**H.4. Extended Ablations**

