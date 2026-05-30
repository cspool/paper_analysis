 # Aggregate expert outputs (3)

**Product Key Retrieval** Since we intend to use a very large number of experts  $(N \ge 10^6)$ , naively computing the top k indices in Eq. 1 can be very expensive. Hence we apply the product key retrieval technique here. Instead of using N independent d-dimensional vectors as our keys  $k_i$ , we create them by concatenating vectors from two independent sets of  $\frac{d}{2}$ -dimensional sub-keys  $\mathbb{C}, \mathbb{C}' \subset \mathbb{R}^{\frac{d}{2}}$ :

<span id="page-2-1"></span>
$$\mathbb{K} = \{ \begin{bmatrix} c \\ c' \end{bmatrix} | c \in \mathbb{C}, c' \in \mathbb{C}' \}$$
 (4)

Note that here  $\mathbb{C}, \mathbb{C}'$  have cardinality  $\sqrt{N}$  and c, c' have dimensionality  $\frac{d}{2}$ . So in practice, we choose N to be a perfect square and d to be an even number.

This Cartesian product structure of K allows us to find the top *k* experts efficiently. Instead of comparing *q*(*x*) to all *N* keys in K and selecting the top k matches, we can split the query vector *q*(*x*) into two subqueries *q*<sup>1</sup> and *q*<sup>2</sup> and apply the top k operations to the inner products between the sub-queries and sub-keys respectively:

$$\mathbb{I}_{\mathbb{C}} = \mathcal{T}_k \left( (q_1^T c_i) \right), \qquad \mathbb{I}_{\mathbb{C}'} = \mathcal{T}_k \left( (q_2^T c_j') \right)$$
 (5)

This results in a set of *k* 2 candidate keys K′ := { *ci cj* |*i* ∈ IC*, j* ∈ I ′ C }, and it is mathematically guaranteed that the *k* most similar keys to *q*(*x*) from K are in this candidate set. Moreover, the inner product between the candidate key and *q*(*x*) is simply the sum of inner products between the sub-keys and sub-queries: *q*(*x*) *T ci cj* = *q T* 1 *c<sup>i</sup>* + *q T* 2 *c<sup>j</sup>* . Hence we can apply the top-k operator again to these *k* 2 inner products to get the top k matching keys from the original set of product keys K. As explained in [Lample et al.](#page-11-6) [\(2019\)](#page-11-6). This reduces the complexity of top k expert retrieval in Eq. [1](#page-2-1) from *O*(*N d*) as done naively by exhaustive search to *<sup>O</sup>*((<sup>√</sup> *N* + *k* 2 )*d*).

Parameter Efficient Experts and Multi-Head Retrieval Unlike other MoE architectures, which often set the hidden layer of each expert to the same size as other FFW layers, in PEER, every expert *e<sup>i</sup>* is a singleton MLP, in other words, it has only one hidden layer with a single neuron:

$$e_i(x) := \sigma(u_i^T x) v_i \tag{6}$$

where *v<sup>i</sup> , u<sup>i</sup>* are not matrices but vectors with the same dimension as *x*, and *σ* is a nonlinear activation function such as ReLU or GELU. We omit bias terms here for brevity.

Instead of varying the size of individual experts, we adjust the expressiveness of a PEER layer by using multihead retrieval, similar to the multi-head attention mechanism in transformers and the multi-head memory in PKMs. In particular, we use *h* independent query networks instead of one, each computes its own query and retrieves a separate set of *k* experts. However, different heads share the same pool of experts with the same set of product keys. The outputs of these *h* heads are simply summed up:

$$f(x) := \sum_{i=1}^{h} f^{i}(x) = \sum_{i=1}^{h} \sum_{j \in \mathbb{I}^{i}} g_{j}(x) e_{j}(x)$$
 (7)

One can verify that when only one expert is retrieved (*k* = 1) per head, using a PEER layer with *h* heads is the same as using one expert with *h* hidden neurons:

$$f(x) = \sum_{i=1}^{h} e^{i}(x) = \sum_{i=1}^{h} \sigma(u_{i}^{T} x) v_{i} = V \sigma(W^{T} x);$$
(8)

where *W* = [*u*1*,* · · · *, uh*]*, V* = [*v*1*,* · · · *, vh*]. In other words, PEER dynamically assembles an MLP with *h* neurons by aggregating *h* singleton MLPs retrieved from a shared repository. Compared to existing MoE approaches that use MLPs with multiple hidden neurons as experts, this design allows shared hidden neurons among experts, enhancing knowledge transfer and parameter efficiency.

Algorithm [1](#page-4-0) shows a simplified implementation of the PEER forward pass, storing parameter-efficient expert weights in embedding layers and combining them with einsum operations. This implementation can be easily extended to experts of the GLU variants [\(Shazeer, 2020\)](#page-11-7) by adding additional linear gating weights. In practice, an efficient implementation may require specialized hardware kernels to accelerate embedding lookup and fusion with the einsum operations.

Why A Large Number of Small Experts? Given an MoE layer, we can characterize it by three hyperparameters: the total number of parameters *P*, the number of active parameters per token *P*active and the size of a single expert *P*expert. [Krajewski et al.](#page-10-4) [\(2024\)](#page-10-4) showed that the scaling law of MoE models has the following form:

<span id="page-3-0"></span>
$$\mathcal{L}(P, D, G) = c + \left(\frac{g}{G^{\gamma}} + a\right) \frac{1}{P^{\alpha}} + \frac{b}{D^{\beta}},\tag{9}$$

where L is the final test loss, *a, b, g, γ, α, β* are constants, *D* is the total number of training tokens and the granularity *G* is the number of active experts:

$$G := \frac{P_{\text{active}}}{P_{\text{expert}}} \tag{10}$$

In order to improve model performance, we need to scale up *P, D, G*. On the other hand, it is essential to limit *P*active because the computational and memory costs are primarily determined by the active parameters during training and inference. Notably, the memory footprint corresponding to *P*active has to be multiplied by the number of tokens in a batch, while the memory cost of *P* is independent of the batch size and sequence length because only one copy of the model needs to be stored.

As a result, we want to increase *P, G* but not *P*active. Since the expert size *P*expert = *P*active*/G* and the number of experts *N* = *P/P*expert = *P* · *G/P*active, this implies that we should decrease the size of each expert, *P*expert, and increase the number of experts *N*. Hence we need a large number of small experts.

In general, for experts that are MLPs with a single hidden layer. *P*expert = (2*d*model + 1)*d*expert and *P*active = (2*d*model+1)*d*active, where *d*model, *d*expert and *d*active are the hidden dimension of the transformer, the number of hidden neurons used in one expert and the total number of hidden neurons activated per token, respectively.

In the case of PEER, we use the smallest expert size possible by setting *d*expert = 1, and the number of activated neurons is the number of retrieval heads multiplied by the number of experts retrieved per head: *d*active = *hk*. Consequently, the granularity of PEER is always *G* = *P*active*/P*expert = *d*active*/d*expert = *hk*.

```
1 def peer_forward ( self , x):
2 # Embedding layers storing the down /up projection weights of all experts
3 self . w_down_embed = nn . Embed ( num_embeddings = self . n_experts , features = self . d_model )
4 self . w_up_embed = nn . Embed ( num_embeddings = self . n_experts , features = self . d_model )
6 # Retrieve the weights of the top matching experts using product keys
7 # indices and scores have the shape 'bthk ', where h is the number of heads
8 indices , scores = self . get_indices ( self . query_proj (x) , self . sub_keys , top_k = self .k)
9 w_down = self . w_down_embed ( indices )
10 w_up = self . w_up_embed ( indices )
12 # Compute weighted average of expert outputs
13 x = jnp . einsum ('btd , bthkd - > bthk ', x , w_down )
14 x = self . activation (x)
15 x = x * nn . softmax ( scores )
16 x = jnp . einsum ('bthk , bthkd -> btd ', x , w_up )
17 return x
```

Algorithm 1: Pseudo code implementation of a PEER layer forward pass. An example implementation of the get\_indices and query\_proj functions in Pytorch can be found in [Lample et al.](#page-11-8) [\(2021\)](#page-11-8)

## <span id="page-4-1"></span>**3 Experiments**

#### **3.1 Pretraining isoFLOP Analysis**

We compare PEER with various baselines using isoFLOP analysis [\(Borgeaud et al., 2022b\)](#page-9-4). We chose a fixed FLOP budget (6*e*18 and 2*e*19) and jointly varied the model size and the number of training tokens from the C4 dataset [\(Raffel et al., 2020\)](#page-11-9) to obtain isoFLOP curves. Each point on an isoFLOP curve has the same computational cost, and we plot them in terms of their model size and final validation perplexity on C4.

For the dense baselines, we varied their size by changing the number of layers, attention heads and model dimensions. For MoE, PKM and PEER methods, we took each of the dense models considered and replaced the FFW layer in the middle block (e.g. in a 12 block transformer, we replace the FFN in block 6) by a layer of MoE, PKM and PEER, respectively.

In MoE, we used the expert-choice [\(Zhou et al., 2022\)](#page-11-2) routing algorithm, which effectively addresses the expert load imbalance issue and generally outperforms token-choice MoEs (see Section [4](#page-7-0) for a review and comparison of these approaches). Each expert has the same size as the original MLPs in the corresponding dense model, and we use 128 experts to cover the same range of model sizes as our PEER models. This type of MoE represents standard coarse-grained MoE approaches, which consist of a small number of large experts.

In PKM, we used 1024<sup>2</sup> memories with *h* = 8 heads and top *k* = 32 memories were selected per head. We also applied query batch normalization, as recommended in the original PKM paper [\(Lample et al., 2019\)](#page-11-6), to enhance memory usage.

In PEER, we used 1024<sup>2</sup> experts with *h* = 8 heads and top *k* = 16 experts per head. By default, we also enabled query BatchNorm to increase expert usage. Ablation studies in subsection [3.3](#page-5-0) investigate the effect of these hyperparameters. Unlike the expert-choice MoE baseline, PEER represents a fine-grained approach where a large number of small experts are employed.

Across all model sizes and methods, we maintained a consistent batch size (128) and sequence length (2048). We calculated the number of training steps by dividing the total compute budget by the FLOPs per training step. Fig. [1](#page-0-0) presents the isoFLOP profiles. Compared to the dense FFW baseline, the sparse alternatives shift the isoFLOP curves downward and to right because they introduce a larger number of total parameters *P* but utilize a smaller or equal number of active parameters *P*active. Given the same compute budget, a PEER model achieves the lowest compute-optimal perplexity.

#### **3.2 Evaluation on Language Modeling Datasets**

After determining the compute-optimal model for each method based on the isoFLOP curves, we evaluated the performance of these pretrained models on several popular language modeling datasets, including Curation Corpus [\(Curation, 2020\)](#page-9-5), Lambada [\(Paperno et al., 2016\)](#page-11-10), the Pile [\(Gao et al., 2020\)](#page-10-8), Wikitext [\(Merity](#page-11-11) [et al., 2016\)](#page-11-11) and the pretraining dataset C4. Table [1](#page-5-1) presents a summary of the evaluation results. We grouped the models based on their FLOP budgets used during training.

| Method       | Curation<br>Corpus | Lambada | Pile  | Wikitext | C4    |
|--------------|--------------------|---------|-------|----------|-------|
| Dense (6e18) | 23.26              | 21.95   | 24.55 | 29.14    | 23.84 |
| MoE (6e18)   | 20.98              | 19.09   | 23.26 | 26.10    | 21.41 |
| PKM (6e18)   | 21.80              | 19.39   | 20.49 | 27.09    | 21.92 |
| PEER (6e18)  | 20.68              | 17.65   | 19.01 | 25.48    | 20.63 |
| Dense (2e19) | 17.70              | 12.28   | 18.19 | 21.21    | 18.31 |
| MoE (2e19)   | 16.88              | 12.97   | 17.41 | 20.28    | 17.12 |
| PKM (2e19)   | 17.03              | 11.18   | 16.34 | 20.26    | 17.36 |
| PEER (2e19)  | 16.34              | 10.33   | 14.99 | 19.09    | 16.45 |

<span id="page-5-1"></span>Table 1: Perplexities of the compute-optimal models of each method on language modeling datasets.

