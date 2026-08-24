# <span id="page-2-1"></span>3.2 Background on permutation-based neuron alignment

We propose a merging technique that combines several similar sublayers into a single parameter set. Our merging technique is inspired by prior work in permutation symmetries of neurons [\(Li et al.,](#page-9-11) [2015\)](#page-9-11). This technique has been used in studying models' convergent learning, as well as merging two or more separate models [\(Tatro et al.,](#page-10-6) [2020;](#page-10-6) [Entezari et al.,](#page-8-7) [2022;](#page-8-7) [Ainsworth et al.,](#page-8-8) [2023\)](#page-8-8).

Permutation-based alignment techniques seek to find an optimal reordering of neurons in one layer that more closely matches ordering of neurons from another layer, without changing the output. Given two layers to align, we compute only forward passes through both using exemplar data to collect activations. These layers are generally corresponding from different models. This results in two activation sets Xα, X<sup>β</sup> ∈ R n×d , where n is the number of example data points, and d is the model dimension.

To determine corresponding neurons from the activations, we compute cross-correlation C, in line with prior work [\(Li et al.,](#page-9-11) [2015\)](#page-9-11). µ represents mean vectors, and σ standard deviation vectors.

$$C = \frac{\mathbb{E}\left[ (X_{\alpha} - \mu(X_{\alpha}))^{T} (X_{\beta} - \mu(X_{\beta})) \right]}{\sigma(X_{\alpha})\sigma(X_{\beta})}$$
(1)

The resulting matrix C ∈ R d×d reflects how each neuron j in X<sup>α</sup> correlates with each neuron k in Xβ. To find the neuron alignment that maximizes total correlation, we solve the following optimization problem, where Π<sup>d</sup> is the space of all permutations of length d [\(Li et al.,](#page-9-11) [2015;](#page-9-11) [Tatro et al.,](#page-10-6) [2020\)](#page-10-6):

$$\pi^* = \max_{\pi \in \Pi_d} \sum_{j=1}^d C(j, \pi(j))$$
 (2)

This problem is a case of the Linear Assignment Problem, and we solve for π <sup>∗</sup> using the Jonker-Volgenant algorithm implementation provided by scipy [\(Crouse,](#page-8-9) [2016\)](#page-8-9).

## <span id="page-2-2"></span>3.3 Combining feed-forward sublayers

Now, with the appropriate background, we describe our compression method. For our method, we assume we have some predetermined number of FF sublayers k to merge. This number can be inferred given a desired parameter reduction ratio, or set otherwise.

Given a window of k adjacent FF sublayers, we compute a forward pass using a subset of data in order to compute features for each sublayer. In other words, for Transformer FF sublayer x out = Woutϕ(Winx in + b in) + b out, we obtain features just before the ϕ activation. We consider only the neurons just *after* Win because prior work has shown that to reorder the input to Win and output of Wout requires permuting many additional weights due to the residual connections in order to maintain functional equivalence [\(Verma and Elbayad,](#page-10-7) [2024\)](#page-10-7). For each of the k feed-forward sublayers, we collect features X<sup>i</sup> ∈ R n×d , i ∈ [0, k − 1], where d is the feed-forward dimension.[2](#page-2-0)

We designate the first FF sublayer of the set to be an "anchor", and compute the permutation-finding algorithm on each pair of features where one index is always the anchor. In other words, for each sublayer i ∈ [1, k − 1], we find π<sup>i</sup> between X<sup>0</sup> and X<sup>i</sup> using the assignment method from Section [3.2.](#page-2-1)

After converting function π<sup>i</sup> to its corresponding permutation matrix P<sup>i</sup> , we transform the k − 1

<span id="page-2-0"></span><sup>2</sup>The layer indices reflect local index within the set of k versus global layer index.

non-anchor FF sublayers. We then average these k FF sublayers, and replace each of them with their average, as in Equations [3](#page-3-0)[–6.](#page-3-1) [3](#page-3-2) Finally, we tie these weights so that in memory they appear as just one sublayer, effectively removing the parameters from k − 1 FF sublayers.

$$W^{\text{in}*} = \frac{1}{k} \left( W_0^{\text{in}} + \sum_{i=1}^{k-1} P_i W_i^{\text{in}} \right)$$
 (3)

$$b^{\text{in}*} = \frac{1}{k} \left( b_0^{\text{in}} + \sum_{i=1}^{k-1} P_i b_i^{\text{in}} \right)$$
 (4)

$$W^{\text{out*}} = \frac{1}{k} \left( W_0^{\text{out}} + \sum_{i=1}^{k-1} W_i^{\text{out}} P_i^T \right)$$
 (5)

$$b^{\text{out}} = \frac{1}{k} \left( \sum_{i=0}^{k-1} b_i^{\text{out}} \right) \tag{6}$$

Practical implications The consequences of tying these weights and reducing model size are 1) the model occupies less space on disk or on GPU when fine-tuning (as well as its gradients), making it easier to use smaller hardware and 2) if some optimizations are made, inference speed and throughput can be improved. While naively our method does not target inference speed, we discuss ways this could be achieved. For example, if larger batch sizes are used given the model memory savings, this can result in increased throughput on the same hardware. Additionally, specific efficient GPU+CPU execution techniques like layer-to-layer that involve on- and off-loading parameters may also benefit from this sharing scheme by reducing the number and size of data transfers [\(Pudipeddi](#page-9-12) [et al.,](#page-9-12) [2020;](#page-9-12) [Aminabadi et al.,](#page-8-10) [2022\)](#page-8-10).

#### <span id="page-3-4"></span>3.4 Selecting sublayers to merge

In selecting the k adjacent feed-forward sublayers to merge, we take a sliding window approach. For all starting layer indices from 0 to (Nlayers −1)−k, we apply the method outlined in Section [3](#page-2-2).3, and evaluate the resulting model on a validation set.

Although we propose to test each potential window, in reality, the cost of computing permutations and parameter arithmetic is low, and it scales only linearly with the number of layers. The largest costs in each iteration is computing features and testing candidates. However, we can compute features only *once* despite testing Nlayers − k models, because one forward pass through the exemplar data is sufficient for creating all necessary correlation matrices. The best candidate is the one with the highest post-merge evaluation score. We note that there may be other possible selection heuristics.

<span id="page-3-0"></span>Finally, we follow our merging procedure with a short recovery fine-tuning to quickly heal performance on the downstream task.[4](#page-3-3)

