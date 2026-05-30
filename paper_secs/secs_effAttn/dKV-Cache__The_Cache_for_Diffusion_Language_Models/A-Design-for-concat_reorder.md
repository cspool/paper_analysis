# A Design for concat\_reorder

concat\_reorder is our implementation of dKV-Cache designed to improve the speed of dKV-Cache in diffusion language models. Unlike standard KV-Cache used in autoregressive models, dKV-Cache requires gathering and scattering keys and values from arbitrary positions, introducing indexing operations that are less efficient than the simple concatenation in contiguous space used in ARs.

In dKV-Cache, the cache process involves two additional indexing operations: (1) At the cache step: After computing keys and values, we need to gather the corresponding states of cached tokens at non-continuous positions. (2) At the reuse step: to obtain the whole matrices for key and value, we need to scatter these vectors back to their original positions in the sequence. In contrast, KV-Cache in ARs only requires matrix slicing and concatenation, making it significantly more efficient.

To minimize the overhead of gathering and scattering, we propose an algorithm similar to that of standard KV-Cache to avoid too many indexing operations. The key idea is to reorder token positions during the forward calculation of the Transformer, placing all cached tokens contiguously on one side (e.g., left) and newly decoded tokens on the other. This allows us to move parts of the indexing operation to the token level (matrices with shape [B, L]) instead of the intermediate states (matrices with shape [B, L, D]):

- At Step t-1: Gather the cached key K I\Mt−<sup>1</sup> t−1 and value states V I\Mt−<sup>1</sup> t−1 based on the position index I \ Mt−<sup>1</sup> with one indexing operation.
- At Step t: Reorder the sequence, making the cached tokens (at position I \ Mt−1) at the left, and uncached tokens (at position Mt−1) at the right.
- At Step t: Using concat\_reorder for K I\Mt−<sup>1</sup> t−1 , K Mt−<sup>1</sup> t and for V I\Mt−<sup>1</sup> t−1 , V Mt−<sup>1</sup> t : First, concatenate the cached and current key/value states directly without further gathering/scattering (concat, for getting all K and V to calculate attention), and reorder the whole KV matrics based on V I\M<sup>t</sup> t to get the cached states for the next step (reorder, for obtaining the cache).

The reorder operation is to know the position mapping from [I \ Mt−1;Mt−1] to [I \ Mt;Mt]. For example, if the unmasked position at t − 1 is [2, 4, 5] from a sequence of 8 tokens, and at step t + 1 is [2, 4, 5, 7]. Then [I \ Mt−1;Mt−1] would be [2, 4, 5, 0, 1, 3, 6, 7], and [I \ Mt;Mt] would be [2, 4, 5, 7, 0, 1, 3, 6]. The mapping would be [0, 1, 2, 7, 3, 4, 5, 6], and we only need to get the corresponding entries [0, 1, 2, 7] from h K I\Mt−<sup>1</sup> t−1 ; K Mt−<sup>1</sup> t i and h V I\Mt−<sup>1</sup> t−1 ; V Mt−<sup>1</sup> t i .

The only remaining thing is that the change of token position would impact the positional encoding. However, this is easy to solve; we can also reorder the positional embedding. Reordering positional embeddings is required only once per model evaluation and can be shared across layers, thus, it would not cost much time.

Furthermore, since our method introduces a one-step shift in caching, the position of cached tokens at step t corresponds to the token positions decoded from step t − 1. This alignment allows us to track which key and value entries need to be cached without storing the entire key/value matrices, which, to cache which tokens, can only be known after the decoding results at step t.

We present the pseudo-algorithm of our approach in Algorithm [1.](#page-13-0) While it largely improves inference speed over the naive implementation, the concat and reorder operations still introduce some overhead. We believe there is substantial potential for further optimization.

