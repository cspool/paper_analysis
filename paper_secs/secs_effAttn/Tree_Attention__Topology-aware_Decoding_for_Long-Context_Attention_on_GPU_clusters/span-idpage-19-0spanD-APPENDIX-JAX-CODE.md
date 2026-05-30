# <span id="page-19-0"></span>D APPENDIX: JAX CODE

Below is the *tree\_flash\_decode* method. Our full code base is available here: [https://](https://anonymous.4open.science/r/tree_attention-7C32) [anonymous.4open.science/r/tree\\_attention-7C32](https://anonymous.4open.science/r/tree_attention-7C32).

```
import jax
from jax import lax
import jax.numpy as jnp
from functools import partial
from jax.sharding import Mesh,NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from flash_attn_jax.flash import _flash_mha_vjp
in_specs=(P(None, None, None, None), P(None, 'i', None, None), P(None,
   'i', None, None))
out_specs=P(None, None, None)
@jax.jit
@partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
   check_rep=False)
def tree_flash_decode(q, k, v):
   def flash_num_lse(q, k, v, config=dict(softmax_scale=1.0,
      is_causal=False, window_size=(-1, -1))):
      tup = _flash_mha_vjp.fwd(q, k, v, config)
      res,lse = tup[1][3],tup[1][4]
      return res,lse
```

```
loc_res, loc_lse = flash_num_lse(q, k, v)
a_max_global = lax.pmax(loc_lse, axis_name='i')
num_global = lax.psum(loc_res * jnp.exp(loc_lse - a_max_global),
```

The function uses Flash Attention 2 Dao (2023) to compute the local numerator and denominator, both of which are accumulated between devices using an Allreduce (which is what psum and pmax call). NCCL determines in what pattern these results are communicated.

### <span id="page-20-0"></span>E THEOREM 1 PROOF

We prove theorem 1 below.

Proof.

**Sequential Case:** On a single GPU, the reduction operation over an array of size N has a time complexity of O(N) since the processor must sequentially process each element.

**Parallel Processing with** p **Processors:** Divide the array of size N into p chunks, each of size  $\frac{N}{p}$ . Each processor performs the reduction operation on its chunk independently. The time complexity for each processor is  $O\left(\frac{N}{p}\right)$ .

**Combining Partial Results:** The partial results from the p processors need to be combined. Using a tree pattern for reduction, the partial results can be reduced in  $O(\log p)$  steps. Each step involves combining pairs of results, halving the number of results at each step until only one result remains.

**Total Time Complexity:** The total time complexity is the sum of the time complexities for processing the chunks and combining the results:

$$O\left(\frac{N}{p}\right) + O(\log p).$$

This proves that the time complexity of a reduction involving an associative operation over an array of size N is  $O\left(\frac{N}{p} + \log p\right)$  when using p parallel processors, and it reduces to  $O(\log N)$  when the number of processors is equal to the size of the array.

#### <span id="page-20-1"></span>F COMPUTING SAFE SOFTMAX

While, mathematically, attention utilizes the softmax operation, in practice this is often numerically unstable using relatively low precision operations. To address this, a mathematically equivalent function, the 'safe softmax' is instead used which subtracts all dot products in the exponential by the max. This ensures that all values being exponentiated are less than 1 and hence less likely to explode and cause numerical instability. Here, we demonstrate that our energy function approach also can account for safe softmax.

Let us suppose we compare our generating function

$$F_{tot} = \sum_{i} \log \sum_{a=1}^{i} \exp\left(q_i \cdot k_a^T + \zeta_a \cdot v_a^T\right)$$
 (35)

and a slightly modified one:

$$F'_{tot} = \sum_{i} \log \sum_{a=1}^{i} \exp\left(q_i \cdot k_a^T + \zeta_i \cdot v_a^T - m_i\right). \tag{36}$$

When we take the derivative of these two quantities, we see that we get the same result:

$$\left. \frac{\partial F_{tot}}{\partial \zeta_i} \right|_{\zeta_i = 0} = \left. \frac{\partial F'_{tot}}{\partial \zeta_i} \right|_{\zeta_i = 0}.$$
(37)

To see it explicitly:

$$\frac{\partial F'_{tot}}{\partial \zeta_i} \bigg|_{\zeta_i = 0} = \frac{\sum_{a=1}^{i} \exp(q_i \cdot k_a^T - m_i) v_a}{\sum_{a=1}^{i} \exp(q_i \cdot k_a^T - m_i)} 
= \frac{\sum_{a=1}^{i} \exp(q_i \cdot k_a^T) v_a}{\sum_{a=1}^{i} \exp(q_i \cdot k_a^T)}.$$
(38)

Normally, when computing the softmax in an online fashion, this procedure is performed where  $m_i$  is the row max of  $q \cdot k^T$ . This shift makes it so that the sum of exponentials doesn't lead to overflows.

### <span id="page-21-0"></span>G NOTATIONS FOR EQUATIONS

Here is a summary of the various variables and indices that will be used in the coming sections:

TABLE I: Variable names.

| X                | Attention Input                     |  |
|------------------|-------------------------------------|--|
| q, k, v          | Query, key and value vectors        |  |
| Γ                | Attention Log-likelihood            |  |
| ζ                | Source vector                       |  |
| m                | Max of $q \cdot k^T$                |  |
| Z                | Partition function                  |  |
| z                | Activation vector                   |  |
| n                | Attention numerator                 |  |
| d                | Attention denominator               |  |
| lse              | Attention score logsumexp           |  |
| F                | Generating function                 |  |
| $\boldsymbol{P}$ | Attention score probability density |  |

TABLE II: Index names and ranges.

| N                                         | Sequence length     |      |
|-------------------------------------------|---------------------|------|
| d                                         | Embedding dimension |      |
| $d_h$                                     | Head dimension      |      |
| p                                         | Number of devices   |      |
| t                                         | Chunk size N/p      | (40) |
| b                                         | Batch size          | (40) |
| $a, i, j \in \{1, \cdots, N\}$            | Sequence Indices    |      |
| $A, B \in \{1, \cdots, d\}$               | Embedding indices   |      |
| $\bar{A}, \bar{B} \in \{1, \cdots, d_h\}$ | Intra-head indices  |      |
| $h \in \{1, \cdots, n_h\}$                | Head indices        |      |
| $\hat{a}, \hat{b} \in \{1, \cdots, t\}$   | Intra chunk indices |      |