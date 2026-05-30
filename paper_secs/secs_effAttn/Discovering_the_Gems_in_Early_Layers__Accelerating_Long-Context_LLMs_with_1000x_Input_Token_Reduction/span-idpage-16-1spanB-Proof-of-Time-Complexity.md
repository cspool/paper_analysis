# <span id="page-16-1"></span>B Proof of Time Complexity

**Theorem B.1** (Complexity analysis. Restatement of Theorem 3.3). Let n be the input sequence (prompt) length and d the hidden feature dimensions. In our Algorithm 1, GemFilter uses the r-th layer as a filter to select k input tokens. Let SnapKV and H2O also use k as their cache size. Assume the LLM has m attention layers, each with h attention heads, and each transformer layer's parameters consume m GPU memory. Assuming that we generate m tokens with the GEN function and  $m \ge max\{d, k, t\}$ , the following table summarizes the complexity for standard attention, m and m and m and m and m and m and m and m and m and m and m and m and m and m and m and m and m and m are m and m and m and m and m are m and m and m and m and m are m and m and m are m and m and m are m and m are m and m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m are m and m are m are m are m and m are m and m are m and m are m are m and m are m and m are m and m are m and m are m and m are m are m and m are m are m and m are m are m and m are m and m are m and m are m are m and m are m are m and m are m and m are m and m are m are m and m are m and m are m are m and m are m are m are m and m are m and m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are m are

| Con      | mplexity                      | Standard attention                      | SnapKV and H2O                          | GemFilter                                |
|----------|-------------------------------|-----------------------------------------|-----------------------------------------|------------------------------------------|
| Time     | Prompt Comp. Iter. generation | $\Theta(mhn^2d) \\ \Theta(mh(nt+t^2)d)$ | $\Theta(mhn^2d) \\ \Theta(mh(kt+t^2)d)$ | $\Theta(rhn^2d) \\ \Theta(mh(k^2+t^2)d)$ |
| GPU mem. | Prompt Comp. Iter. generation | mw + 2mhnd $mw + 2mh(n+t)d$             | mw + 2hnd + 2mhkd $mw + 2mh(k+t)d$      | $rw + 2hnd \\ mw + 2mh(k+t)d$            |

*Proof of Theorem 3.3.* We prove each method separately.

#### Proof of standard attention:

During prompting computation, it takes  $\Theta(mhn^2d)$  time complexity, as there are m transformer layers, each layer has h attention head, and each head takes  $\Theta(n^2d)$  to calculate the attention (Attn<sub>i</sub> in Definition 3.2) and  $\Theta(nd)$  for other operations ( $g_i$  in Definition 3.2).

During iterative generation, it takes  $\Theta(mh(nt+t^2)d)$  time complexity.

During prompting computation, mw GPU memory consumption is taken for the model weights and 2mhnd GPU memory consumption for the KV cache.

During iterative generation, it takes mw GPU memory consumption for the model weights and 2mh(n + t)d GPU memory consumption for the KV cache. Proof of SnapKV and H2O:

During prompting computation, it takes Θ(mhn2d) time complexity, which is the same as standard attention.

During iterative generation, it takes Θ(mh(kt + t 2 )d) time complexity, as it reduces the KV cache size from n to k.

During prompting computation, mw GPU memory is consumed for the model weights, 2hnd for the selection of the key-value matrix for each layer, and 2mhkd for the selected KV cache.

During iterative generation, mw GPU memory is consumed for the model weights and 2mh(k+ t)d GPU memory is consumed for the KV cache.

#### Proof of our Algorithm [1](#page-6-1) GemFilter:

During prompting computation, GemFilter takes Θ(rhn2d) time complexity, which is faster than other methods.

During iterative generation, it takes Θ(mh(k <sup>2</sup> + kt + t 2 )d) = Θ(mh(k <sup>2</sup> + t 2 )d) time complexity, as it reduces the KV cache size from n to k.

During prompting computation, rw + 2hnd GPU memory is consumed for the model weights and the selection of the key value matrix for each layer.

During iterative generation, mw + 2mh(k + t)d GPU memory is consumed for the KV cache and model weights.

Thus, we finish the proof.

