# <span id="page-21-0"></span>**E** Inference Strategy

At test time, VideoAuto-R1 employs a confidence-based early-exit mechanism to determine whether to stop after generating the initial direct answer or to proceed with a full chain-of-thought rationale followed by a reviewed answer. Algorithm 1 summarizes this procedure, which consists of three main steps: (1) generate the initial answer, (2) compute its confidence score, and (3) decide whether to exit early or continue reasoning.

<span id="page-21-1"></span>For implementation simplicity, we terminate generation early by detecting the appearance of the opening <think> tag during greedy decoding. We then extract the token sequence enclosed in the first  $\$ block, which always precedes the <think> tag. Since the initial answer  $a_1$  typically consists of only a few tokens, this strategy enables low-overhead confidence computation while providing substantial savings in decoding latency and token budget whenever early exit is triggered.

### <span id="page-22-1"></span>Algorithm 1 Inference Strategy of VideoAuto-R1

```
Require: Trained model p_{\theta}, video v, question q, confidence threshold \tau, fallback string f
Ensure: Predicted answer \hat{a}
 1: Given input (v,q), perform greedy decoding until the first <think> tag is generated.
 2: Let a_1 = (t_1, \dots, t_L) be the tokens inside the first box, and let y_{<\ell_0} denote the prefix up to (and including)
    the opening of a_1
 a_1 = f then
                                                                                                   ▶ designated fallback string
        s(a_1) \leftarrow -1e6
 5: else
        Compute length-normalized confidence s(a_1) \leftarrow \frac{1}{L} \sum_{\ell=1}^{L} \log p_{\theta}(t_{\ell} \mid y_{\leq \ell_0 + \ell - 1}, x)
 6:
 7: end if
    if s(a_1) \geq \log \tau then
                                                                                                                       ▶ early exit
 8:
         Accept the initial answer
        return \hat{a} \leftarrow a_1
10:
11: else

        Resume decoding from the current prefix
12:
        Generate rationale r enclosed in <think>... </think> and the second boxed answer a_2
13:
        return \hat{a} \leftarrow a_2
14:
15: end if
```

