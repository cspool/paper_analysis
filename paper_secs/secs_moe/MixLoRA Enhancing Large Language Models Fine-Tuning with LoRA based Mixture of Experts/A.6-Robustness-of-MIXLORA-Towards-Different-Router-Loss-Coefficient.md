# A.6 Robustness of MIXLORA Towards Different Router Loss Coefficient

Table 9: Accuracy comparison of MIXLORA and MIXDORA with different Router Loss for LLaMA2-7B on the commonsense reasoning tasks.

| PEFT Method | Router Loss Coef. | ARC-e | ARC-c | BoolQ | OBQA                                                         | Avg. |
|-------------|-------------------|-------|-------|-------|--------------------------------------------------------------|------|
|             | -                 | 75.5  | 55.5  | 72.8  |                                                              | 70.5 |
|             | 1e-3              | 77.7  | 58.1  | 72.7  |                                                              | 73.2 |
| MixLoRA     | 1e-2              | 77.0  | 56.4  | 73.1  |                                                              | 71.8 |
|             | 1e-1              | 76.6  | 55.7  | 72.7  | 78.8<br>84.4<br>80.6<br>80.8<br>79.2<br>80.9<br>80.6<br>79.8 | 71.5 |
|             | -                 | 77.7  | 56.9  | 72.8  |                                                              | 71.5 |
|             | 1e-3              | 77.5  | 58.2  | 72.6  |                                                              | 72.3 |
| MixDoRA     | 1e-2              | 77.6  | 56.2  | 73.0  |                                                              | 71.9 |
|             | 1e-1              | 77.3  | 54.6  | 72.0  |                                                              | 70.9 |

#### <span id="page-17-0"></span>A.7 Optimization Algorithm

By combining two optimization strategies mentioned in Section 3.3, reducing computational complexity (I) and multi-model high-throughout training (II), we propose a forward propagation algorithm described in Algorithm 1. Specifically, the multi-task input sequences  $\mathbf{T}^{l-1}$  include various token sequences from M tasks, where each token sequence is sequentially allocated to different MIXLORA modules for processing (line 1). Given that the pretrained dense model weights remain frozen, it becomes feasible to maintain two or more MIXLORA models that share the same pretrained dense model weights. This approach reduces GPU memory cost and improves training efficiency by allowing multiple MIXLORA modules to be trained on a single GPU and reducing kernel launch time. Next, we linearly project the token sequences of task t

## Algorithm 1 Optimal Forward Propagation of MIXLORA

```
Require: multi-task token sequence \mathbf{T}^{l-1}: (M, B, N, D)
Ensure: multi-task token sequence \mathbf{T}^l: (M, B, N, D)
 1: /* Allocate multi-task sequences to various MIXLORAs */ 2: for t in {multi-task sequences \mathbf{T}^{l-1}} do
                \mathbf{T}_t^{l-1}: (B, N, D) \leftarrow \mathbf{T}^{l-1}[t,:,:,:]
                \mathbf{r}_t: (\mathtt{B} \times \mathtt{N}, \mathtt{K}) \leftarrow \mathbf{Linear}_t(\mathbf{T}_t^{l-1})
  4:
  5:
                \mathbf{r}_t': (B \times N, K) \leftarrow \mathbf{Norm}(\mathbf{Top2}(\mathbf{Softmax}(\mathbf{r}_t)))
                /* Reduction of duplicative calculations */
  6:
                \begin{split} \bar{\mathbf{h}}_t^{W_1} \colon (\mathbf{B}, \mathbf{N}, \mathbf{D}') \leftarrow \mathbf{\dot{L}inear}_l^{W_1}(\mathbf{T}_t^{l-1}) \\ \bar{\mathbf{h}}_t^{W_3} \colon (\mathbf{B}, \mathbf{N}, \mathbf{D}') \leftarrow \mathbf{\dot{L}inear}_l^{W_3}(\mathbf{T}_t^{l-1}) \end{split}
  7:
  8:
                /* Calculate the outputs of LoRA experts */
  9:
                for k in {LoRA experts \mathbf{E}_t^l} do
10:
                       \hat{\mathbf{h}}_t^{W_1} \colon (\mathtt{B}, \mathtt{N}, \mathtt{D}') \leftarrow \bar{\mathbf{h}}_t^{W_1} + \mathbf{LoRA}_{\iota}^{W_1}(\mathbf{T}_{\iota}^{l-1})
                       \hat{\mathbf{h}}_t^t : (\mathbb{S}, \mathbb{N}, \mathbb{D}') \leftarrow \bar{\mathbf{h}}_t^{W_3} + \mathbf{LoRA}_k^{W_3} (\mathbf{T}_t^{t-1})
12:
                       \hat{\mathbf{h}}_t: (\mathsf{B}, \mathsf{N}, \mathsf{D}') \leftarrow \mathbf{SiLU}(\hat{\mathbf{h}}_t^{W_1}) \odot \hat{\mathbf{h}}_t^{K_{W_3}}
                       \mathbf{h}_t \colon (\mathtt{B}, \mathtt{N}, \mathtt{D}) \leftarrow \mathbf{Linear}_l^{W_2}(\hat{\mathbf{h}}_t) + \mathbf{LoRA}_k^{W_2}(\hat{\mathbf{h}}_t)
                       \mathbf{T}_t^l: (B, N, D) \leftarrow \mathbf{T}_t^l + \mathbf{h}_t \bigcirc \mathbf{r}_t'[:,k]
16:
                end for
                \mathbf{T}^l: (t+1, B, N, D) \leftarrow \mathbf{concat}(\mathbf{T}^l, \mathbf{T}^l_t.\mathbf{unsqueeze}(0))
18: end forReturn: \mathbf{T}^{t}
```

to the logits  $\mathbf{r}_t$  (line 4) and compute the normalized logits  $\mathbf{r}_t'$  of activated experts by employing Softmax and Top-2 functions (line 5). We observe that the shared FFN sublayer repeatedly participates in the computation in multiple LoRA experts, which can be avoided with the linear layer  $W_1$  and the linear layer  $W_3$  of the FFN. Therefore, the projected token sequences  $\bar{\mathbf{h}}_t^{W_1}$  and  $\bar{\mathbf{h}}_t^{W_3}$  are saved in advance before computing the outputs of each expert (lines 7 and 8). Finally, we compute the product of the k-th LoRA metrics  $LoRA_k$  plus the shared FFN weights as the weights of the k-th expert and weight the outputs of all activated experts with the logits  $\mathbf{r}_t'$  generated by the router to get the output token sequence of the t-th task (line 15).