# 4 Communication-computation Overlap

After optimizing parallelism strategies to minimize communication volume, we further reduce the communication overhead to nearly zero using comprehensive communication-computation overlapping techniques. Training large models

<span id="page-5-1"></span>

| Activation              | Shape                    | Obtained From                                |
|-------------------------|--------------------------|----------------------------------------------|
| hidden                  | [b, s/n, h]              | # Input                                      |
| ln1_out                 | [b, s/n, h]              | # RMSNorm(hidden)                            |
| qkv                     | [b, $s/n$ , $h(1+2/m)$ ] | <pre># MatMul(ln1_out, qkv_weight)</pre>     |
| q_rope                  | [b, s/n, h]              | <pre># RopeEmbedding(q)</pre>                |
| k_rope                  | [b, s/n, h/m]            | <pre># RopeEmbedding(k)</pre>                |
| qkv_a2a                 | [b, s, $h(1+2/m)/n$ ]    | <pre># All-to-All(q_rope, k_rope, v)</pre>   |
| attn                    | [b, s, h/n]              | <pre># SelfAttention(qkv_a2a)</pre>          |
| attn_a2a                | [b, s/n, h]              | # All-to-All(attn)                           |
| attn_out                | [b, s/n, h]              | <pre># MatMul(attn_a2a, out_weight)</pre>    |
| ln2_in                  | [b, s/n, h]              | <pre># Add(hidden, attn_out)</pre>           |
| ln2_out                 | [b, s/n, h]              | <pre># RMSNorm(ln2_in)</pre>                 |
| ln2_out_ag              | [b, s, h]                | # All-Gather(ln2_out)                        |
| ffn_in                  | [b*s*k/n, h]             | # Scatter(ln2_out_ag)                        |
| fc1_out                 |                          | <pre># GroupedGEMM(ffn_in, fc1_weight)</pre> |
| fc3_out                 | [b*s*k/n, fh]            | <pre># GroupedGEMM(ffn_in, fc3_weight)</pre> |
| fc2_in                  | [b*s*k/n, fh]            | # SiLU(fc1_out, fc3_out)                     |
| fc2_out                 | [b*s*k/n, h]             | <pre># GroupedGEMM(fc2_in, fc2_weight)</pre> |
| fc2_out_rs              | [b, s, h]                | # Gather(fc2_out)                            |
| ffn_out                 | [b, s/n, h]              | <pre># Reduce-Scatter(fc2_out_rs)</pre>      |
| <pre>hidden(next)</pre> | [b, s/n, h]              | <pre># Add(ln2_in, ffn_out)</pre>            |

Figure 9. Activation shapes in rematerialization.

involves integrating various techniques, which increases the complexity of communication overlap. For instance, at any given moment, the device might concurrently handle computation and communication kernels, overlap PP and DP communications, and manage data transfers between the device and host. Existing frameworks like Megatron-LM assemble attention and FFN modules into MoE layers and rely on the torch.autograd package for backward propagation, which limits the flexibility of communication overlap. In contrast, MegaScale-MoE decomposes the attention and FFN modules of each MoE layer into operators that run as GPU kernels, enabling fine-grained communication overlap through flexible scheduling.

