# A.3 Expressiveness of MLA with Decoupled RoPE

The full MLA architecture, as defined in Section 3.3 (main paper), employs a decoupled RoPE strategy. The query  $\mathbf{q}_{t,i}$  and key  $\mathbf{k}_{t,i}$  for head i (in the MHA-like training paradigm, Equation 9) are:

$$\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R] \tag{29}$$

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R] \tag{30}$$

where  $\mathbf{k}_t^R$  is a shared RoPE key component across all heads for token t. The bilinear attention score (numerator of the softmax argument) for head i between query at t and key at i is:

$$(\mathbf{q}_{t,i}^C)^{\top} \mathbf{k}_{j,i}^C + (\mathbf{q}_{t,i}^R)^{\top} \mathbf{k}_j^R$$
(31)

Let's analyze the two components of this score:

- 1. Content Component Interaction:  $(\mathbf{q}_{t,i}^C)^{\top} \mathbf{k}_{j,i}^C$ . The content keys  $\mathbf{k}_{j,i}^C$  are derived from  $W^{UK}(W^{DKV}\mathbf{x}_j)$ . This key generation mechanism for  $\mathbf{k}_{j,i}^C$  is precisely that of the MLA-Factorized model discussed in Section A.1. As established, MLA-Factorized is strictly more expressive than GQA for the non-positional part of the representation.
- 2. **Positional Component Interaction**:  $(\mathbf{q}_{t,i}^R)^{\top} \mathbf{k}_j^R$ . This interaction, where h distinct query-side RoPE components  $\mathbf{q}_{t,i}^R$  attend to a single, shared key-side RoPE component  $\mathbf{k}_j^R$ , is an MQA structure specifically for the positional information. As shown in Section A.2.3, MQA is strictly more expressive than  $\text{MLA}_{\text{Factorized}}$ , and by extension, GQA.

In summary, we have demonstrated that the expressive power of MLA with decoupled RoPE is stronger than that of the traditional GQA. However, it is worth noting that in the previously proven proposition, the  $MLA_{\rm Factorized}$  does not have a low-rank decomposition on the query; this differs from DeepSeek MLA. In the full MLA architecture, the query is also decomposed.

