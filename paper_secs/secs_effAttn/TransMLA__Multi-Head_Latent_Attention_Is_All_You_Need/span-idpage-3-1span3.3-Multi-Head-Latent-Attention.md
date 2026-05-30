# <span id="page-3-1"></span>3.3 Multi-Head Latent Attention

MLA saves KV cache by multiplying the matrix  $W^{DKV} \in \mathbb{R}^{r_{kv} \times D}$  with the input sequence to obtain low-rank latent features. Then, it uses the matrices  $W^{UK}$  and  $W^{UV} \in \mathbb{R}^{hd \times r_{kv}}$  to derive

the key  $\mathbf{k}$  and value  $\mathbf{v}$  representations for each attention head. Additionally, MLA also decomposes  $W^Q$  to  $W^{DQ} \in \mathbb{R}^{r_q \times D}$  and  $W^{UQ} \in \mathbb{R}^{hd \times r_q}$ , which reduces the activation memory during training. For positional embedding, MLA uses a decoupled RoPE strategy that uses additional multi-head queries  $\mathbf{q}_{t,i}^R \in \mathbb{R}^{d^R}$  and a shared key  $\mathbf{k}_t^R \in \mathbb{R}^{d^R}$ , which are generated from  $W^{QR} \in \mathbb{R}^{hd^R \times r_q}$  and  $W^{KR} \in \mathbb{R}^{d^R \times d}$ , to carry RoPE, where  $d^R$  denotes the per-head dimension of the decoupled queries

$$\mathbf{c}_{t}^{KV} = W^{DKV}\mathbf{x}_{t}, \qquad \mathbf{c}_{t}^{Q} = W^{DQ}\mathbf{x}_{t},$$

$$[\mathbf{k}_{t,1}^{C}; \mathbf{k}_{t,2}^{C}; ...; \mathbf{k}_{t,h}^{C}] = \mathbf{k}_{t}^{C} = W^{UK}\mathbf{c}_{t}^{KV}, \qquad [\mathbf{q}_{t,1}^{C}; \mathbf{q}_{t,2}^{C}; ...; \mathbf{q}_{t,h}^{C}] = \mathbf{q}_{t}^{C} = W^{UQ}\mathbf{c}_{t}^{Q},$$

$$\mathbf{k}_{t}^{R} = \text{RoPE}(W^{KR}\mathbf{x}_{t}, t), \qquad [\mathbf{q}_{t,1}^{R}; \mathbf{q}_{t,2}^{R}; ...; \mathbf{q}_{t,h}^{R}] = \mathbf{q}_{t}^{R} = \text{RoPE}(W^{QR}\mathbf{c}_{t}^{Q}, t),$$

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^{C}; \mathbf{k}_{t}^{R}], \qquad (7) \qquad \mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^{C}; \mathbf{q}_{t,i}^{R}]. \qquad (8)$$

MLA supports switching between two computational paradigms tailored for different stages. During the compute-intensive training phase, it operates in a paradigm similar to standard MHA, where the computational overhead is slightly lower than that of conventional MHA, as shown in Equation 9. For communication-intensive inference, it can seamlessly switch to a paradigm resembling MQA, as described in Equation 10. In this inference paradigm, the latent features function as a shared large KV head, which interacts with all query heads and output heads to produce the final output efficiently. This operation is called the Absorb operation, which is crucial for accelerating inference speed.

$$\begin{aligned} [\mathbf{v}_{t,1}^C; \mathbf{v}_{t,2}^C; ...; \mathbf{v}_{t,h}^C] &= \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}, & \hat{\mathbf{q}}_{t,i} &= [W_i^{UK}^\top \mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R], & \hat{\mathbf{k}}_t &= [\mathbf{c}_t^{KV}; \mathbf{k}_t^R], \\ \mathbf{o}_{t,i} &= \sum_{j=1}^t \operatorname{softmax}_j (\frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d+d^R}}) \mathbf{v}_{j,i}^C, & \hat{\mathbf{o}}_{t,i} &= \sum_{j=1}^t \operatorname{softmax}_j (\frac{\hat{\mathbf{q}}_{t,i}^T \hat{\mathbf{k}}_j}{\sqrt{d+d^R}}) \mathbf{c}_j^{KV}, \\ \mathbf{y}_t &= W^O[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,h}], & (9) & \mathbf{y}_t &= W^O[W_1^{UV} \hat{\mathbf{o}}_{t,1}; ...; W_h^{UV} \hat{\mathbf{o}}_{t,h}], & (10) \\ \end{aligned}$$
 where  $W_i^{\{UK,UV\}}$  denotes slices of the projection matrices corresponding to the  $i$ -th attention head.

<span id="page-4-1"></span><span id="page-4-0"></span>

One of the main contributions of this paper is the seamless support for the Absorb operation, significantly enhancing inference speed.

