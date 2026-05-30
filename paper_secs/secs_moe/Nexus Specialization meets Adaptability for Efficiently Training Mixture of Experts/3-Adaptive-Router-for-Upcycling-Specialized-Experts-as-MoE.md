# 3 Adaptive Router for Upcycling Specialized Experts as MoE

The core component of an MoE model is the router, as it determines which experts to activate for any given input. In vanilla MoEs, the router is a learned linear layer that takes the token intermediate representations as input and computes the expert probabilities. However, this router does not

```
def router(self, inputs, domain_embeddings):
      # domain_to_expert_ffn learns projection domain to expert embeddings
      # domain_embeddings: [e_dim x n_experts]
      # expert_embeddings: [h_dim x n_experts]
      expert_embeddings = self.domain_to_expert_ffn(self.domain_embeddings)
      # router probs: [batch, seq, n_experts]
      router_probs = nn.softmax(inputs @ expert_embeddings)
      # Top-1 gate for routed experts
11
      index, gate = nn.topk(1, router_probs)
12
      # routed_experts_ffns: An MoE layer with FFN experts
13
      # routed_expert_out: [batch, seq, h_dim]
14
      # shared_expert_out: [batch, seq, h_dim]
      routed_expert_out = self.routed_expert_ffns[index](input)
      shared_expert_out = self.shared_expert_ffn(input)
17
18
      return shared_expert_out + gate * routed_expert_out
```

Figure 2: Router layer in Nexus: PyTorch-like pseudo-code illustrating a router layer, which consists of a 2-layer MLP network (domain\_to\_expert\_ffn) to project domain embeddings to expert embeddings, shared and routed expert FFNs, and sparse Top-k gating. Note that the expert embeddings are independent of the input and could be precomputed once and stored during inference.

necessarily learn specialization as MoEs are commonly trained using an auxiliary load balancing loss to improve training stability [Fedus et al., 2022; Jiang et al., 2024]. In Nexus, we propose a novel MoE router where per MoE block we learn a projection layer from given pre-computed domain embeddings to expert embeddings. We parametrize this projection layer  $P_r$  as a two-layer MLP with a SwiGLU activation function [Shazeer, 2020]:

$$e_i = P_r(d_i)$$
 (Domain to Expert Embeddings)  
=  $W_2 \cdot \text{SwiGLU}(W_1 \cdot d_i)$ 

where  $d_i \in \mathbb{R}^m$ , and  $e_i \in \mathbb{R}^h$  are the domain and expert embeddings for the *i*th domain respectively., where m and h are the domain embedding and the model dimensions.  $W_1 \in \mathbb{R}^{2h \times d}$ ,  $W_2 \in \mathbb{R}^{l \times l}$  are linear layers, and SwiGLU is defined as  $\mathbb{R}^{2n} \to \mathbb{R}^n$ . Given the expert embeddings  $e_i$  and layer inputs  $x \in \mathbb{R}^{s \times h}$ , we then compute routing probabilities  $s_i$  as:

$$s_i = \operatorname{softmax}(x \cdot e_i)$$
 (Routing Scores)

Unlike the standard router, Nexus's router includes a stronger inductive bias through pre-computed domain embeddings<sup>1</sup> that enables expert embedding to specialize. Thus,  $x \cdot e_i$  gives a high value

<span id="page-4-0"></span><sup>&</sup>lt;sup>1</sup>We used an Cohere Embed v3 (https://cohere.com/blog/introducing-embed-v3) as an external embedding model to compute domain embeddings based on existing individual data sources. However, similar to Gururangan et al. [2023], pre-training data can also be clustered and the centroid of each cluster can be used for domain embeddings.

for input tokens that are closer to the domain of the corresponding expert. Notably, this router is particularly suited for the sparse upcycling setting where the dense experts are separately trained on different domains.

Connection to hypernetworks. Our router parametrization is closely related to hypernetworks [\[Ha](#page-17-7) [et al.,](#page-17-7) [2016\]](#page-17-7) as the projection layer P<sup>r</sup> generates parameters for the router during runtime for a given input. We use domain embeddings as the input to the projection layer, enabling efficient adaptation and also a better cross-domain transfer based on the similarity between domain embeddings as shown in previous work [\[Mahabadi et al.,](#page-19-3) [2021;](#page-19-3) [Üstün et al.,](#page-21-6) [2022\]](#page-21-6).

Upcycling dense experts as an MoE. After training dense expert models, we merge the individual experts into a unified MoE by appending their FFNs along a new dimension to create an MoE layer per Transformer block. Unlike [Sukhbaatar et al.](#page-21-2) [\[2024\]](#page-21-2), instead of using the original FFN of the seed model as one of the routed experts in an MoE layer, we use it as the shared expert (FFNs) to better preserve the previous capabilities in the MoE model. For all non-FFN parameters including the attention weights, we merge expert parameters using simple weight averaging:

<span id="page-5-0"></span>
$$\text{FFN}_{moe} = \text{FFN}_s + [\text{FFN}e_1, \text{FFN}e_2, ..., \text{FFN}e_n] \qquad \text{(MoE Layer FFNs)}$$
 
$$\phi_{moe} = \frac{\sum_{i=1}^n \phi_i}{n} \qquad \text{(Merge Non-FFN params.)}$$

Efficient adaptation to new domains. An important advantage of method is that when a new data domain is present after MoE training, we use the learned projection P<sup>r</sup> to compute expert embedding of the new domain as enew = Pr(dnew). This enables to enhance the trained MoE model with additional dense experts, which are trained in the same way as the initial experts. The FFN parameters of the new expert are simply appended to the array of existing experts.

To adequately preserve the non-FFN parameters of existing experts, we perform a weighted average ϕ<sup>f</sup> = (1 − λ)· ϕmoe + λ · ϕnew where ϕ<sup>f</sup> , ϕe, and ϕmoe are parameters of the final MoE, dense expert, and initial MoE model and λ = 1/(n + 1). This enables efficient adaptation Nexus to new domain by extending it with the new dense expert trained independently. After extending the MoE with a new expert, we perform a lightweight finetuning with a limited number of tokens for quick adaptation.

