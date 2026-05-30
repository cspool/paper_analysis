# <span id="page-16-0"></span>A More Preliminary

In this section, we introduce some key definitions of language modeling modules. We begin with the input embedding function and the output embedding function. They are functions that bridge between the input token space and the real vector space.

**Definition A.1** (Input embedding function and input tokens). The input embedding function  $\mathcal{E}$ :  $\mathcal{V}^n \to \mathbb{R}^{n \times d}$  maps the input tokens to hidden features using the vocabulary dictionary  $D^{\text{voc}} \in \mathbb{R}^{|\mathcal{V}| \times d}$ . Let  $T \in \mathcal{V}^n$  be input tokens. Then, we have  $\mathcal{E}(T) \in \mathbb{R}^{n \times d}$  and  $\mathcal{E}(T)_i = D_{T_i}^{\text{voc}} \in \mathbb{R}^d$  for any  $i \in [n]$ .

**Definition A.2** (Output embedding function). The output embedding function  $\mathcal{G}: \mathbb{R}^d \to \mathbb{R}^{|\mathcal{V}|}$  maps hidden features to the probability logits of the vocabulary dictionary.

We introduce Softmax, which allows self-attention to learn the probability distribution rather than function anymore.

**Definition A.3** (Softmax). Let  $z \in \mathbb{R}^n$ . We define Softmax:  $\mathbb{R}^n \to \mathbb{R}^n$  satisfying

$$\mathsf{Softmax}(z) := \exp(z)/\langle \exp(z), \mathbf{1}_n \rangle.$$

