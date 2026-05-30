# A. MoE Layer

Generally, an MoE model follows the architecture of a dense model by replacing its dense feed-forward layers with MoE layers as shown in Fig. 2a. An MoE layer consists of two primary elements: a gating function and a collection of trainable feed-forward neural networks (FFNs) called experts. At each training iteration, the input tokens are distributed to selected experts according to the gating function. The gating function also uses a small trainable FFN followed by a softmax layer and top-K selection to determine which experts should process which tokens.

