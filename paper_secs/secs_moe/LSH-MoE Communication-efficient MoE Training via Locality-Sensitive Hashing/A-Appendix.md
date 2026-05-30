# A Appendix

#### <span id="page-13-1"></span>A.1 Framework of LSH-MoE

As illustrated in Algorithm 1, our LSH-MoE training process begins by dispatching each input token in *X* to its designated expert based on the gating network (Line 2-3). It then utilizes locality-sensitive hashing to cluster tokens into groups, calculating the centroid for each cluster to represent the mean of its tokens, and recording the differences between each token and its centroid for later error compensation (Lines 4-11). These centroids are subsequently transmitted to the experts via all-to-all communication for processing and their results are sent back in a similarly manner (Line 13-15). Finally, a residual-based error compensation is applied to determine the output of the MoE layer (Lines 16-18). This method effectively minimizes the communication load, thus improving the scalability and efficiency of MoE model training.

## <span id="page-13-2"></span>A.2 Scalability Analysis

First of all, we want to highlight that as the scale of both models and machines increases, the proportion of all-to-all communication time relative to the total time remains nearly constant. This consistency suggests that the LSH-MoE method remains effective even at larger scales. We will now present our derivation step by step, using the notations listed in Table 4.

**Formulate all-to-all communication.** For any given training server, the amount of tokens (i.e. m) communicated with any other GPU node can be expressed as  $m = n \times k/w$ . Similarly, the volume of communication within the same GPU node is also equal to  $m = n \times k/w$ . Consequently, the time required for all-to-all communication during model training can be modeled as follows, with each layer involving two instances for the forward pass and two for the backward pass:

$$T_{all\_to\_all} = 4 \times l \times \left(\frac{m \times h}{B_{intra}} + \frac{m \times h \times (w - 1)}{B_{inter}}\right) \approx 4l \times \frac{nk}{w} \times \frac{h(w - 1)}{B_{inter}}.$$
 (7)

**Formulate model computation.** Based on the derivation in [27], for a standard decoder model, given the number of layers l, and the hidden size h of the model, the activated parameter count per token

Table 4: Notations used in scalability analysis.

<span id="page-14-0"></span>

| Notation      | Description                                                    |  |  |  |
|---------------|----------------------------------------------------------------|--|--|--|
| $\overline{}$ | The number of tokens processed per GPU                         |  |  |  |
| m             | The number of tokens communicated between two training servers |  |  |  |
| k             | The number of experts activated per token                      |  |  |  |
| h             | Hidden size for each token                                     |  |  |  |
| l             | The number of layers for the model                             |  |  |  |
| w             | The number of training servers                                 |  |  |  |
| $B_{intra}$   | The intra-machine network bandwidth                            |  |  |  |
| $B_{inter}$   | The inter-machine network bandwidth                            |  |  |  |

can be formalized as #ActivatedParams. =  $4(1+2k)lh^2$ . According to the theory in the appendix of the GPT-3 paper [3], the computation time per GPU can be formalized as  $T_{compute}$ , where FLOPs represents the computation ability of GPU.

$$T_{compute} = 6 \times \text{\#tokens} \times \frac{\text{\#ActivatedParams.}}{\text{FLOPs}} = \frac{24(1+2k)nlh^2}{\text{FLOPs}}. \tag{8}$$

Formulate all-to-all communication / computation. Therefore, as the machine scale (w) and model scale (l) and (l) increase, the ratio of computation time to communication time can be formalized as:

$$\frac{T_{all\_to\_all}}{T_{compute}} = \frac{4l \times \frac{nk}{w} \times \frac{h(w-1)}{B_{inter}}}{(24(1+2k)nlh^2)/\text{FLOPs}} = \frac{\text{FLOPs}}{6B_{inter}} \times \frac{k}{1+2k} \times \frac{w-1}{wh},\tag{9}$$

where the first term  $\frac{\text{FLOPs}}{6B_{inter}}$  is constant. As MoE models scale up, the emphasis is generally placed on increasing the number of layers and experts, with a more gradual increase in hidden size (e.g. Switch-Transformer [9]). Consequently, the proportion of communication time remains significant as both the model size and the number of servers increase. These observations and theoretical proofs underscore the sustained effectiveness of the LSH-MoE method in larger environments, thus reinforcing its scalability and applicability for future advancements.

