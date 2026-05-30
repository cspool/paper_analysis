# <span id="page-3-1"></span>5.1. Globally-Modulated Local Gating

In light of *Insight* (i) in Sec. 4.1, we present a globallymodulated local gating (GMLG) mechanism, which combines the global contributions of experts with local routing behaviors to estimate expert importance for given tokens. During inference, experts in  $S^{(l)}$  (Eq. (2)) with importance scores lower than the thresholds (defined in Sec. 5.2) will be skipped. Specifically, for  $\mathtt{Expert}_i^{(l)}$   $(i \in \mathcal{S}^{(l)})$  with an input token  $\mathbf{x}^{(l)}$ , the importance score is defined as:

<span id="page-3-4"></span>
$$s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)},$$
 (3)

where  $\pi_i^{(l)}$  is the local routing probability (Eq. (1)) that  $\mathsf{Expert}_i^{(l)}$  will be activated for  $\mathbf{x}^{(l)}$ . The globallymodulated factor  $\alpha^{(l)}$  reflects the impact of experts in the layer on the final prediction, which is obtained by offline calibration. This  $s_i^{(l)}$  accounts for both global and local contributions, yielding an accurate importance estimation.

To obtain  $\alpha^{(l)}$ , we calculate the Kullback-Leibler (KL) divergence between the output distribution of the original model and that of a counterpart where experts in the l-th layer are skipped:

<span id="page-3-2"></span>
$$\alpha^{(l)} = \frac{1}{N} \sum_{j=1}^{N} \mathcal{D}_{KL} \left( \operatorname{prob}_{j} || \operatorname{prob}_{j}^{(l)} \right), \tag{4}$$

where N is the size of data (i.e.,  $C = \{c_1, \dots, c_N\}$ ) used for

this calibration.  $prob_i$  and  $prob_i^{(l)}$  are the output probabilities for the j-th example of C from the original and modified models, respectively. This process quantifies the sensitivity of the model's output to the removal of experts in certain layers, and  $\alpha^{(l)}$  serves as a global importance weight reflecting their relative contributions. With the pre-computed  $\alpha^{(l)}$ , the final importance score  $s_i^{(l)}$  can be obtained without additional overhead during inference.

