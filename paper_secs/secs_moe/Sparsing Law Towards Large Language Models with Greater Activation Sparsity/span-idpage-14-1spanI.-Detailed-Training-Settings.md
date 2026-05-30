# <span id="page-14-1"></span>I. Detailed Training Settings

We utilize the  $\mu P$  Transformer (Hu et al., 2024) architecture and adopt its hyper-parameter policies, along with the WSD learning rate scheduling method. Across all parameter scales, the ratio of  $d_f$  to  $d_h$  is equal to 2.5 consistently, the number of query heads always matches that of key and value heads, and the width-depth ratios range from 48 to 56, generally similar across different scales. The specific number of parameters of various settings are shown in Table 3. We employ the following pre-training hyper-parameters across all

settings: peak learning rate lr = 0.01, β<sup>1</sup> = 0.9, β<sup>2</sup> = 0.95, weight decay = 0.1. The batch size depends on the parameter scale, as presented in Table [3.](#page-14-4)

