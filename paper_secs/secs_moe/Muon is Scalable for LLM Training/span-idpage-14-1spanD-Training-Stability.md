# <span id="page-14-1"></span>D Training Stability

**No Loss or Grad Norm Spike** The Moonlight training process was very smooth and we did not meet any loss spike or gradient norm spike. The loss and grad norm curve can be seen in Figure 7 (Moonlight is colored in blue and Moonlight-A trained by AdamW is colored in red)

Max Attention Logit During training, we observed that while both the training loss and gradient norm remained stable throughout the process, the maximum attention logit (computed as the single largest logit value across the global

```
1 import numpy as np
3 def sigmoid ( x ):
4 return 1 / (1 + np . exp ( - x ))
6 def calc_gate_scaling_factor ( num_experts : int , topk : int , iter_times : int ):
7 """ Calculate the gate scaling factor for MoE.
8
9 Args :
10 num_experts (int ): The number of experts .
11 topk (int ): The number of experts to select .
12 iter_timers (int ): The number of iterations .
14 Returns :
15 float : The gate scaling factor .
16 """
17 factors = []
18 for _ in range ( iter_times ):
19
20 # mock gaussian logits
21 logits = np . random . randn ( num_experts )
22 # select topk logits
23 p = np . sort ( sigmoid ( logits ))[:: -1]
24 p = p [: topk ]
25 # renormalize
26 p = p / p .sum ()
27 # calculate the scaling factor
28 factors . append ( 1/ (p **2). sum ()**0.5)
29 return np . mean ( factors )
```

Figure 6: Python implementation for calculating the gate scaling factor.

batch) exhibited a distinct upward trajectory in specific layers during the initial training phase, exceeding a threshold of 100. Notably, AdamW demonstrated healthier behavior in controlling this metric compared to alternative optimizers.

To further investigate the impacts of this phenomenon, we introduced the large attention logits ratio metric, defined as the proportion of attention logits exceeding 100 within a batch. As shown in Fig[.7,](#page-16-0) this ratio remained consistently low (about 10<sup>−</sup><sup>4</sup> ), indicating that extreme large logit values were sparse. Furthermore, the maximum logit values gradually decrease as training progressed, suggesting that the optimization dynamics become healthier.

RMSNorm Gamma Weight Decay It is noteworthy that applying weight decay to the RMSNorm gamma parameter is crucial for ensuring training stability, as it effectively prevents excessively high output RMS values in each layer.

