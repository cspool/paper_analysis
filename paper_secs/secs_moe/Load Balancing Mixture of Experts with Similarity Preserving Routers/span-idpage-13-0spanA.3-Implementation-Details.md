# <span id="page-13-0"></span>A.3 Implementation Details

Here we provide some implementation details related to the auxiliary losses used in the paper in Figure [6.](#page-14-2) For our LBL baseline, we use an open-source repository implementation based on [Zoph](#page-12-9)

| Router         | SimBal      | LBL        |
|----------------|-------------|------------|
| Layer 0 Router | 1.94017e-10 | 0.00146701 |
| Layer 1 Router | 1.70156e-10 | 0.01486    |
| Layer 2 Router | 1.91267e-10 | 0.0155954  |
| Layer 3 Router | 1.89254e-10 | 0.0102319  |
| Layer 4 Router | 1.50925e-08 | 0.0100937  |
| Layer 5 Router | 2.99727e-08 | 0.0143029  |
| Layer 6 Router | 1.82301e-10 | 0.020765   |
| Layer 7 Router | 1.73648e-10 | 0.0258847  |

Table 9: Router orthogonality of MoE-M, as measured by pR<sup>T</sup> R ´ Iq 2

<span id="page-14-0"></span>

| Router          | SimBal      | LBL        |
|-----------------|-------------|------------|
| Layer 0 Router  | 1.49951e-08 | 0.0125956  |
| Layer 1 Router  | 1.00854e-10 | 0.027788   |
| Layer 2 Router  | 1.03228e-10 | 0.0183506  |
| Layer 3 Router  | 4.47955e-08 | 0.0128958  |
| Layer 4 Router  | 1.5001e-08  | 0.00668315 |
| Layer 5 Router  | 9.38376e-11 | 0.00399825 |
| Layer 6 Router  | 1.16159e-10 | 0.00375414 |
| Layer 7 Router  | 2.99078e-08 | 0.00736187 |
| Layer 8 Router  | 4.47949e-08 | 0.0200508  |
| Layer 9 Router  | 2.99088e-08 | 0.0377724  |
| Layer 10 Router | 5.97087e-08 | 0.083971   |
| Layer 11 Router | 1.49907e-08 | 0.138501   |

<span id="page-14-1"></span>Table 10: Router orthogonality of MoE-L, as measured by pR<sup>T</sup> R ´ Iq 2

[et al.](#page-12-9) [\[2022\]](#page-12-9), available at [lucidrains/st-moe-pytorch](https://github.com/lucidrains/st-moe-pytorch) on GitHub. For both, we multiply the output of the function by the scaling coefficient if/where applicable during training. These losses can then be added to the final model loss (by adding them), or included using the AddAuxiliaryLoss autograd trick used in [DeepSeek's modeling\\_deepseek.py on HuggingFace.](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py)

```
1 import torch
2 from einops import reduce
4 # LBL
5 def balance_loss ( gates : torch . Tensor ) -> torch . Tensor :
6 batch_size , num_tokens , num_experts = gates . shape
8 # bal_loss = E * sum (f_i * P_i), expert i
9 expert_mask = gates > 0.0
10 f_i = reduce ( expert_mask . float () , "b t e -> b e", " mean ")
11 P_i = reduce ( gates , "b t e -> b e", " mean ")
12 loss_per_batch = num_experts * torch .sum( f_i * P_i , dim = -1)
13 return loss_per_batch . mean ()
15 # SimBal
16 def simbal_loss ( router_linear , p =1) :
17 w = router_linear . weight
18 w_ortho = torch . matmul (w , w . T )
19 eye = torch . eye ( w . shape [0] , device = w . device )
20 loss = torch . norm ( w_ortho - eye , p = p )
21 return loss
```

<span id="page-14-2"></span>Figure 6: Python implementations of the LBL and SimBal loss functions.