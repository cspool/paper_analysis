# <span id="page-16-2"></span><span id="page-16-0"></span>Algorithm 1 QuantSparse: Calibration to Inference Pipeline

```
Require: Pre-trained video diffusion transformer M (FP16), calibration dataset \mathcal{D}_{cal}, target bit-
     width (W/A), denoising steps T, cache interval \tau
Ensure: Quantized-sparse model M_{OS}, generated video Y
 1: Calibration Phase:
        Initialize quantization parameters \{s, z\} for weights (W) and activations (A)
 2:
 3:
        Input X \in \mathcal{D}_{cal} to M
 4:
        Compute token saliency s_i using Eq. 7 for FP model M
        Select top-k salient tokens I = \{j \mid s_j \text{ is top-}k\}
 5:
        Global Guidance Distillation:
 6:
 7:
           Calculate \mathcal{L}_{global} using Eq. 6
 8:
        Local Guidance Distillation:
 9:
            Calculate \mathcal{L}_{local} using Eq. 8
10:
        Optimize quantization parameters using Eq. 9 with \mathcal{L}_{global} and \mathcal{L}_{local}
11:
        Obtain quantized model M_{\text{quant}} with optimized \{s, z\}
12: Inference Phase:
13:
        Load M_{\text{quant}} and input prompt P.
        Input P into M_{\rm quant} and initialize cached residuals \{\Delta^{(t_{\rm ref})}_{\rm quant},\hat{\Delta}^{(t_{\rm ref})}_{\rm quant}\}
14:
15:
        for t in T
           Compute quantized sparse attention:
16:
                                   A_{s,q}^{(t)} = \text{SparseAttention}(Q_{\text{quant}}, K_{\text{quant}}, V_{\text{quant}}; M)
           if t - t_{\text{ref}} \leq \tau
17:
               Reuse cached residuals: \Delta_{\mathrm{curr}} = \Delta_{\mathrm{quant}}^{(t_{\mathrm{ref}})} + \hat{\Delta}_{\mathrm{quant}}^{(t_{\mathrm{ref}})}
18:
19:
20:
               Update t_{ref} = t, recompute and cache residuals
21:
22:
            Refine attention using Eq. 16
23:
        endfor
```

