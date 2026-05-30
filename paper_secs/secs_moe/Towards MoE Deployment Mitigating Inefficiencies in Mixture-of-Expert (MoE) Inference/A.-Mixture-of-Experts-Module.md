# *A. Mixture-of-Experts Module*

Using different models for different inputs has long been discussed as a way to improve model versatility and robustness. Mixture-of-Experts (MoE) module [\[29\]](#page-11-11) is a practical application of this idea for neural networks. An MoE module (Figure [3\)](#page-2-0) consists of multiple independent models (called experts), and a gating function that assigns inputs to each of the experts. Each input only activates its assigned expert network, which theoretically allows the model capacity (*i.e.*, the number of parameters in the model) to expand "outrageously" with minimal computation efficiency loss.

