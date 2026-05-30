# <span id="page-11-0"></span>6.3 Model Convergence

We evaluate model convergence with MegaScale-MoE. Figure 19 demonstrates the loss curves of training a 35B MoE model from scratch and continuing training a 176B MoE model from a checkpoint, with results shown for both BF16 and FP8 precision. MegaScale-MoE ensures stable convergence and consistent training loss across BF16 and FP8 formats.

