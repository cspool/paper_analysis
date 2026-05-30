# <span id="page-16-0"></span>C. Experiment Results

### C.1. Weight Visualization

In addition to the linear projection for Key mentioned earlier, we also compute the entropy of quantized weights in various other types of layers, as illustrated in Figure [5.](#page-16-2) This observation confirms that ICQ effectively boosts the information entropy of weights and augments the mutual information between the weights of quantized and original LLMs, consequently alleviating the constraints on representational capacity imposed by quantization.

### C.2. Efficiency Ablation

We counted the number of parameters and training time for different sizes of LLaMA, as shown in Table [15.](#page-17-0) It is evident that IEC introduces a minimal number of additional parameters and does not increase the training time. In the case of ICQ, a small number of extra parameters are introduced, and the increase in training time is only incurred once, as the results can be efficiently cached after ICQ is applied. Thus, IR-QLoRA maintains nearly the same efficiency as Vanilla.

