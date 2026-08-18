# <span id="page-1-0"></span>II. INFORMATION REDUNDANCY IN CONTEMPORARY LLM WEIGHT

Large language models consist of many transformer layers, each layer performing several projections with highdimensional matrix multiplications between the weight matrices and the hidden-state activations. These projection weights dominate the static memory footprint of the model and must be frequently accessed from device memory during inference.

#### *A. Heavy-Tailed Distributions in LLM Weights*

LLM weights exhibit heavy-tailed statistical distributions [\[33\]](#page-13-17) and, importantly, the distribution can vary substantially across layers. Most existing studies evaluate compression [\[16\]](#page-13-8), [\[20\]](#page-13-9), [\[26\]](#page-13-10), [\[31\]](#page-13-11), [\[39\]](#page-14-2), [\[47\]](#page-14-3) and quantization [\[10\]](#page-13-12), [\[15\]](#page-13-13), [\[18\]](#page-13-14), [\[25\]](#page-13-15), [\[29\]](#page-13-16), [\[29\]](#page-13-16), [\[33\]](#page-13-17), [\[36\]](#page-14-4), [\[44\]](#page-14-5) primarily from a numerical-format perspective, emphasizing bit width, scaling, and calibration strategies. However, such approaches do not explicitly capture the underlying data distribution or its crosslayer variability, leaving significant opportunities for further redundancy reduction unaddressed.

To evaluate the potential for further reducing the memory footprint of model weights, we begin with an information-theoretic analysis based on Shannon's source coding theorem. In information theory, the Shannon limit defines the theoretical lower bound on the average number of bits required to represent data without loss. This bound is determined by the Shannon entropy of the source distribution, which quantifies the intrinsic information content of the weight values.

# <span id="page-1-0"></span>II. INFORMATION REDUNDANCY IN CONTEMPORARY LLM WEIGHT

Large language models consist of many transformer layers, each layer performing several projections with highdimensional matrix multiplications between the weight matrices and the hidden-state activations. These projection weights dominate the static memory footprint of the model and must be frequently accessed from device memory during inference.

#### *A. Heavy-Tailed Distributions in LLM Weights*

LLM weights exhibit heavy-tailed statistical distributions [\[33\]](#page-13-17) and, importantly, the distribution can vary substantially across layers. Most existing studies evaluate compression [\[16\]](#page-13-8), [\[20\]](#page-13-9), [\[26\]](#page-13-10), [\[31\]](#page-13-11), [\[39\]](#page-14-2), [\[47\]](#page-14-3) and quantization [\[10\]](#page-13-12), [\[15\]](#page-13-13), [\[18\]](#page-13-14), [\[25\]](#page-13-15), [\[29\]](#page-13-16), [\[29\]](#page-13-16), [\[33\]](#page-13-17), [\[36\]](#page-14-4), [\[44\]](#page-14-5) primarily from a numerical-format perspective, emphasizing bit width, scaling, and calibration strategies. However, such approaches do not explicitly capture the underlying data distribution or its crosslayer variability, leaving significant opportunities for further redundancy reduction unaddressed.

To evaluate the potential for further reducing the memory footprint of model weights, we begin with an information-theoretic analysis based on Shannon's source coding theorem. In information theory, the Shannon limit defines the theoretical lower bound on the average number of bits required to represent data without loss. This bound is determined by the Shannon entropy of the source distribution, which quantifies the intrinsic information content of the weight values.

