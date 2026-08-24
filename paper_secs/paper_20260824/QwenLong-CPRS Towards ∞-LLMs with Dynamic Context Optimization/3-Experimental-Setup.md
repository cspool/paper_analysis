# 3 Experimental Setup

### 3.1 Implementation Details

QWENLONG-CPRS was initialized using the Qwen2-7b-Base architecture [\[45\]](#page-15-2), inheriting its parameters and vocabulary. The first 21 Transformer layers were retained as causal attention modules, while layers 22–28 were reconfigured as bi-directional location reasoning layers following the design in Section [2.2.](#page-3-0) A 3-epoch supervised fine-tuning regimen was implemented for the token critic task, employing the following configurations: window-parallel inference with 8192-token context windows, global batch size of 256, and constant learning rate of 1e-5. Training stability was ensured through Zero-3 partitioning with optimizer state offloading [\[35\]](#page-14-12). To address input token optimization sparsity in long-context processing, we applied random gradient masking to 50% of non-critical token positions during backpropagation.

