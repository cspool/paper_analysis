# D Experimental Details

### D.1 Model Architecture

Our model uses a standard Diffusion Transformer architecture [\[50\]](#page-11-18). We also incorporate popular general-purpose improvements, including SwiGLU [\[61\]](#page-12-15), RMSNorm [\[80\]](#page-13-8), RoPE [\[67\]](#page-12-16), and qk-norm [\[24\]](#page-10-18). We use in-context conditioning instead of adaLN-Zero [\[50\]](#page-11-18) conditioning, which allows us to significantly reduce the number of parameters; for example, the ELF-B model size is reduced from 148M to 105M parameters. Tab. [3](#page-23-0) summarizes the configurations of ELF across different model sizes. We report the Transformer depth, hidden size, number of attention heads, and parameter count. We also report the number of training epochs used on the OWT dataset for each variant. Larger models tend to learn faster in our setup, and therefore require fewer training epochs.

