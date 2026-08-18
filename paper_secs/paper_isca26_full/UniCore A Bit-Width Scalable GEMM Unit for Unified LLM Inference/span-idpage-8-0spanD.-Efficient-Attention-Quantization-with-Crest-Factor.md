# <span id="page-8-0"></span>*D. Efficient Attention Quantization with Crest Factor*

Unlike offline weight-format search, K/V activations are quantized online during inference and require a lightweight method that avoids evaluating all DynFP candidates per group. To achieve this, UNICORE uses the *crest factor* κ [\[5\]](#page-14-21) as a lightweight proxy for a group's dynamic format selection. Different DynFP formats exhibit distinct quantization signalto-noise ratio (QSNR) [\[9\]](#page-14-31) behaviors as functions of κ, allowing us to precompute a small set of thresholds that map κ to the most suitable E/M layout.

During inference, κ is computed in a single streaming pass using one max-abs reduction and one RMS reduction, incurring only four scalar operations per element [\[43\]](#page-15-6). The overhead is negligible: for typical sequence lengths (L ≥ 2K), the crest-factor computation contributes less than 0.2% of the FLOPs of QK<sup>⊤</sup> and remains fully memory-bound, enabling seamless fusion into the quantization kernel. This crest-factorguided selection provides accurate, distribution-aware DynFP quantization for K/V activations with negligible runtime cost.

