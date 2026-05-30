# <span id="page-13-1"></span>A. xLSTM 7B Architecture Summary

<span id="page-13-2"></span>The xLSTM 7B architecture consists of 32 post-up projection blocks and is described in Fig. 1 and Tab. 4. We use the GPT-NeoX-20B tokenizer (Black et al., 2022) with vocabulary size 50257 and do not tie the weights for input layers (embedding) and output layers (logits).

Table 4. Hyperparameters of xLSTM 7B.

| NUM           | VOCAB | NUM    | Model | NUM   |
|---------------|-------|--------|-------|-------|
| PARAMS        | SIZE  | BLOCKS | Dim   | HEADS |
| 6,865,424,896 | 50257 | 32     | 4096  | 8     |

<span id="page-13-0"></span>![](_page_13_Picture_5.jpeg)

Figure 8. Improved xLSTM Block. The lower part is a output-gated sequence-mix layer with the mLSTM at its core, whereas the upper part is a Gated MLP (SwiGLU) as a feature/channel-mix layer. Multiple Heads are shown in depth, larger light gray boxes without are linear layers. For the SwiGLU we use a projection factor of 2.66 matching common Transformers. For the query/key dimension we use a factor of 0.5. The Norm layers are RMS norms (Zhang & Sennrich, 2019), the Headwise Norm is a Layernorm (Ba et al., 2016).

