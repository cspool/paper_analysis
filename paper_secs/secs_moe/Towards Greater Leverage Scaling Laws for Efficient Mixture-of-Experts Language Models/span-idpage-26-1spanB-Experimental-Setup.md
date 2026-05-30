# <span id="page-26-1"></span>**B Experimental Setup**

Our experiments primarily follow the architecture and training configurations of the Ling series models [\(Ling-Team et al.,](#page-24-14) [2025\)](#page-24-14).

*Architecture and Tokenizer* We adopt a Grouped Query Attention (GQA) [\(Ainslie et al.,](#page-22-2) [2023\)](#page-22-2) architecture based on the standard decoder-only Transformer, consisting of an embedding layer, multiple alternating layers of attention mechanisms and feed-forward networks, and a final deembedding layer. Additionally, we use the BPE (Byte-Pair Encoding) algorithm [\(Sennrich et al.,](#page-25-16) [2015\)](#page-25-16) and RoPE (Rotary Positional Embedding) [\(Su et al.,](#page-25-3) [2024\)](#page-25-3) to handle positional information. The vocabulary size is 126,464, and the sequence length is 4,096.

*Expert Routing Strategy* In our MoE layers, a routing network assigns each token's hidden state *h<sup>t</sup>* to the top-*N<sup>a</sup>* experts. This is achieved by generating gating scores *g<sup>t</sup>* = Softmax(*W<sup>g</sup>* · *ht*), where *W<sup>g</sup>* is a learnable matrix. The final output is a weighted sum of the selected experts' outputs: *o<sup>t</sup>* = ∑*i*∈TopK(*gt*) *gt*,*<sup>i</sup>* · *Ei*(*ht*), where *E<sup>i</sup>* is the *i*-th expert in total *N* experts. To ensure balanced expert utilization and stable training, we incorporate two standard auxiliary losses: a load balancing loss [\(Lepikhin et al.,](#page-24-13) [2020\)](#page-24-13) (coefficient of 0.01) to encourage uniform token distribution, and a router z-loss [\(Zoph et al.,](#page-26-7) [2022\)](#page-26-7) (coefficient of 0.001) to regularize the magnitude of the gating logits.

*Optimizer and Scheduler* The parameters of experimental models are initialized from a distribution with a standard deviation of 0.006 and optimized using the AdamW optimizer [\(Loshchilov and](#page-24-4) [Hutter,](#page-24-4) [2017\)](#page-24-4). The optimizer's hyperparameters are set to *β*<sup>1</sup> = 0.9 and *β*<sup>2</sup> = 0.95, with 0.1 weight decay applied. The learning rate schedule employs a WSD (warmup-stable-decay) strategy [\(Hu](#page-23-4) [et al.,](#page-23-4) [2024\)](#page-23-4): the first 1% of training steps use linear warm-up, followed by exponential decay that reduces the learning rate to 10% of its peak value.

*Pre-training Data* The training data is sourced from a large-scale multilingual corpus created by the Ling Team, primarily covering English and Chinese, while also including various other languages. This corpus encompasses web text, mathematical materials, programming scripts, published literature, and diverse textual content. To validate model performance, we extracted a 2T-token subset from this corpus for training. In Table [6,](#page-27-1) we present the composition of the training datasets for all experiments. Unless otherwise specified, this configuration is used throughout.

Table 6 Pre-training data composition.

<span id="page-27-1"></span>

| Type  | Web   | Books | Wiki | Academic | Code  | News | Social | Domain | SFT  | Math | Exam |
|-------|-------|-------|------|----------|-------|------|--------|--------|------|------|------|
| Ratio | 46.0% | 5.0%  | 4.0% | 6.0%     | 25.0% | 0.1% | 1.9%   | 1.0%   | 4.0% | 6.0% | 1.0% |

