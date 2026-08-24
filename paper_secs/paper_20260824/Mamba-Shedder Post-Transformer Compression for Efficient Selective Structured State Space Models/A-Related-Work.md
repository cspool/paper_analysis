# A Related Work

Transformers [\(Vaswani et al.,](#page-11-0) [2017\)](#page-11-0) and its variants are the primary building block of successful deep learning architectures, e.g., Llama [\(Touvron](#page-11-12) [et al.,](#page-11-12) [2023\)](#page-11-12) and GPT [\(Brown et al.,](#page-9-5) [2020\)](#page-9-5), that have revolutionized Natural Language Processing (NLP) [\(Devlin et al.,](#page-10-12) [2019;](#page-10-12) [Gao et al.,](#page-10-10) [2023\)](#page-10-10), Computer Vision (CV) [\(Parmar et al.,](#page-11-1) [2018;](#page-11-1) [Radford](#page-11-13) [et al.,](#page-11-13) [2021;](#page-11-13) [Zhang et al.,](#page-11-14) [2023\)](#page-11-14), and many other domains. Due to the Transformer's popularity, researchers have proposed variants to improve their computational and memory efficiency further and tackle issues like their quadratic complexity in sequence length during training [\(Correia et al.,](#page-10-13) [2019;](#page-10-13) [Beltagy et al.,](#page-9-6) [2020;](#page-9-6) [Dai et al.,](#page-10-14) [2020;](#page-10-14) [Choroman](#page-10-15)[ski et al.,](#page-10-15) [2021;](#page-10-15) [Katharopoulos et al.,](#page-11-15) [2020;](#page-11-15) [Zheng](#page-11-16) [et al.,](#page-11-16) [2022\)](#page-11-16).

A parallel research effort investigates alternatives to Transformers in the form of *structured state space models* (SSMs) that can power the next generation of sequence models. The initial proposals of structured SSMs were linear time-invariant, e.g., LSSL [\(Gu et al.,](#page-10-16) [2024\)](#page-10-16), S4 [\(Gu et al.,](#page-10-2) [2022\)](#page-10-2), H3 [\(Fu et al.,](#page-10-17) [2023\)](#page-10-17). Recent improvements to the state space model formulation have resulted in the proposal of time-varying selective SSMs, e.g., Mamba [\(Gu and Dao,](#page-10-3) [2023;](#page-10-3) [Dao and Gu,](#page-10-4) [2024\)](#page-10-4).

To our knowledge, Mamba-Shedder is the first study on pruning selective structured state space models (Mamba) and their hybrids. On the other hand, many works have proposed pruning techniques for Transformer-based models [\(Hoefler](#page-10-18) [et al.,](#page-10-18) [2021\)](#page-10-18). Several of these works focus on *unstructured* pruning [\(Sun et al.,](#page-11-17) [2023;](#page-11-17) [Xu et al.,](#page-11-18) [2024;](#page-11-18) [Frantar et al.,](#page-10-19) [2022\)](#page-10-19), which can achieve higher sparsity levels. However, it requires highly optimized runtimes to realize the benefits of sparsity. Sophisticated solutions have been proposed to fine-tune sparse models and recover any accuracy drop from the pruning stage [\(Muñoz et al.,](#page-11-19) [2024\)](#page-11-19). Recently, *training-free* approaches have been proposed for *structured* pruning of Transformers. These approaches cannot achieve high sparsity levels as the *unstructured* pruning approaches. However, they are very convenient because their compressed models do not require specialized runtimes and exhibit beneficial inference acceleration. In this line of research, LLMPruner [\(Ma et al.,](#page-11-11) [2023\)](#page-11-11), ShortGPT [\(Men et al.,](#page-11-3) [2024\)](#page-11-3), BlockPruner [\(Lagunas et al.,](#page-11-20) [2021\)](#page-11-20), SliceGPT [\(Ashkboos et al.,](#page-9-1)

<span id="page-13-0"></span>

| Hyper-parameter                       | Value            |
|---------------------------------------|------------------|
| Pruning Stage:                        |                  |
| Calibration Dataset                   | tatsu-lab/alpaca |
| Importance Metric                     | Perplexity (PPL) |
| Number of Calibration Samples         | 256              |
| MLP Channel Group Size (Zamba2)       | 1024             |
| Steps of MLP Channel Pruning (Zamba2) | 20               |

Table 13: Hyper-parameters used in the experiments.

[2024\)](#page-9-1), and MultiPruner [\(Muñoz et al.,](#page-11-5) [2025\)](#page-11-5) have demonstrated efficient methods for Transformer pruning. BlockPruner improved over many previous approaches by proposing a global metric that can be used to determine the importance of a selected network structure. MultiPruner extended this approach to pruning the width dimension, as well. Mamba-Shedder builds on these works and the rest of the extensive literature on *structured* block pruning to explore opportunities for removing redundancies in models with Mamba blocks.

