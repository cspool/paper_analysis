# <span id="page-30-0"></span>A Related Work

The fundamental challenge of transforming raw sequential data into computationally efficient representations manifests across multiple domains through implicit chunking processes. In language modeling, this challenge is addressed through tokenization using static vocabularies derived from frequency-based algorithms such as Byte-Pair Encoding (BPE) (Sennrich, Haddow, and Birch [2015\)](#page-28-0) in GPT models (Radford et al. [2019;](#page-27-7) Brown et al. [2020\)](#page-24-0) and SentencePiece (Kudo and Richardson [2018\)](#page-26-0) in Llama architectures (Touvron, Martin, et al. [2023;](#page-28-10) Grattafiori et al. [2024\)](#page-25-0). Computer vision addresses similar challenges through spatial pooling operations (Ronneberger, Fischer, and Brox [2015\)](#page-27-5) that aggregate neighboring pixels into meaningful representations.

Despite achieving strong empirical performance, it is widely known that traditional tokenization approaches in language models suffer from fundamental limitations that constrain model capabilities. Fixed vocabularies exhibit biases toward highresource languages, demonstrate fragility when handling adversarial inputs, and show lower performance on character-level tasks (Petrov et al. [2023;](#page-27-0) Ahia, Kumar, Gonen, Kasai, et al. [2023;](#page-24-1) Belinkov and Bisk [2017;](#page-24-2) Sun et al. [2020;](#page-28-1) Xue et al. [2022\)](#page-28-2). These limitations stem from the static nature of predefined vocabularies, which cannot adapt their chunking strategies to input content or context.

To address these constraints, tokenizer-free methods have emerged that avoid the reliance on predefined vocabularies.

- In Appendix [A.1,](#page-30-1) we discuss the most directly related prior work on autoregressive sequence models, extending the overview from Section [1.](#page-0-2)
- In Appendix [A.2,](#page-32-0) we discuss non-autoregressive models. We note that essentially all autoregressive architectures can be turned into non-autoregressive architectures (including our proposed H-Net), and vice versa, which provide possible extensions of H-Net in future work. However, we provide this delineation because it marks an important difference in motivation that influences design considerations and downstream evaluations.
- Appendix [A.3](#page-33-0) mentions other works in non-language modalities related to tokenization.

We summarize our discussion on tokenizer-free architectures in Table [6.](#page-20-0)

## <span id="page-30-1"></span>A.1 Autoregressive Tokenizer-free Architectures

As outlined in Section [1,](#page-0-2) prior work on autoregressive tokenizers for architectures can be divided into four categories:

- 1. Non-hierarchical isotropic architectures.
- 2. Hierarchical architectures with static chunking strategies, where chunk boundaries are content-agnostic (usually some variant of fixed-width pooling).
- 3. Hierarchical architectures with external chunking strategeies, where chunk boundaries are provided by an external function or module.
- 4. Hierarchical architectures with dynamic chunking strategies, where chunk boundaries are content-dependent and learned end-to-end.

### A.1.1 Isotropic Architectures

The most direct approach to modeling language with tokenizers is to simply model raw byte sequences with a standard sequence model architecture. Since this naive approach suffers from computational challenges on long sequences, MambaByte (Wang et al. [2024\)](#page-28-4) proposed using a state space model for its linear-time efficiency. We similarly use Mamba(-2) (Dao and Gu [2024\)](#page-24-5) layers in the outer stages of an H-Net. Notably, through extensive ablations we show that Mamba is not just more efficient but also better at modeling high-resolution data such as text characters and DNA base pairs.

### A.1.2 Static Chunking

To reduce sequence length, several approaches downsample the input sequence hierarchically. The most straightforward methods operate independently of input context, partitioning sequences using fixed-size intervals. Many strategies could be used to aggregate a width- window, including direct downsampling, average pooling, linear transformations that mix across the chunk, convolutions, and more; we lump these together as pooling operations.

Hourglass Transformer (Nawrot, Tworkowski, et al. 2022) and MegaByte (Yu, Simig, et al. 2023) exemplify this strategy. Other recent variants include the Block Transformer (Ho et al. 2024) and Multiscale Byte Language Model (MBLM) (Egli, Manica, and Born 2025), which use similar multi-stage static chunking architectures. Concurrently to H-Net, the MBLM also proposes using Mamba layers in the outer stages.

These approaches share conceptual similarity with spatial pooling operations in vision models that reduce resolution through fixed-window aggregation (Krizhevsky, Sutskever, and Hinton 2012; He et al. 2016). While these content-agnostic methods have simple and efficient implementations, they face an inherent limitation: they do not reflect natural semantic boundaries in the data. Fixed-size chunking inevitably creates arbitrary separations that can split meaningful units such as words, morphemes, or phrases, thereby limiting model expressivity.

This class of models may also be called "autoregressive U-Nets", characterized by the U-Net multi-scale architecture (Ronneberger, Fischer, and Brox 2015) with additional considerations to maintain causality. Prior to these, the S4 and SaShiMi models (Gu, Goel, and Ré 2022; Goel et al. 2022) used the same architecture successfully in the vision and audio modalities, where fixed-window downsampling exhibits more appropriate inductive bias in contrast to language. SaShiMi specifically operated over 8-bit quantized audio inputs, hence also was a form of byte-level modeling that used BPB as a metric.

#### <span id="page-31-0"></span>A.1.3 External Chunking

An improvement to hierarchical architectures with static downsampling is to use content-aware chunking strategies that attempt to identify natural token boundaries based on semantic or statistical properties of the input data. Several recent models propose using the boundaries provided by an external module, with two main variations appearing.

**Delimiter-based methods.** The most intuitive content-aware approach segments on surface-level syntactical boundaries, which can be often implemented by simple rules or regular expressions.

Dynamic Pooling Transformer (DPT) (Nawrot, Chorowski, et al. 2023) proposed a variant that segmented on whitespace characters, effectively making each word its own token. SpaceByte (Slagle 2024) extends this to "space-like" delimiters (*e.g.*, /, ], :) as natural boundary signals. This approach provides semantically meaningful chunking for languages with explicit word separators such as English text and code.

However, delimiter-based methods cannot be used for inputs lacking explicit separators (e.g. many non-European languages, or other modalities such as DNA). Additionally, these approaches cannot be extended to multi-level hierarchical chunking due to ambiguities in defining natural delimiters at higher semantic levels. AU-Net (Videau et al. 2025) is a concurrent work that augments SpaceByte with additional stages of hierarchy using fixed-width chunking. Specifically, AU-Net 2 is SpaceByte with minor architectural modifications, while AU-Net 3 (and AU-Net 4) add additional levels of hierarchical with width-2 downsampling.

In this work, we show that SpaceByte's delimiter chunking strategy can be a very powerful baseline on appropriate languages – competitive with or outperforming traditional tokenizers on English and code – when augmented with several of H-Net's additional techniques (Section 3.1, Section 3.3, Figure 5, Figure 9).

**Entropy-based methods.** Another approach to circumvent the delimiter dependency is using the autoregressive conditional entropy as a heuristic to identify semantic boundaries. This was first proposed by the Dynamic Pooling Transformer (DPT) (Nawrot, Chorowski, et al. 2023), which detects entropy spikes that correlate with semantic transitions. The recent Byte Latent Transformer (BLT) (Pagnoni et al. 2024) employs entropy thresholds computed by a separate pre-trained model to determine chunking boundaries.

Despite showing promise, these entropy-based approaches face several practical limitations. First, they require extensive domain-specific hyperparameter tuning to establish appropriate entropy thresholds, reducing their general applicability. Second, they still fall behind in performance; for example, BLT necessitates an extra 3B parameters (at the 8B scale) solely for multi-gram hash embeddings to match BPE Transformer baselines. Finally, these methods also cannot be extended hierarchically because computing cross-entropy loss requires access to target vocabularies, which are unavailable for intermediate latent representations in multi-stage architectures.

In this work, we do not compare against BLT because of its complexity: (i) necessitating training an auxiliary language model to provide proxy autoregressive conditional entropies (ii) converting it into an external neural tokenizer through

tuning entropy heuristics (iii) using hash embeddings, which can be considered an orthogonal architectural component which may be incorporated into H-Net as well if desired.

Instead, we compared against SpaceByte (and our own stronger versions of SpaceByte), which we believe to be representative of the external-chunking family of methods and competitive to the entropy-based chunking strategy of BLT (for our main experiments such as English data).

### A.1.4 Dynamic Chunking

The ideal tokenizer-free architecture would incorporate a dynamic chunking method that attempts to learn optimal segmentation strategies directly from data through gradient-based optimization. Such a method would be optimized jointly together with the outer (fine-resolution) and inner (coarse-resolution) networks, and be able to create boundaries that are content- and context- aware.

The only prior work we are aware of that attempted a true dynamic chunking method is (one variant of) the Dynamic Pooling Transformer (DPT) (Nawrot, Chorowski, et al. [2023\)](#page-27-4), which incorporates stochastic exploration mechanisms using Gumbel noise (Jang, Gu, and Poole [2017;](#page-26-2) Maddison, Mnih, and Teh [2017\)](#page-26-13) to enable differentiable boundary selection during training. Despite their theoretical flexibility, trainable methods encounter critical challenges. The stochastic exploration process requires careful tuning of noise magnitudes and introduces high-variance gradients that destabilize training, making it difficult to scale to larger model sizes.

In practice, the end-to-end (stochastic reparameterization) variant of DPT underperformed the external chunking variants (drawing boundaries on entropy spikes or whitespaces) (Nawrot, Chorowski, et al. [2023\)](#page-27-4), illustrating the difficulty of this problem. Furthermore, the training instability prevented DPT from expanding to multiple hierarchical stages, constraining these methods to single-stage chunking.

We additionally highlight simple architectural modifications of DPT motivated by improved inference (Fleshman and Van Durme [2023\)](#page-25-18) or multilingual ability (Ahia, Kumar, Gonen, Hofmann, et al. [2024\)](#page-24-15). Such techniques can also be easily adapted to H-Nets in future work.

## <span id="page-32-0"></span>A.2 Non-Autoregressive Tokenizer-free Architectures

Each class of autoregressive architectures from Appendix [A.1](#page-30-1) has corresponding non-autoregressive variants as well. Although these often have similar design principles, they are also motivated by different tasks, settings, and design considerations (e.g. no evaluation on large-scale autoregressive pretraining) and thus can be difficult to compare directly to autoregressive models. We include these for context and completeness.

Isotropic. ByT5 (Xue et al. [2022\)](#page-28-2) directly models bytes using a bidirectional encoder-decoder architecture, showing improved performance with small models (because more power is moved into model parameters rather than vocabulary embeddings) and spelling-sensitive tasks.

Hierarchical (Static). Funnel-Transformer (Dai et al. [2020\)](#page-24-16) is an early architecture that uses a U-Net-like architecture for language, focusing on the non-causal setting. Canine (Clark, Garrette, et al. [2022\)](#page-24-3) proposes a hierarchical model with convolution-based static downsampling; their method also targets non-autoregressive language models.

Charformer (Tay et al. [2021\)](#page-28-17) presents a gradient-based subword tokenization (GBST) method that pools the input sequence at different resolutions, inducing an implicit ensemble of hierarchical models. It shows improved efficiency to performance trade-offs compared to models that use a single downsample resolution.

We note that these methods can also be endowed with implicit supervision from external tokenizers; for example, Canine proposes a variant that uses subword tokens in the objective function (via masking out subwords in the masked language modeling objective), but does not need the tokenizer at inference time. We also note that such techniques are particular to non-autoregressive models, since they allow for variations in the modeling objective.

Hierarchical (External). Thawani et al. [\(2023\)](#page-28-18) propose the eByte method, which resembles MegaByte but chunks on spaces with Transformer-based CLS-token pooling, and lacks the byte-level residual stream that enables autoregressive modeling. Word-based self-attention fusion (WSF) (Sreedhar et al. 2023) proposes a similar pooling strategy for encoder language models.

**Hierarchical (Dynamic).** MANTa (Godey et al. 2022) introduces an end-to-end method that predicts segmentation boundaries and pools bytes into blocks using a matching objective. MrT5 (Kallini et al. 2025) is a recent method improving on ByT5 with a gating mechanism that allows for explicit dynamic token-merging at inference time, reducing sequence lengths by up to 80%.

#### <span id="page-33-0"></span>A.3 Other Tokenization-related Work

**Tokenizers for Other Modalities.** While computer vision pipelines do not use tokenizers like BPE in the same way as language models do, they frequently need to turn raw perceptual data (images and videos) into shorter sequences of representations. One approach is the simple patchification step first introduced by the Vision Transformer (ViT) (Dosovitskiy et al. 2021). However, images, videos, and audio can have varying amounts of semantic content and non-uniform redundancies. A number of more recent approaches attempt to produce variable length tokenizations that adapt to the information content of the data, Which performs a more similar role to tokenization in language models. This can be done in the latent space of an autoencoder (Yu, Weber, et al. 2024; Duggal et al. 2024) or through explicit token merging (or "run length encoding") with heuristics (Bolya et al. 2022; Choudhury et al. 2024). In the audio domain, SlowAE (Dieleman et al. 2021) proposes a joint autoencoder with autoregressive modeling that finds semantic segmentation boundaries, which resembles H-Net's approach at a high level.

FAST (Lin et al. 2025) introduces a tokenizer for robotics, Which tokenizes continuous control actions by combining the Discrete Cosine Transform (DCT) with BPE.

**Vocabulary Scaling.** While scaling laws for language models have generally kept tokenizers fixed (Kaplan et al. 2020; Hoffmann et al. 2022; Grattafiori et al. 2024), recent works have showed that the tokenizer also warps scaling laws, in fact more so than model architecture changes (Mayilvahanan et al. 2025). Tao et al. (2024) and Huang et al. (2025) directly show that it is more optimal to scale an LLM's vocabulary together with the rest of the model parameters.

In H-Nets, which are designed to operate over higher resolution raw data, the actual vocabulary can be kept minimal, but the chunking mechanism can be viewed as an implicit "tokenizer" with infinite vocabulary. As H-Nets scale in size, one expects that more iterations of hierarchy can be added (increasing effective chunk size), or the chunk size can directly be increased to leverage parameters more efficiently. This resembles the idea of increasing a vocabulary in tokenized models (which would generally increase the average length of tokens).

SuperBPE (Liu et al. 2025) shows that allowing vocabulary tokens to cross whitespace boundaries can also improve performance. This is related to H-Net's motivation of higher-level chunking of words into phrases; empirically, Figure 4 shows how the 2-stage H-Net finds semantic multi-word groups in the inner stage.

**Cross-Tokenizer Transfer.** Minixhofer, Ponti, and Vulić (2024) and Minixhofer, Vulić, and Ponti (2025) address the problem of *tokenizer transfer*, or adapting models across different tokenizers (for example for cross-language or cross-modality usage, or for knowledge distillation).

**Other Effects of Tokenization.** Lee et al. (2024) discuss the effects that tokenization has on arithmetic in LLMs. For example, comparing the performance of left-to-right vs. right-to-left tokenization. Hayase et al. (2024) show that examining the vocabulary of a BPE tokenizer leaks information about the data mix that it was trained on.

**Tokenization Theory.** Schmidt et al. (2024) examined the hypothesis that the primary role of tokenization is to shrink the input sequence length. They invented a new tokenizer that has even higher compression rates than BPE (actually, they keep the same vocabulary but simply find different segmentations that are more compressed) yet leads to worse language models, providing evidence against the hypothesis.

Rajaraman, Jiao, and Ramchandran (2024) showed that for certain data distributions, applying tokenization qualitatively changes what Transformers can learn.

Phan et al. (2024) and Vieira et al. (2024) propose various algorithms for converting a language model over tokens into a language model over characters or bytes. This helps alleviate some limitations of tokenizers such as the "prompt boundary" problem, the ability to compare different LLMs with different tokenizers, and simply produces better estimates of a language model's true compressive ability (as measured by bits-per-byte). However, such algorithms are complex and expensive, and compared to direct byte-level models they are not practical for use during inference decoding (repeated autoregressive sampling).

