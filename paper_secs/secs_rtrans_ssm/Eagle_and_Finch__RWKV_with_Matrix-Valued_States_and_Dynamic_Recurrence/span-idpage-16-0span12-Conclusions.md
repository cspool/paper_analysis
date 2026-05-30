# <span id="page-16-0"></span>**12 Conclusions**

In this work, we introduced Eagle (RWKV-5) and Finch (RWKV-6), marking substantial progress in RNN-based language models by integrating multiheaded matrix-valued states and dynamic data-driven recurrence mechanisms. These models demonstrate exceptional performance on MQAR and diverse linguistic benchmarks, challenging the dominance of traditional Transformer architectures while retaining key RNN advantages. With models publicly available under the Apache 2.0 license and trained on an extensive multilingual corpus, our work not only advances the capabilities of language models but also emphasizes community accessibility and applicability across various domains. While acknowledging the computational and ethical challenges ahead, we hope that Eagle and Finch's efficient new architecture and wide availability will help push the boundaries of language modeling and pave the way for future innovations.

**Limitations** The Eagle and Finch models fall short on certain aspects that can be mitigated and addressed in future work.

We experimented with using Eagle as an embedding model on the Massive Text Embedding Benchmark (MTEB) [\(Muennighoff et al.,](#page-21-8) [2023\)](#page-21-8) but were not able to get strong embedding performance. We believe that its state is a very high-quality embedding of the context but an appropriate method is required to aggregate the information content. We leave this to future work.

Because our training corpus contains some synthetic data from GPT-3.5 and ChatGPT, our released models exhibit behaviors similar to ChatGPT and will mimic ChatGPT's conversation style and tone. For instance, the model might occasionally claim that it is trained by OpenAI. However, this is not a general property the RWKV architecture but rather a specific outcome of the data and training process.

<span id="page-16-2"></span>**Future Work** Our 1.12 trillion token multilingual training corpus is much smaller than the training data sizes for contemporary models such as LLaMA2 [\(Touvron et al.,](#page-23-5) [2023\)](#page-23-5), and expanding our training corpus to be more diverse and expansive is a key priority to improving model performance [\(Albalak et al.,](#page-17-10) [2024\)](#page-17-10). We also plan to train and release larger versions of Finch such as 7B and 14B parameters, and further extend its performance with reduced inference and training costs via Mixture of Experts [\(Shazeer et al.,](#page-22-11) [2017\)](#page-22-11).

#### **Acknowledgments**

We thank Stability AI for the compute used to train our models and for technical support in the development of RWKV. We also thank the members of the RWKV and EleutherAI Discord servers for their help and work on further extending the applicability of RWKV to different domains. We also thank Shenzhen Yuanshi Intelligence Co., Ltd. for its contribution to the promotion and commercialization of RWKV. We thank Songlin Yang for assistance with the code and ideas for our time-parallel implementations.

