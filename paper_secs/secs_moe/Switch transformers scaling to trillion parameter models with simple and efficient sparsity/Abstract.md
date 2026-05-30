# Abstract

In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) models defy this and instead select different parameters for each incoming example. The result is a sparsely-activated model—with an outrageous number of parameters—but a constant computational cost. However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs, and training instability. We address these with the introduction of the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques mitigate the instabilities, and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats. We design models based off T5-Base and T5-Large [\(Raffel](#page-37-0) [et al.,](#page-37-0) [2019\)](#page-37-0) to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the "Colossal Clean Crawled Corpus", and achieve a 4x speedup over the T5-XXL model.[1](#page-0-0)[2](#page-0-1)

Keywords: mixture-of-experts, natural language processing, sparsity, large-scale machine learning, distributed computing

<sup>∗</sup>. Equal contribution.

<span id="page-0-0"></span><sup>1.</sup> JAX code for Switch Transformer and all model checkpoints are available at [https://github.com/](https://github.com/google-research/t5x) [google-research/t5x](https://github.com/google-research/t5x)

<span id="page-0-1"></span><sup>2.</sup> Tensorflow code for Switch Transformer is available at [https://github.com/tensorflow/mesh/blob/](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py) [master/mesh\\_tensorflow/transformer/moe.py](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)

#### Fedus, Zoph and Shazeer

