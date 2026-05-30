# <span id="page-15-3"></span>B IMPLEMENTATION DETAILS

### <span id="page-15-1"></span>B.1 DATA DETAILS

Consistent with previous works, we use the tokenizer of LLaMA2, which contains 65,536 vocabulary tokens. It is worth noting that MoE++ is trained exclusively on public datasets, making it accessible for academic research settings. Specifically, we sample from the following datasets according to different sampling probabilities:

- The RedPajama [\(Computer,](#page-10-6) [2023a\)](#page-10-6) includes training data from seven domains: Common-Crawl, C4, Github, Wikipedia, Books, ArXiv, and StackExchange.
- The Dolma [\(Soldaini et al.,](#page-12-11) [2024\)](#page-12-11), a large and diverse open English text corpus, contains 3 trillion tokens sampled from seven sources, including web pages from Common Crawl, code from The Stack, curated web data from C4 [\(Raffel et al.,](#page-12-6) [2020\)](#page-12-6), social media conversations from Reddit, academic papers from PeS2o, public domain books from Project Gutenberg, and comprehensive content from Wikipedia and Wikibooks.
- The Pile [\(Gao et al.,](#page-11-11) [2020\)](#page-11-11), an open-source English text corpus for training large language models, includes 22 diverse, publicly available datasets such as Wikipedia, NIH ExPorter, ArXiv, Books3, BookCorpus2, OpenSubtitles, YoutubeSubtitles, and Enron Emails.

Tab. [A](#page-15-0) shows the detailed sample ratios of different open-source datasets. We find that increasing the ratio of high-quality data, such as Books and Wikipedia, during the later stages of training significantly enhances model performance. Consequently, for the "MoE++ 7B/(16+4)E" model in Tab. [4,](#page-7-0) We increase the ratio of high-quality data for the final 100B tokens. Specifically, this model is trained using strategy 1 for the first 900B tokens and strategy 2 for the last 100B tokens, for a total training budget of 1T tokens. In contrast, for simplicity, all MoE++ and MoE models in Tab. [3](#page-6-0) are trained with strategy 1, using a budget of 100B tokens.

<span id="page-15-0"></span>Table A: Sampling ratio of different open-source datasets. All MoE++ and MoE models in Tab. [3](#page-6-0) are trained using strategy 1 with a budget of 100B tokens. In contrast, for the "MoE++ 7B/(16+4)E" model in Tab. [4,](#page-7-0) strategy 1 is applied for the first 900B tokens, and strategy 2 for the final 100B tokens, resulting in a total training budget of 1T tokens.

| Strategy 1 | Strategy 2                       |
|------------|----------------------------------|
|            | 13.93%                           |
|            | 9.03%                            |
|            | 11.36%                           |
|            | 9.77%                            |
| 10.94%     | 7.42%                            |
| 61.28%     | 41.53%                           |
| 12.48%     | 6.96%                            |
|            | 4.24%<br>3.50%<br>4.37%<br>3.19% |

### <span id="page-15-2"></span>B.2 TRAINING HYPER-PARAMETERS

Tab. [B](#page-16-0) shows the detailed training hyper-parameters. Specifically, the hyper-parameters for MoE++ are selected based on the common practice for dense transformer language models. We replace all FFN layers in the transformer with MoE++ layers and set the Top-K to 2 for every layer, resulting in approximately twice the computation compared to a dense model. The weight β for the heterogeneous load balance loss is set to 0.01, and the expert capacity factor γ is set to 1.1. MoE++ is trained using the AdamW optimizer [\(Loshchilov & Hutter,](#page-12-16) [2017\)](#page-12-16). During training, a weight decay of 0.1 and gradient clipping of 1.0 are applied. All MoE++ (except for the "MoE++ 7B/(16+4)E" with 8-way pipeline parallel) and MoE models in Tab. [3](#page-6-0) are trained using strategy 1 with a maximum learning

rate of 5e-4 and a batch size of 4 million tokens with a sequence length of 2048. In contrast, for the "MoE++ 7B/(16+4)E" model in Tab. [4,](#page-7-0) strategy 2 is applied for the first 900B tokens, and strategy 3 for the final 100B tokens, resulting in a total training budget of 1T tokens.

<span id="page-16-0"></span>Table B: Training hyper-parameters. All MoE++ (except for the "MoE++ 7B/(16+4)E" with 8-way pipeline parallel) and MoE models in Tab. [3](#page-6-0) are trained using strategy 1 with a budget of 100B tokens. In contrast, for the "MoE++ 7B/(16+4)E" model in Tab. [4,](#page-7-0) strategy 2 is applied for the first 900B tokens, and strategy 3 for the final 100B tokens, resulting in a total training budget of 1T tokens.

|                       | Strategy 1 | Strategy 2 | Strategy 3 |
|-----------------------|------------|------------|------------|
| Training budget       | 100B       | 900B       | 100B       |
| Maximum learning rate | 5e-4       | 5e-4       | 1e-4       |
| Final learning rate   | 5e-5       | 5e-5       | 1e-5       |
| LR warmup init        | 1e-7       | 1e-7       | 1e-7       |
| LR warmup iters       | 2000       | 500        | 200        |
| Sequence length       | 2048       | 2048       | 2048       |
| Batch size (tokens)   | 4M         | 4M         | 4M         |
| Capacity factor γ     | 1.1        | 1.1        | 1.1        |
| β for Lb              | 0.01       | 0.01       | 0.01       |
| Tensor parallel       | 1          | 1          | 1          |
| Pipeline parallel     | 1          | 8          | 8          |

