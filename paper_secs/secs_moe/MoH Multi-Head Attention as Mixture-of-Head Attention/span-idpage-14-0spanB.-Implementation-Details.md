# <span id="page-14-0"></span>B. Implementation Details

### B.1. ViT for Image Classification

Training Details. Our MoH-ViT models are trained for 300 epochs using automatic mixed precision across 8 GPUs. We follow the training strategy of TransNeXt, which includes various data augmentation techniques, including Random Augmentation [\(Cubuk et al.,](#page-9-14) [2020\)](#page-9-14), Mixup [\(Zhang,](#page-12-14) [2017\)](#page-12-14), CutMix [\(Yun et al.,](#page-12-15) [2019\)](#page-12-15), and Random Erasing [\(Zhong et al.,](#page-12-16) [2020\)](#page-12-16). We also apply Label Smoothing [\(Szegedy et al.,](#page-12-17) [2016\)](#page-12-17) and DropPath [\(Huang et al.,](#page-10-9) [2016\)](#page-10-9) to regularize our models. We optimize our models using AdamW optimizer [\(Loshchilov & Hutter,](#page-11-12) [2017\)](#page-11-12) with a gradient clipping norm of 1.0 and a weight decay of 0.05. The initial learning rate is set to 1e-3, with a 5-epoch warm-up starting at 1e-6. A cosine learning rate scheduler [\(Loshchilov & Hutter,](#page-11-13) [2016\)](#page-11-13) is employed to decay the learning rate. During training, images are randomly cropped to a size of 224×224. It is worth noting that we do not use Exponential Moving Average (EMA) weights.

#### B.2. DiT for Class-Conditional Image Generation

Training Details. Following DiT, the final linear layer is initialized with zeros, and all other layers follow standard ViT weight initialization. We train all models using the AdamW optimizer [\(Loshchilov & Hutter,](#page-11-12) [2017\)](#page-11-12) with a constant learning rate of 1e-4, no weight decay, and a batch size of 256, applying horizontal flips for data augmentation. Following DiT, we employ the Exponential Moving Average (EMA) of MoH-DiT weights during training with a decay rate of 0.9999, generating all images using the EMA model. We use an off-the-shelf pre-trained variational autoencoder [\(Kingma,](#page-10-13) [2013\)](#page-10-13) model from Stable Diffusion [\(Rombach et al.,](#page-11-16) [2022\)](#page-11-16). Following TransNeXt, our attention-head activation budget is unevenly distributed across layers, with fewer attention heads activated in the shallow layers and more in the deeper layers.

### B.3. Training LLMs from Scratch

Model Settings. For training LLMs from scratch, we use Megatron [\(Shoeybi et al.,](#page-11-17) [2019\)](#page-11-17), an open-source training code, as the training framework. The detailed hyper-parameter settings of various MoH-LLMs are shown in Tab. [C.](#page-5-0)

Table C. Sizes and architectures of MoH-LLMs and LLMs. "MoH-LLM-B" has more parameters than "LLM-B" due to the additional parameters introduced by the router network.

| Methods            | #Params    | #Layers | #Hidden Size | #Intermediate Size | #Heads | #Head Dim |
|--------------------|------------|---------|--------------|--------------------|--------|-----------|
| LLM-S<br>MoH-LLM-S | 186<br>186 | 12      | 768          | 2048               | 12     | 64        |
| LLM-B<br>MoH-LLM-B | 881<br>882 | 24      | 1536         | 4096               | 16     | 96        |

Data Details. Consistent with previous works, we use the tokenizer of LLaMA2, which contains 65,536 vocabulary tokens. It is worth noting that MoH-LLM is trained exclusively on public datasets, making it accessible for academic research settings. Tab. [D](#page-5-1) shows the detailed sample ratios of different open-source datasets. Specifically, we sample from the following datasets according to different sampling probabilities:

- The RedPajama [\(Computer,](#page-9-16) [2023\)](#page-9-16) includes training data from seven domains: CommonCrawl, C4, Github, Wikipedia, Books, ArXiv, and StackExchange.
- The Dolma [\(Soldaini et al.,](#page-11-21) [2024\)](#page-11-21), a large and diverse open English text corpus, contains 3 trillion tokens sampled from seven sources, including web pages from Common Crawl, code from The Stack, curated web data from C4 [\(Raffel](#page-11-26) [et al.,](#page-11-26) [2020\)](#page-11-26), social media conversations from Reddit, academic papers from PeS2o, public domain books from Project Gutenberg, and comprehensive content from Wikipedia and Wikibooks.
- The Pile [\(Gao et al.,](#page-10-15) [2020\)](#page-10-15), an open-source English text corpus for training large language models, includes 22 diverse, publicly available datasets such as Wikipedia, NIH ExPorter, ArXiv, Books3, BookCorpus2, OpenSubtitles, YoutubeSubtitles, and Enron Emails.

Table D. Sampling ratio of different open-source datasets for MoH-LLMs. MoH-LLM is trained exclusively on public datasets, making it accessible for academic research settings.

|                         | Sampling Ratio |
|-------------------------|----------------|
| Redpajama Books         | 4.24%          |
| Redpajama Wikipedia     | 3.50%          |
| Redpajama ArXiv         | 4.37%          |
| Redpajama StackExchange | 3.19%          |
| Redpajama C4            | 10.94%         |
| Dolma                   | 61.28%         |
| Pile                    | 12.48%         |

Training Hyper-Parameters. Tab. [E](#page-6-1) shows the detailed training hyper-parameters of MoH-LLMs. Specifically, all MoH-LLMs are trained with the AdamW optimizer [\(Loshchilov & Hutter,](#page-11-12) [2017\)](#page-11-12), using a batch size of 4 million tokens with a sequence length of 2048. The final learning rate is set to 10% of the maximum. During training, a weight decay of 0.1 and gradient clipping of 1.0 are applied. For LLM-S and MoH-LLM-S, the maximum learning rate is set to 3e-4. For LLM-B and MoH-LLM-B, the maximum learning rate is set to 5e-4.

MoH-LLM-S 100B MoH-LLM-B 100B MoH-LLM-B 200B (LLM-S 100B) (LLM-B 100B) (LLM-B 200B) Training budget 100B 100B 200B Maximum learning rate 3e-4 5e-4 5e-4 Final learning rate 3e-5 5e-5 5e-5 LR warmup init 1e-7 1e-7 1e-7 LR warmup iters 2000 500 500 Sequence length 2048 2048 2048 Batch size (tokens) 4M 4M 4M β for L<sup>b</sup> 0.01 0.01 0.01 Tensor parallel 1 1 1 Pipeline parallel 1 1 1

Table E. Training hyper-parameters of MoH-LLMs.

#### B.4. Continue-Tuning LLaMA3-8B

Training Hyper-Parameters. Tab. [F](#page-7-1) shows the detailed training hyper-parameters of MoH-LLaMA3-8B. We find that if there is a discrepancy between the continue-training data and the original training data distribution of the model, the performance of the model may fluctuate wildly at the beginning of the training process. Since we do not have access to the raw training data of LLaMA3, we address these potential performance fluctuations by dividing the training process into two stages. In the first stage, we continue-tune the original LLaMA3-8B model using 300B tokens to adapt it to our dataset. In addition, during the first stage, to enhance the Chinese ability of the model, we expand the vocabulary size. Specifically, we increase the original LLaMA3-8B vocabulary size from 128,256 to 160,896. In the second stage, we continue-tune this adapted model into our proposed MoH model with 100B tokens. During the first stage, the maximum learning rate is set to 6e-5, and the final learning rate is 6e-6. In the second stage, the maximum learning rate is set to 2e-5, and the final learning rate is 1e-6. For both stages, we employ the AdamW optimizer [\(Loshchilov & Hutter,](#page-11-12) [2017\)](#page-11-12), with a batch size of 16 million tokens with a sequence length of 8192. During training, we use a weight decay of 0.1 and gradient clipping of 1.0.

Table F. Training hyper-parameters of MoH-LLaMA3-8B. We divide the training process into two stages. In the first stage, we continue-tune the LLaMA3-8B model using 300B tokens. In the second stage, we continue-tune this adapted model into our proposed MoH model with 100B tokens.

|                       | The First Stage | The Second Stage |
|-----------------------|-----------------|------------------|
| Training budget       | 300B            | 100B             |
| Maximum learning rate | 6e-5            | 2e-5             |
| Final learning rate   | 6e-6            | 1e-6             |
| LR warmup iters       | 50              | 50               |
| Sequence length       | 8192            | 8192             |
| Batch size (tokens)   | 16M             | 16M              |
| β for Lb              | -               | 0.01             |
| Tensor parallel       | 2               | 1                |
| Pipeline parallel     | 1               | 8                |

Table G. Comparisons between MoH-LLaMA3-8B and LLaMA3-8B-stage1. MoH-LLaMA3-8B outperforms LLaMA3-8B-stage1 by utilizing only 75% of the attention heads.

| Methods          | #Activated<br>Heads (%) | MMLU (5)       | CMMLU (5)  | NQ (32)    | GSM8K(8)   | TruthfulQA |
|------------------|-------------------------|----------------|------------|------------|------------|------------|
| LLaMA3-8B-stage1 | 100                     | 66.2           | 66.0       | 28.1       | 58.6       | 41.9       |
| MoH-LLaMA3-8B    | 75                      | 65.8<br>64.4   |            | 28.3       | 56.9       | 44.0       |
| Methods          | #Activated<br>Heads (%) | HellaSwag (10) | LogiQA     | BoolQ (32) | LAMBADA    | SciQ       |
| LLaMA3-8B-stage1 | 100                     | 79.4           | 30.4       | 85.1       | 75.8       | 92.2       |
| MoH-LLaMA3-8B    | 75                      | 80.1           | 30.3       | 84.0       | 76.4       | 92.2       |
| Methods          | #Activated<br>Heads (%) | PIQA           | WinoGrande | ARC-E      | ARC-C (25) | Average    |
| LLaMA3-8B-stage1 | 100                     | 79.1           | 73.0       | 70.9       | 59.6       | 64.7       |
| MoH-LLaMA3-8B    | 75                      | 78.8           | 72.9       | 72.5       | 60.1       | 64.8       |

