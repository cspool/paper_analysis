# X2Img、VQGAN+ARM、LDM：19-22

## Text2Img、Img2Img

### 文本、图片作为条件生成

VQVAE类模型生成图像是生成编码或纹理序列后解码出图像，Diffusion类模型生成图像是随机噪声多步去噪生成图像。文生图的目标是按照文本描述生成图像，则VQVAE类模型推理的直观改变是**按照文本描述生成纹理序列**，Diffusion类模型推理的直观改变是**按照文本描述控制去噪**过程。

VQVAE中GPT生成的texture序列是图像tile信息的”空间“序列，Diffusion中生成轨迹是含噪声图像逐渐去除噪声的”时间“过程。texture”空间“序列和生成”时间“序列都是**以过去状态为条件**逐个生成，在生成每个texture code或预测去除的噪声时，**增加文本信息作为额外条件**，以便生成符合文本描述的图像。

### NAACL19：BERT（ARM）

NAACL19：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> **[图片提取文字 (image.png)]:**
> ## 自回归模型应用线性回归,其输出中的滞后变量取自先前的步骤。与线性回归不同,自回归模型除了先前预测的结果外,不使用其他自变
> 
> 量。请考虑以下公式。
> 
> 白回归
> 
>  $p(x) = \prod_{i=1}^{n} p(x_i | x_1, x_2, \dots x_{i-1}) = \prod_{i=1}^{n} p(x_i | x_{< i})$ 
> 
> 我们也可以用下面的方程来表示自回归建模。
> 
> $$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t$$
> 
>  $y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t$ 
> 
> 在这里,y是先前结果的多个阶数乘以它们各自的系数 Ø 的预测结果。系数表示影响预测器对新结果的重要性的权重或参数。该公式还考虑了可能影响预测的随机噪声,表明模型并不理想,可以进一步改进。
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image.png)

[CLS]是句子开头，[SEP]是句子分隔符，类似标点符号，表示句子之间结构信息。

输入层将输入转为Token+Segment+Position，输出层连接预训练任务，或下游任务用于微调。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Overall pre-training and fine-tuning procedures for BERT. Apart from output layers, the same architectures are used in both pre-training and fine-tuning. The same pre-trained model parameters are used to initialize models for different down-stream tasks. During fine-tuning, all parameters are fine-tuned. [CLS] is a special symbol added in front of every input example, and [SEP] is a special separator token (e.g. separating questions/answers).
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%201.png)

> **[图片提取文字 (image.png)]:**
> handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., (Question, Answer)) in one token sequence. Throughout this work, a "sentence" can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A "sequence" refers to the input token sequence to BERT, which may be a sin-
> 
> gle sentence or two sentences packed together.
> 
> **Input/Output Representations** To make BERT
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%203.png)

> **[图片提取文字 (image.png)]:**
> token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B. As shown in Figure 1, we denote input embedding as E, the final hidden vector of the special [CLS] token as  $C \in \mathbb{R}^H$ , and the final hidden vector for the  $i^{th}$  input token as  $T_i \in \mathbb{R}^H$ . For a given token, its input representation is constructed by summing the corresponding token,
> 
> segment, and position embeddings. A visualiza-
> 
> tion of this construction can be seen in Figure 2.
> 
> We use WordPiece embeddings (Wu et al.,
> 
> 2016) with a 30,000 token vocabulary. The first
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%204.png)

LLM所用Transformer架构区别

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-to-left LSTMs to generate features for downstream tasks. Among the three, only BERT representations are jointly conditioned on both left and right context in all layers. In addition to the architecture differences, BERT and OpenAI GPT are fine-tuning approaches, while ELMo is a feature-based approach.
> 
> Figure 3: Differences in pre-training model architectures. BERT uses a bidirectional Transformer. OpenAI GPT
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%205.png)

pre-training task

> **[图片提取文字 (image.png)]:**
> Masked LM and the Masking Procedure Assuming the unlabeled sentence is my dog is hairy, and during the random masking procedure we chose the 4-th token (which corresponding to hairy), our masking procedure can be further illustrated by
> 
> • 80% of the time: Replace the word with the
> 
> [MASK] token, e.g., my dog is hairy  $\rightarrow$ 
> 
> • 10% of the time: Replace the word with a
> 
> We provide examples of the pre-training tasks in
> 
> the following.
> 
> random word, e.g., my dog is hairy → my dog is apple
> 10% of the time: Keep the word un-
> 
> my dog is [MASK]
> 
> • 10% of the time: Keep the word unchanged, e.g., my dog is hairy → my dog is hairy. The purpose of this is to bias the representation towards the actual observed word.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%206.png)

> **[图片提取文字 (image.png)]:**
> prediction task can be illustrated in the following examples.
> 
> **Next Sentence Prediction** The next sentence
> 
> ```
> Input = [CLS] the man went to [MASK] store [SEP]
> he bought a gallon [MASK] milk [SEP]
> ```
> 
> Label = IsNext
> 
> ```
> Input = \text{[CLS]} the man [MASK] to the store [SEP]
>  penguin [MASK] are flight ##less birds [SEP]
> ```
> 
> Label = NotNext
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%207.png)

pre-training过程

> **[图片提取文字 (image.png)]:**
> ## **A.2 Pre-training Procedure**To generate each training input sequence, we sam-
> 
> refer to as "sentences" even though they are typically much longer than single sentences (but can be shorter also). The first sentence receives the A embedding and the second receives the B embed-
> 
> ding. 50% of the time B is the actual next sentence
> 
> that follows A and 50% of the time it is a random
> 
> ple two spans of text from the corpus, which we
> 
> sentence, which is done for the "next sentence prediction" task. They are sampled such that the combined length is  $\leq 512$  tokens. The LM masking is applied after WordPiece tokenization with a uniform masking rate of 15%, and no special consideration given to partial word pieces.
> 
> We train with batch size of 256 sequences (256)
> 
> sequences \* 512 tokens = 128,000 tokens/batch)
> 
> for 1,000,000 steps, which is approximately 40
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%208.png)

> **[图片提取文字 (image.png)]:**
> rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. We use a gelu activation (Hendrycks and Gimpel, 2016) rather than the standard relu, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood. Training of BERT<sub>BASE</sub> was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total).<sup>13</sup> Training of BERT<sub>LARGE</sub> was performed on 16 Cloud TPUs (64 TPU chips total). Each pretraining took 4 days to complete. Longer sequences are disproportionately expensive because attention is quadratic to the sequence length. To speed up pretraing in our experiments, we pre-train the model with sequence length of 128 for 90% of the steps. Then, we train the rest 10% of the steps of sequence of 512 to learn the positional embeddings.
> 
> epochs over the 3.3 billion word corpus. We
> 
> use Adam with learning rate of 1e-4,  $\beta_1 = 0.9$ ,
> 
>  $\beta_2 = 0.999$ , L2 weight decay of 0.01, learning
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%209.png)

fine-tune过程

> **[图片提取文字 (image.png)]:**
> ## **3.2 Fine-tuning BERT**Fine-tuning is straightforward since the self-
> 
> lows BERT to model many downstream tasks—whether they involve single text or text pairs—by swapping out the appropriate inputs and outputs. For applications involving text pairs, a common pattern is to independently encode text pairs before applying bidirectional cross attention, such
> 
> attention mechanism in the Transformer al-
> 
> as Parikh et al. (2016); Seo et al. (2017). BERT instead uses the self-attention mechanism to unify these two stages, as encoding a concatenated text pair with self-attention effectively includes *bidirectional* cross attention between two sentences. For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end. At the input, sentence A and sentence B from pre-training
> 
> are analogous to (1) sentence pairs in paraphras-
> 
> ing, (2) hypothesis-premise pairs in entailment, (3)
> 
> question-passage pairs in question answering, and
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2010.png)

> **[图片提取文字 (image.png)]:**
> (4) a degenerate text-\( \varnothing \) pair in text classification or sequence tagging. At the output, the token representations are fed into an output layer for tokenlevel tasks, such as sequence tagging or question answering, and the [CLS] representation is fed into an output layer for classification, such as entailment or sentiment analysis.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## **A.3** Fine-tuning Procedure For fine-tuning, most model hyperparameters are
> 
> the same as in pre-training, with the exception of the batch size, learning rate, and number of training epochs. The dropout probability was always
> 
> <sup>13</sup>https://cloudplatform.googleblog.com/2018/06/Cloud-
> 
> of possible values to work well across all tasks:
> 
> TPU-now-offers-preemptible-pricing-and-global-
> 
> • **Batch size**: 16, 32
> 
> availability.html
> 
> kept at 0.1. The optimal hyperparameter values are task-specific, but we found the following range
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2012.png)

> **[图片提取文字 (image.png)]:**
> • **Number of epochs**: 2, 3, 4
> 
> • Learning rate (Adam): 5e-5, 3e-5, 2e-5
> 
> 100k+ labeled training examples) were far less sensitive to hyperparameter choice than small data sets. Fine-tuning is typically very fast, so it is reasonable to simply run an exhaustive search over the above parameters and choose the model that performs best on the development set.
> 
> We also observed that large data sets (e.g.,
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2013.png)

不同任务的fine-tuning

> **[图片提取文字 (image.png)]:**
> ## Tasks The illustration of fine-tuning BERT on different tasks can be seen in Figure 4. Our task-specific models are formed by incorporating BERT with
> 
> one additional output layer, so a minimal num-
> 
> A.5 Illustrations of Fine-tuning on Different
> 
> ber of parameters need to be learned from scratch. Among the tasks, (a) and (b) are sequence-level tasks while (c) and (d) are token-level tasks. In the figure, E represents the input embedding,  $T_i$  represents the contextual representation of token i, [CLS] is the special symbol for classification out-
> 
> put, and [SEP] is the special symbol to separate
> 
> non-consecutive token sequences.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Sentence Pair Classification Tasks: MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG
> 
> ![](_page_0_Figure_2.jpeg)
> 
> (c) Question Answering Tasks: SQuAD v1.1
> 
> ![](_page_0_Picture_4.jpeg)
> 
> (b) Single Sentence Classification Tasks: SST-2, CoLA
> 
> ![](_page_0_Figure_6.jpeg)
> 
> (d) Single Sentence Tagging Tasks: CoNLL-2003 NER
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2015.png)

### ICLR19：BigGAN（label2img）

ICLR19：LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS

BigGAN标准架构

> **[图片提取文字 (image.png)]:**
> et al., 2018), which is identical to that used by (Miyato et al., 2018), but with the channel pattern in D modified so that the number of filters in the first convolutional layer of each block is equal to the number of output filters (rather than the number of input filters, as in Miyato et al. (2018); Gulrajani et al. (2017). We use a single shared class embedding in G, and skip connections for the latent vector z (skip-z). In particular, we employ hierarchical latent spaces, so that the latent vector z is split along its channel dimension into chunks of equal size (20-D in our case), and each chunk is concatenated to the shared class embedding and passed to a corresponding residual block as a conditioning vector. The conditioning of each block is linearly projected to produce per-sample gains and biases for the BatchNorm layers of the block. The bias projections are zero-centered, while the gain projections are centered at 1. Since the number of residual blocks depends on the image resolution, the full dimensionality of z is 120 for  $128 \times 128$ , 140 for  $256 \times 256$ , and 160 for  $512 \times 512$  images.
> 
> In the BigGAN model (Figure 15), we use the ResNet (He et al., 2016) GAN architecture of (Zhang
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (b) A Residual Block (*ResBlock up*) in BigGAN's G. (c) A Residual Block (*ResBlock down*) in BigGAN's D.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2017.png)

BigGAN标准架构，**不同分辨率的输入有不同的ResBlk深度**

> **[图片提取文字 (image.png)]:**
> Table 4: BigGAN architecture for  $128 \times 128$  images. ch represents the channel width multiplier in each network from Table 1.  $z \in \mathbb{R}^{120} \sim \mathcal{N}(0, I)$ RGB image  $x \in \mathbb{R}^{128 \times 128 \times 3}$  $\text{Embed}(y) \in \mathbb{R}^{128}$ 
> 
> ResBlock down  $ch \rightarrow 2ch$ Linear  $(20 + 128) \rightarrow 4 \times 4 \times 16ch$ Non-Local Block  $(64 \times 64)$ ResBlock up  $16ch \rightarrow 16ch$ ResBlock down  $2ch \rightarrow 4ch$ ResBlock up  $16ch \rightarrow 8ch$ ResBlock down  $4ch \rightarrow 8ch$ ResBlock up  $8ch \rightarrow 4ch$ ResBlock down  $8ch \rightarrow 16ch$ ResBlock up  $4ch \rightarrow 2ch$ ResBlock down  $16ch \rightarrow 16ch$ Non-Local Block  $(64 \times 64)$ ResBlock  $16ch \rightarrow 16ch$ ResBlock up  $2ch \rightarrow ch$ ReLU, Global sum pooling
> 
> BN, ReLU,  $3 \times 3$  Conv  $ch \rightarrow 3$ Embed(y)·h + (linear  $\rightarrow$  1) Tanh (b) Discriminator Generator
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2018.png)

> **[图片提取文字 (image.png)]:**
> Table 6: BigGAN architecture for  $512 \times 512$  images. Relative to the  $256 \times 256$  architecture, we add an additional ResBlock at the  $512 \times 512$  resolution. Memory constraints force us to move the non-local block in both networks back to  $64 \times 64$  resolution as in the  $128 \times 128$  pixel setting.
> 
> | $z \in \mathbb{R}^{160} \sim \mathcal{N}(0, I)$<br>Embed $(y) \in \mathbb{R}^{128}$ | RGB image $x \in \mathbb{R}^{512 \times 512 \times 3}$                                  |
> |-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
> | $\frac{1}{\text{Linear} (20+128) \rightarrow 4 \times 4 \times 16ch}$               | ResBlock down $ch \to ch$                                                               |
> | ResBlock up $16ch \rightarrow 16ch$                                                 | ResBlock down $ch \rightarrow 2ch$                                                      |
> | ResBlock up $16ch \rightarrow 8ch$                                                  | ResBlock down $2ch \rightarrow 4ch$                                                     |
> | ResBlock up $8ch \rightarrow 8ch$                                                   | Non-Local Block (64 × 64)                                                               |
> | <u> </u>                                                                            | ResBlock down $4ch \rightarrow 8ch$                                                     |
> | ResBlock up $8ch \rightarrow 4ch$                                                   | ResBlock down $8ch \rightarrow 8ch$                                                     |
> | Non-Local Block (64 × 64)                                                           | ResBlock down $8ch \rightarrow 16ch$                                                    |
> | ResBlock up $4ch \rightarrow 2ch$                                                   | ResBlock down $16ch \rightarrow 16ch$                                                   |
> | ResBlock up $2ch \rightarrow ch$                                                    | $\frac{\text{ResBlock } 16ch \rightarrow 16ch}{\text{ResBlock } 16ch \rightarrow 16ch}$ |
> | ResBlock up $ch \to ch$                                                             |                                                                                         |
> | BN, ReLU, $3 \times 3$ Conv $ch \rightarrow 3$                                      | ReLU, Global sum pooling                                                                |
> | Tanh                                                                                | Embed $(y) \cdot h$ + (linear $\rightarrow 1$ )                                         |
> | (a) Generator                                                                       | (b) Discriminator                                                                       |
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2019.png)

> **[图片提取文字 (image.png)]:**
> Table 5: BigGAN architecture for  $256 \times 256$  images. Relative to the  $128 \times 128$  architecture, we add an additional ResBlock in each network at  $16 \times 16$  resolution, and move the non-local block in **G** to  $128 \times 128$  resolution. Memory constraints prevent us from moving the non-local block in **D**.
> 
> | $z \in \mathbb{R}^{140} \sim \mathcal{N}(0, I)$<br>Embed $(y) \in \mathbb{R}^{128}$           | RGB image $x \in \mathbb{R}^{256 \times 256 \times 3}$ |
> |-----------------------------------------------------------------------------------------------|--------------------------------------------------------|
> | $\frac{1}{\text{inear } (20+128) \to 4 \times 4 \times 16ch}$                                 | ResBlock down $ch \rightarrow 2ch$                     |
> | $\frac{\text{ResBlock up } 16ch \rightarrow 16ch}{\text{ResBlock up } 16ch \rightarrow 16ch}$ | ResBlock down $2ch \rightarrow 4ch$                    |
> | ResBlock up $16ch \rightarrow 10ch$                                                           | Non-Local Block (64 × 64)                              |
> | ResBlock up $8ch \rightarrow 8ch$                                                             | ResBlock down $4ch \rightarrow 8ch$                    |
> | ResBlock up $8ch \rightarrow 3ch$                                                             | ResBlock down $8ch \rightarrow 8ch$                    |
> | ResBlock up $4ch \rightarrow 2ch$                                                             | ResBlock down $8ch \rightarrow 16ch$                   |
> | Non-Local Block (128 $\times$ 128)                                                            | ResBlock down $16ch \rightarrow 16ch$                  |
> | ResBlock up $2ch \rightarrow ch$                                                              | ResBlock $16ch \rightarrow 16ch$                       |
> | BN, ReLU, $3 \times 3$ Conv $ch \rightarrow 3$                                                | ReLU, Global sum pooling                               |
> | Tanh                                                                                          | Embed(y)· $h$ + (linear $\rightarrow$ 1)               |
> | () =                                                                                          | (b) Discriminator                                      |
> | (a) Generator                                                                                 |                                                        |
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2020.png)

BigGAN-deep架构，对不同分辨率的输入有不同的ResBlk深度

> **[图片提取文字 (image.png)]:**
> two additional  $1 \times 1$  convolutions: the first reduces the number of channels by a factor of 4 before the more expensive  $3 \times 3$  convolutions; the second produces the required number of output channels. While BigGAN relies on  $1 \times 1$  convolutions in the skip connections whenever the number of channels needs to change, in BigGAN-deep we use a different strategy aimed at preserving identity throughout the skip connections. In G, where the number of channels needs to be reduced, we simply retain the first group of channels and drop the rest to produce the required number of channels. In D, where the number of channels should be increased, we pass the input channels unperturbed, and concatenate them with the remaining channels produced by a  $1 \times 1$  convolution. As far as the network configuration is concerned, the discriminator is an exact reflection of the generator. There are two blocks at each resolution (BigGAN uses one), and as a result BigGAN-deep is four times deeper than BigGAN. Despite their increased depth, the BigGAN-deep models have significantly
> 
> fewer parameters mainly due to the bottleneck structure of their residual blocks. For example, the
> 
> 128 × 128 BigGAN-deep G and D have 50.4M and 34.6M parameters respectively, while the corre-
> 
> sponding original BigGAN models have 70.4M and 88.0M parameters. All BigGAN-deep models
> 
> use attention at  $64 \times 64$  resolution, channel width multiplier ch = 128, and  $z \in \mathbb{R}^{128}$ .
> 
> The BigGAN-deep model (Figure 16) differs from BigGAN in several aspects. It uses a simpler vari-
> 
> ant of skip-z conditioning: instead of first splitting z into chunks, we concatenate the entire z with
> 
> the class embedding, and pass the resulting vector to each residual block through skip connections.
> 
> BigGAN-deep is based on residual blocks with bottlenecks (He et al., 2016), which incorporate
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2021.png)

> **[图片提取文字 (image.png)]:**
> Table 7: BigGAN-deep architecture for  $128 \times 128$  images.  $z \in \mathbb{R}^{128} \sim \mathcal{N}(0, I)$ RGB image  $x \in \mathbb{R}^{128 \times 128 \times 3}$  $\text{Embed}(y) \in \mathbb{R}^{128}$  $3 \times 3$  Conv  $3 \rightarrow ch$ Linear  $(128 + 128) \rightarrow 4 \times 4 \times 16ch$ 
> 
> ResBlock down  $ch \rightarrow 2ch$ 
> 
> ResBlock  $2ch \rightarrow 2ch$ 
> 
> Non-Local Block  $(64 \times 64)$ 
> 
> ResBlock down  $2ch \rightarrow 4ch$ 
> 
> ResBlock  $4ch \rightarrow 4ch$ 
> 
> ResBlock down  $4ch \rightarrow 8ch$ 
> 
> ResBlock  $8ch \rightarrow 8ch$ 
> 
> ResBlock down  $8ch \rightarrow 16ch$ 
> 
> ResBlock  $16ch \rightarrow 16ch$ 
> 
> ResBlock down  $16ch \rightarrow 16ch$ 
> 
> ResBlock  $16ch \rightarrow 16ch$ 
> 
> ReLU, Global sum pooling
> 
>  $\text{Embed}(y) \cdot \boldsymbol{h} + (\text{linear} \rightarrow 1)$ 
> 
> (b) Discriminator
> 
> ResBlock  $16ch \rightarrow 16ch$ ResBlock up  $16ch \rightarrow 16ch$ 
> 
> ResBlock  $16ch \rightarrow 16ch$ ResBlock up  $16ch \rightarrow 8ch$ 
> 
> ResBlock  $8ch \rightarrow 8ch$ ResBlock up  $8ch \rightarrow 4ch$ 
> 
> ResBlock  $4ch \rightarrow 4ch$ ResBlock up  $4ch \rightarrow 2ch$ 
> 
> ResBlock  $2ch \rightarrow 2ch$ 
> 
> Non-Local Block  $(64 \times 64)$ 
> 
> (a) Generator
> 
> ResBlock up  $2ch \rightarrow ch$ 
> 
> BN, ReLU,  $3 \times 3$  Conv  $ch \rightarrow 3$ 
> 
> Tanh
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2022.png)

> **[图片提取文字 (image.png)]:**
> Table 8: BigGAN-deep architecture for  $256 \times 256$  images.  $z \in \mathbb{R}^{128} \sim \mathcal{N}(0, I)$ RGB image  $x \in \mathbb{R}^{256 \times 256 \times 3}$  $\text{Embed}(y) \in \mathbb{R}^{128}$  $3 \times 3$  Conv  $3 \rightarrow ch$ Linear  $(128 + 128) \rightarrow 4 \times 4 \times 16ch$ ResBlock down  $ch \rightarrow 2ch$ ResBlock  $16ch \rightarrow 16ch$ ResBlock  $2ch \rightarrow 2ch$ ResBlock up  $16ch \rightarrow 16ch$ ResBlock down  $2ch \rightarrow 4ch$ 
> 
> $$\begin{array}{c}
> 16ch \\
>  \hline
>  16ch
> \end{array}$$
> 
> ResBlock 
> $$16ch \rightarrow 16ch$$
>   
> ResBlock up  $16ch \rightarrow 8ch$   
> ResBlock  $8ch \rightarrow 8ch$ 
> 
> ResBlock up  $8ch \rightarrow 8ch$ ResBlock  $8ch \rightarrow 8ch$ 
> 
> $$\begin{array}{c} ch \rightarrow 8ch \\ h \rightarrow 8ch \\ ch \rightarrow 4ch \\ k (64 \times 64) \end{array}$$
> 
> ResBlock up  $8ch \rightarrow 4ch$ Non-Local Block  $(64 \times 64)$ 
> 
> $$4ch \rightarrow 4ch$$
> \nup  $4ch \rightarrow 2ch$ 
>  $2ch \rightarrow 2ch$ 
> 
> ResBlock  $4ch \rightarrow 4ch$ ResBlock up  $4ch \rightarrow 2ch$ ResBlock  $2ch \rightarrow 2ch$ 
> 
> ResBlock up  $2ch \rightarrow ch$ 
> 
> BN, ReLU,  $3 \times 3$  Conv  $ch \rightarrow 3$ 
> 
> Tanh
> 
> (a) Generator
> 
> $$k \ 4ch \rightarrow 4ch$$
>   
>  $up \ 4ch \rightarrow 2ch$ 
> 
> $$h \to 4ch$$
>  $ch \to 2ch$ 
> 
> ResBlock down  $16ch \rightarrow 16ch$ 
> 
> ResBlock down 
> $$8ch \rightarrow 16ch$$
>   
> ResBlock  $16ch \rightarrow 16ch$ 
> 
> ResBlock  $16ch \rightarrow 16ch$ 
> 
> ReLU, Global sum pooling
> 
>  $\text{Embed}(y) \cdot \boldsymbol{h} + (\text{linear} \rightarrow 1)$ 
> 
> (b) Discriminator
> 
> ResBlock  $4ch \rightarrow 4ch$ 
> 
> Non-Local Block  $(64 \times 64)$ 
> 
> ResBlock down  $4ch \rightarrow 8ch$ 
> 
> ResBlock  $8ch \rightarrow 8ch$ 
> 
> ResBlock 
> $$8ch \rightarrow 8ch$$
>   
> Block down  $8ch \rightarrow 10$ 
> 
> ResBlock down 
> $$8ch \rightarrow 8ch$$
>   
> ResBlock  $8ch \rightarrow 8ch$ 
> 
> $$\frac{\rightarrow 8ch}{8ch}$$
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 16: (a) A typical architectural layout for BigGAN-deep's **G**; details are in the following tables. (b) A Residual Block (*ResBlock up*) in BigGAN-deep's **G**. (c) A Residual Block (*ResBlock down*) in BigGAN-deep's **D**. A *ResBlock* (without *up* or *down*) in BigGAN-deep does not include the *Upsample* or *Average Pooling* layers, and has identity skip connections.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2024.png)

> **[图片提取文字 (image.png)]:**
> Table 9: BigGAN-deep architecture for  $512 \times 512$  images.
> 
> | $z \in \mathbb{R}^{128} \sim \mathcal{N}(0, I)$<br>Embed $(y) \in \mathbb{R}^{128}$                            | RGB image $x \in \mathbb{R}^{512 \times 512 \times 3}$ |
> |----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
> |                                                                                                                | $3 \times 3 \text{ Conv } 3 \rightarrow ch$            |
> | $\frac{\text{Linear } (128 + 128) \rightarrow 4 \times 4 \times 16ch}{\text{ResBlock } 16ch \rightarrow 16ch}$ | ResBlock down $ch \rightarrow ch$                      |
> |                                                                                                                | ResBlock $ch \to ch$                                   |
> | $\frac{\text{ResBlock up } 16ch \rightarrow 16ch}{\text{ResBlock } 16ch \rightarrow 16ch}$                     | ResBlock down $ch \rightarrow 2ch$                     |
> |                                                                                                                | ResBlock $2ch \rightarrow 2ch$                         |
> | ResBlock up $16ch \rightarrow 8ch$                                                                             | ResBlock down $2ch \rightarrow 4ch$                    |
> | ResBlock $8ch \rightarrow 8ch$                                                                                 | ResBlock $4ch \rightarrow 4ch$                         |
> | ResBlock up $8ch \rightarrow 8ch$                                                                              | Non-Local Block (64 × 64)                              |
> | ResBlock $8ch \rightarrow 8ch$                                                                                 | ResBlock down $4ch \rightarrow 8ch$                    |
> | ResBlock up $8ch \rightarrow 4ch$                                                                              | ResBlock $8ch \rightarrow 8ch$                         |
> | Non-Local Block (64 × 64)                                                                                      | ResBlock down $8ch \rightarrow 8ch$                    |
> | ResBlock $4ch \rightarrow 4ch$                                                                                 | ResBlock $8ch \rightarrow 8ch$                         |
> | ResBlock up $4ch \rightarrow 2ch$                                                                              | ResBlock down $8ch \rightarrow 16ch$                   |
> | ResBlock $2ch \rightarrow 2ch$                                                                                 | ResBlock $16ch \rightarrow 16ch$                       |
> | ResBlock up $2ch \rightarrow ch$                                                                               | ResBlock down $16ch \rightarrow 16ch$                  |
> | ResBlock $ch \to ch$                                                                                           | ResBlock $16ch \rightarrow 16ch$                       |
> | ResBlock up $ch \rightarrow ch$                                                                                | ReLU, Global sum pooling                               |
> | BN, ReLU, $3 \times 3$ Conv $ch \rightarrow 3$                                                                 | Embed $(y) \cdot h + (linear \rightarrow 1)$           |
> | Tanh                                                                                                           | (b) Discriminator                                      |
> | (a) Generator                                                                                                  | (6) 2100111111111101                                   |
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2025.png)

### PMLR21：**Zero-Shot（**text2img**）**

PMLR21：**Zero-Shot Text-to-Image Generation**

text2img：**隐变量分布p(z, y)，条件分布和后验分布都以caption y和z的pair作为条件**，其他同VQVAE+GPT。

> **[图片提取文字 (image.png)]:**
> dure, similar to (Oord et al., 2017; Razavi et al., 2019):
> Stage 1. We train a discrete variational autoen-
> 
> coder (dVAE)<sup>1</sup> to compress each  $256 \times 256$  RGB image
> 
> without a large degradation in visual quality (see Fig-
> 
> We address these issues by using a two-stage training proce-
> 
> into a  $32 \times 32$  grid of image tokens, each element of which can assume 8192 possible values. This reduces the context size of the transformer by a factor of 192
> 
> https://github.com/openai/DALL-E
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2026.png)

> **[图片提取文字 (image.png)]:**
> • Stage 2. We concatenate up to 256 BPE-encoded text tokens with the  $32 \times 32 = 1024$  image tokens, and train an autoregressive transformer to model the joint
> 
> distribution over the text and image tokens.
> 
> ure 1).
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2027.png)

> **[图片提取文字 (image.png)]:**
> for the encoded RGB image. We model this distribution using the factorization  $p_{\theta,\psi}(x,y,z) = p_{\theta}(x\,|\,y,z)p_{\psi}(y,z)$ , which yields the lower bound  $\ln p_{\theta,\psi}(x,y) \geqslant \mathop{\mathbb{E}}_{z \sim q_{\phi}(z\,|\,x)} \left( \ln p_{\theta}(x\,|\,y,z) - \frac{1}{2} \right) = \frac{1}{2} \sum_{z \sim q_{\phi}(z\,|\,x)} \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1}{2} \right) \left( \frac{1$ 
> 
>  $\beta D_{\mathrm{KL}}(q_{\phi}(y,z\,|\,x),p_{\psi}(y,z))),$ 
> 
> (1)
> 
> The overall procedure can be viewed as maximizing the
> 
> evidence lower bound (ELB) (Kingma & Welling, 2013;
> 
> Rezende et al., 2014) on the joint likelihood of the model
> 
> distribution over images x, captions y, and the tokens z
> 
> •  $q_{\phi}$  denotes the distribution over the 32 × 32 image tokens generated by the dVAE encoder given the RGB image  $x^2$ ;
> 
> where:
> 
> •  $p_{\theta}$  denotes the distribution over the RGB images generated by the dVAE decoder given the image tokens; and
> 
> •  $p_{\psi}$  denotes the joint distribution over the text and image tokens modeled by the transformer.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3. Comparison of samples from our model to those from prior approaches on captions from MS-COCO. Each of our model samples is the best of 512 as ranked by the contrastive model. We do not use any manual cherrypicking with the selection of either the captions or the samples from any of the models.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> operations for backpropagation. We scale the incoming gradient for each resblock by its gradient scale, and unscale the outgoing gradient before it is added to the sum of the gradients from the successive resblocks. The activations and gradients along the identity path are stored in 32-bit precision. The "filter" operation sets all Inf and NaN values in the activation gradient to zero. Without this, a nonfinite event in the current resblock would cause the gradient scales for all preceding resblocks to unnecessarily drop, thereby resulting in underflow.
> 
> Figure 4. Illustration of per-resblock gradient scaling for a trans-
> 
> former resblock. The solid line indicates the sequence of opera-
> 
> tions for forward propagation, and the dashed line the sequence of
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Each box denotes a vector of size  $d_{\text{model}} = 3968$ . In this illustration, the caption has a length of 4 tokens, so 2 padding tokens are used (as described in Section 2.2). Each image vocabulary embedding is summed with a row and column embedding.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> 6 tokens and image length of 16 tokens (i.e., corresponding to a  $4 \times 4$  grid). Mask (a) corresponds to row attention in which each image token attends to the previous 5 image tokens in raster order. The extent is chosen to be 5, so that the last token being attended to is the one in the same column of the previous row. To obtain better GPU utilization, we transpose the row and column dimensions of the image states when applying column attention, so that we can use mask (c) instead of mask (b). Mask (d) corresponds to a causal convolutional attention pattern with wraparound behavior (similar to the row attention) and a  $3 \times 3$  kernel. Our model uses a mask corresponding to an  $11 \times 11$  kernel.
> 
> Figure 11. Illustration of the three types of attention masks for a hypothetical version of our transformer with a maximum text length of
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2032.png)

### NIPS20：**Net2Net Translation（BERT+BigGAN）**

NIPS20：**Network-to-Network Translation with Conditional Invertible Neural Networks**

*motivation*

**现有模型协作**是对特定domain重新训练或微调：GPT辅助VQVAE进行文生图，VQVAE学习图片编码和解码，编码是patch的空间序列，GPT根据VQVAE的编码案例，学习生成patch序列。

现有多模态模型相比单模态的训练开销显著增大，而单模态如GPT的训练开销已经耗尽几乎所有资源。为每个domain训练新的模型开销过大，因此需要寻找新的“**计算方法**”（Bitter Lesson）。

**Bitter Lesson**：寻找**合适的数据计算方法**来构建AI，相比基于人类知识构建AI更有用：如NLP中基于人类语言学知识构建的算法，效果不如基于隐马尔可夫链和统计学构建DNN来“计算”数据；如CV中基于人类认知中图像特征（边缘、SIFT特征等）构建的算法，效果不如基于卷积和其变体构建DNN来“计算“数据。

> **[图片提取文字 (image.png)]:**
> diverse domains and modalities [12, 73, 65, 68, 61]. In contrast, artificial intelligence research has made great progress in learning powerful representations for *individual* domains [28, 71, 19, 69, 15, 5] that can even achieve superhuman performance on confined tasks such as traffic sign recognition [10, 11], image classification [29] or question answering [15]. However, learning representations for different domains that also allow a domain-to-domain transfer of information between them is significantly more challenging [2]: There is a trade-off between the expressiveness of individual domain representations and their compatibility to another to support transfer. While for limited training data multimodal learning has successfully trained representations for different domains together [66, 74], the overall most powerful domain-specific representations typically result from training huge models specifically for *individual* challenging domains using massive amounts of training data and computational resources, e.g. [19, 69, 5]. With the dawn of even more massive models like the recently introduced GPT-3 [5], where training on only a single domain already demands most of the available resources, we must find new, creative ways to make use of these powerful models, which none but the largest institutions can afford to train and experiment with,
> 
> and thereby utilize the huge amount of resources and knowledge which are distilled into the model's
> 
> representations—in other words, we have to find ways to cope with "The Bitter Lesson" [67].
> 
> One of the key features of intelligence is the ability to combine and transfer information between
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2033.png)

训练**trade-off**：增强自身**domain**的表达能力 vs. 向其他domain模型的**信息迁移能力**

**论文**认为不同domain模型的输入和输出是**”统一概念“的不同隐变量表达**，相同概念的不同隐变量表达可以**相互转换（Domain Transfer）**。基于合适的Domain Transfer方法，模型能够自由拼接。

*多模型协作模式*

Model A过程中输出的”概念”，通过domain transfer方法**“对齐”**到Model B过程中输入的“概念”，让训练好的模型类似“插件”自由拼接，而**不需要重新端到端训练，只需要训练domain transfer方法**。

文本生成模型(BERT)输出**图像的文本隐变量表示**，图像生成模型(BigGAN)输入**图像的编码隐变量表示**，BERT输出和BigGAN输入是图像内容物”概念“的不同隐变量表达。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: *BERT* [15] to *BigGAN* [4] transfer: Our approach enables translation between fixed off-the-shelve expert models such as BERT and BigGAN without having to modify or finetune them.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: SBERT architecture with classification objective function, e.g., for fine-tuning on SNLI dataset. The two BERT networks have tied weights (siamese network structure).
> 
> Figure 2: SBERT architecture at inference, for example, to compute similarity scores. This architecture is also used with the regression objective function.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2035.png)

 

*Domain Transfer方法*

BackBone A + Transfer + Head B架构中，zΦ是BackBone输出“概念”的隐变量表达，zΘ是Head输入“概念”的隐变量表达，模型A和B对**相同“概念”的理解相同**，找到**转换隐变量表达z**的方法。

相同概念在任一domain中存在多种表达z，所以zΦ和zΘ的关系由**联合分布p(zΘ, zΦ)**表达，zΦ到zΘ的映射是**一对多的条件分布**p(zΘ | zΦ)，引入随机变量v，建立**zΦ到zΘ的一一映射zΘ=τ(v | zΦ)**。

> **[图片提取文字 (image.png)]:**
> f(x) denotes an expert model that has been trained to map  $x \in \mathcal{D}_x$  onto desired outputs, e.g. class
> 
> labels in case of classification tasks, or synthesized images for generative image models. To solve
> 
> its task, a neural network f has learned a latent representation  $z_{\Phi} = \Phi(x)$  of domain  $\mathcal{D}_x$  in some
> 
> intermediate layer, so that subsequent layers  $\Psi$  can then solve the task as  $f(x) = \Psi(\Phi(x))$ . For
> 
>  $y \in \mathcal{D}_y$  let  $g(y) = \Lambda(\Theta(y))$  be another, totally different model that provides a feature representation
> 
> vector  $z_{\Theta} = \Theta(y)$ .
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2036.png)

> **[图片提取文字 (image.png)]:**
> textual descriptions are conceivable for the same image by focusing on different aspects. This implies a non-unique mapping from  $z_{\Phi}$  to  $z_{\Theta}$ . Moreover, much of the power of model f trained for a specific task stems from its ability to ignore task-irrelevant properties of x. The invariances of  $z_{\Phi}$  with respect to  $z_{\Theta}$  further increase the ambiguity of the domain translation. Obtaining a plausible  $z_{\Theta}$  for a given  $z_{\Phi}$  is therefore best described probabilistically as sampling from  $p(z_{\Theta}|z_{\Phi})$ . Our goal is to model this
> 
> In general, we cannot expect a translation from x to y to be unique, since two arbitrary domains and
> 
> their representations are not necessarily isomorphic. For example, a textual description x of an image
> 
> y usually leaves many details open and the same holds in the opposite direction, since many different
> 
> process with a translation function  $\tau$ . Thus, we must introduce a residual v, such that for a given  $z_{\Phi}$ ,
> 
>  $z_{\Theta} = \tau(v|z_{\Phi})$ 
> 
> v uniquely determines  $z_{\Theta}$  resulting in the translation function  $\tau$ :
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Proposed architecture. We provide post-hoc model fusion for two given deep networks  $f = \Phi \circ \Psi$  and  $g = \Theta \circ \Lambda$  which live on arbitrary domains  $\mathcal{D}_x$  and  $\mathcal{D}_y$ . For deep representations  $z_{\Phi} = \Phi(x)$  and  $z_{\Theta} = \Theta(y)$ , a conditional INN  $\tau$  learns to transfer between them by modelling the ambiguities w.r.t. the translation as an explicit residual, enabling transfer between given off-the-shelf models and their respective domains.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2038.png)

采样v表达**“概念”已知**条件下的z表达的不确定性，所以src-domain中zΦ表达的“概念”在tar-domain中的表达zΘ，是采样v的函数，但是**v不包含zΦ所表达“概念”的信息**。

因此在zΘ=τ(v | zΦ)或v=τ-1(zΘ | zΦ)定义的关系中，**v和zΦ相互无关** ，即以zΦ为条件的条件分布**p(v | zΦ)近似v的先验分布q(v)**。

> **[图片提取文字 (image.png)]:**
> **Learning a Domain Translation**  $\tau$ : How can we estimate v? v must capture all information of  $z_{\Theta}$  not represented in  $z_{\Phi}$ , but no information that is already represented in  $z_{\Phi}$ . Hence, to infer v, we must take into account both  $z_{\Theta}$ , to extract information, and  $z_{\Phi}$ , to discard information. The unique determination of  $z_{\Theta}$  from v for a given  $z_{\Phi}$  implies the existence of the inverse of  $\tau$ , when considered as a function of v. Thus for every  $z_{\Phi}$ , the inverse  $\tau^{-1}(\cdot|z_{\Phi})$  of  $\tau(\cdot|z_{\Phi})$  exists,  $v = \tau^{-1}(z_{\Theta}|z_{\Phi}). \tag{2}$ 
> 
> This structure of 
> $$\tau$$
>  is most naturally represented by a conditionally invertible neural network (cINN), for which  $\tau^{-1}$  can be explicitly computed, and which we build from affine coupling [17], actnorm [35] and shuffling layers, see Sec. [G.1] It then remains to derive a learning task which ensures that
> 
> [35] and shuffling layers, see Sec. [G.1] It then remains to derive a learning task which ensures that information of  $z_{\Phi}$  is discarded in v. To formalize this goal, we consider training pairs  $\{(x,y)\} \subset \mathcal{D}_x \times \mathcal{D}_y$  and their corresponding features  $\{(z_{\Phi}, z_{\Theta})\}$  as samples from their joint distribution  $p(z_{\Phi}, z_{\Theta})$ . v can then be considered as a random variable via the process  $v = \tau^{-1}(z_{\Theta}|z_{\Phi}), \quad \text{with } z_{\Phi}, z_{\Theta} \sim p(z_{\Phi}, z_{\Theta}). \tag{3}$ 
> 
> Then 
> $$v$$
>  discards all information of  $z_{\Phi}$  if  $v$  and  $z_{\Phi}$  are independent. To achieve this independence, we minimize the distance between the distribution  $p(v|z_{\Phi})$  induced by  $\tau$  via Eq. (3) and some prior distribution  $q(v)$ . The latter can be chosen arbitrarily as long as it is independent of  $z_{\Phi}$ , its density can be evaluated and samples can be drawn. In practice we use a standard normal distribution. Using the
> 
> invertibility of  $\tau$ , we can then explicitly calculate the Kullback-Leibler divergence between  $p(v|z_{\Phi})$ 
> 
> and q(v) averaged over  $z_{\Phi}$  (see Sec. B for the derivation):
> 
>  $\mathbb{E}_{z_{\Phi}} \operatorname{KL}(p(v|z_{\Phi})|q(v)) = \mathbb{E}_{z_{\Theta},z_{\Phi}} \left\{ -\log q(\tau^{-1}(z_{\Theta}|z_{\Phi})) - |\det J_{\tau^{-1}}(z_{\Theta}|z_{\Phi})| \right\} - H(z_{\Theta}|z_{\Phi}). \tag{4}$ 
> 
> Here, det  $J_{\tau^{-1}}$  denotes the determinant of the Jacobian of  $\tau^{-1}$  and H is the (constant) data entropy.
> 
> If  $\tau$  minimizes Eq. (4), we have  $p(v|z_{\Phi})=q(v)$ , such that the desired independence is achieved. Moreover, we can now simply achieve the original goal of sampling from  $p(z_{\Theta}|z_{\Phi})$  by translating from  $z_{\Phi}$  to  $z_{\Theta}=\tau(v|z_{\Phi})$  with v sampled from q(v), which properly models the inherent ambiguity.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2039.png)

domain transfer函数τ是：**zΦ作为“概念”已知**的条件下，zΘ和残差采样v之间的映射。

> **[图片提取文字 (image.png)]:**
> Domain Transfer Between Fixed Models: At inference time, we obtain translated samples  $z_{\Theta}$  for given  $z_{\Phi}$  by sampling from the residual space v given  $z_{\Phi}$  and then applying  $\tau$ ,  $z_{\Theta} \sim p(z_{\Theta}|z_{\Phi}) \iff v \sim q(v), \ z_{\Theta} = \tau(v|z_{\Phi}). \tag{7}$ 
> 
> After training our domain translator, transfer between  $\mathcal{D}_x$  and  $\mathcal{D}_y$  is thus achieved by the following steps: (i) sample x from p(x), (ii) encode x into the latent space  $z_{\Phi} = \Phi(x)$  of expert model f, (iii) sample a residual v from the prior q(v), (iv) conditionally transform  $z_{\Theta} = \tau(v|z_{\Phi})$ , and (v) decode  $z_{\Theta}$  into the domain  $\mathcal{D}_y$  of the second expert model:  $y = \Lambda(z_{\Theta})$ .
> 
> Note that this approach has multiple advantages: (i) hidden representations usually have lower dimensionality than x, which makes transfer between arbitrary complex domains affordable, (ii) the cINN  $\tau$  can be trained by minimizing the negative log-likelihood, independent of the domains  $\mathcal{D}_x$  and  $\mathcal{D}_y$ , and (iii) the approach does not require to take any gradients w.r.t. the expert models f and g.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2040.png)

*基于Transfer Model的多模型协作实验*

BERT+Transfer+BigGAN中，**训练Transfer Model而不用微调或重新训练BERT和BigGAN**。

不同src domain和不同tar domain的模型组合，完成不同目标任务。

text2img：BERT（BackBone）+ BigGAN（Head），BackBone输出Caption上下文，迁移到BigGAN输入的图像编码。 

segment2img（3a）：segmentation NN（BackBone）+ 自编码器（Head），BackBone输出语义分割的**结果**，迁移到自编码器输入的图像编码。

segment2img（3b）：segmentation NN（BackBone）+自编码器（Head），BackBone输出语义分割的**logit分布**，迁移到自编码器输入的图像编码，说明logit去除对语义分割无用的信息（颜色）。

edge2img（3c）：Sobel Filter（edge x）+ ResNet（BackBone）+ 自编码器（Head），BackBone输出**edge图像的识别结果**，迁移到自编码器输入的图像编码。

img inpainting（3d）：（masked x）+ ResNet（BackBone）+ 自编码器（Head），BackBone输出**被遮挡图像的识别结果**，迁移到自编码器输入的图像编码。

> **[图片提取文字 (image.png)]:**
> ## 4.1 Translation to BigGAN
> 
> dataset. As most GAN frameworks in general and BigGAN in particular do not include an explicit encoder into a latent space, we aim to provide an encoding from an arbitrary domain by using an appropriate expert model f. Aiming at the reusability of a fixed BigGAN g, and given the hidden representation  $z_{\Phi} = \Phi(x)$  of the expert model  $f = \Psi \circ \Phi$ , we want to find a mapping between  $z_{\Phi}$  and the latent space  $z_{\Theta}$  of BigGAN's generator  $\Lambda$ , where, in accordance with Fig.  $2 \Theta = 1$  and  $2 \Theta = 1$  and  $2 \Theta = 1$ . Technical details regarding the training of our cINN can be found in Sec.  $2 \Theta = 1$  as led to an immense leap in the field of natural language processing, where a popular model is the so-called
> 
> This section is dedicated to the task of using a popular but computationally expensive to train expert
> 
> model as an image generator: BigGAN [4], achieving state-of-the-art FID scores [31] on the ImageNet
> 
> BERT model. Here, we make use of a variant of the original model, which modifies BERT such that it produces a latent space in which input sentences can be compared for similarity via the cosine-distance measure [57]. We aim to combine this representational power with the synthesis capabilities of BigGAN and thus train our model  $\tau$  to map from the language representations  $z_{\Phi} = \Phi(x)$  into the latent space  $z_{\Theta}$  of BigGAN's generator as described above; hence  $f = \Phi$  and  $\Psi = 1$ . During training, access to textual descriptions is obtained by using a captioning model as in [77], trained on the COCO [40] dataset. In a nutshell, at training time, we sample  $z_{\Theta}$ , produce a corresponding image  $\Lambda(z_{\Theta})$ , utilize [77] to produce a text-caption x describing the image and subsequently produce
> 
> a sentence representation  $z_{\Phi} = \Phi(x)$  which we use to minimize the overall objective Eq. (4). Results
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2041.png)

> **[图片提取文字 (image.png)]:**
> | Table 1: Inception and                                                                                         | d FID scores fo | r BERT-to-Big | gGAN transfer o | on captions fror | n COCO-stuff. C | Our approach |  |  |  |  |
> |----------------------------------------------------------------------------------------------------------------|-----------------|---------------|-----------------|------------------|-----------------|--------------|--|--|--|--|
> | is on-par with the current state of the art but does not require training of a text-encoder and image-decoder. |                 |               |                 |                  |                 |              |  |  |  |  |
> |                                                                                                                | CD CAN TO       | A. CANTE      | C. LCANION      | DM CANION        | M: CANUED       | IID CAN IOC  |  |  |  |  |
> 
> |      | our            | SD-GAN [79]    | AttnGAN [78]   | StackGAN 82   | DM-GAN 88      | MirrorGAN [52] | HDGAN 86       |
> |------|----------------|----------------|----------------|---------------|----------------|----------------|----------------|
> | IS ↑ | $34.7 \pm 0.3$ | $35.7 \pm 0.5$ | $25.9 \pm 0.5$ | $8.5 \pm 0.1$ | $30.5 \pm 0.6$ | $26.5 \pm 0.4$ | $11.9 \pm 0.2$ |
> 
> 32.64
> 
> 35.49
> 
>  $FID \downarrow$ 
> 
> 30.63
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2042.png)

> **[图片提取文字 (image.png)]:**
> can be found in Fig. 1 and Tab. 1 Our model captures both fine-grained and coarse descriptions (e.g. blue bird vs. yellow bird; school bus vs. pizza) and is able to synthesize images with highly different content, based on given textual inputs x. Although not being trained on the COCO images, Tab. I shows that our model is highly competitive and on-par with the state-of-the art in terms of Inception [59] and FID [31] scores where available.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2043.png)

> **[图片提取文字 (image.png)]:**
> ## 4.2 Repurposing a single target generator for different source domain models
> 
> Here, we train the cINN  $\tau$  conditioned on hidden representations of networks such as classifiers and segmentation models, and thereby show that standard classifiers on arbitrary source domains can drive the same generator to create content by transfer. Referring to Fig. 2 this means that f is represented by a classifier/segmentation model, whereas  $\Lambda$  is a decoder of an autoencoder that is pretrained on a dataset of interest. Furthermore, we evaluate the ability of our approach to combine a single, powerful domain expert (the autoencoder) with different source models to solve a variety of image-to-image translation tasks. The autoencoder is trained on a combination of all carnivorous animal classes in ImageNet and images of the AwA2 dataset 75, split into 211306 training images and 10000 testing images, which we call the Animals dataset. The details regarding architecture and training of this autoencoder are provided in Sec.  $\boxed{F}$ .
> 
> models  $\Phi$  onto the same generator  $\Lambda$  using our cINN  $\tau$ . In Fig. 3a f is a segmentation network trained on COCOStuff, and  $\Phi = f$ , i.e.  $z_{\Phi}$  is given by the final segmentation output of the network. This case corresponds to a translation from segmentation masks to images and we observe that our approach can successfully fuse the segmentation model with the autoencoder to obtain a wide variety of generated image samples corresponding to a given segmentation mask. Fig. 3b uses the same segmentation network for f, but this time,  $\Phi$  are the logit predictions of the network (visualized by a projection to RGB values). The diversity of generated samples is greatly reduced compared to
> 
> Fig. 3a, which indicates that logits still contain a lot of information which are not strictly required for
> 
> segmentation, e.g. the color of animals. This shows how different layers of an expert can be selected
> 
> to obtain more control over the synthesis process.
> 
> **Image-to-Image Translation:** In Fig. 3, we investigate the translation from different source domain
> 
> In Fig. 3c we consider the task of translating edge images to natural images. Here, x is obtained through the Sobel filter, and we choose a ResNet pretrained for image classification on stylized ImageNet as a domain expert for edge images, as it has shown sensitivity to shapes 25. This combination of  $\Phi$  and  $\Lambda$  through  $\tau$  enables edge-to-image translation. Fig. 3d shows an image
> 
> combination of  $\Phi$  and  $\Lambda$  through  $\tau$  enables edge-to-image translation. Fig. 3d shows an image inpainting task, where x is a masked image. In this case, large portions of the shape are missing from the image but the unmasked regions contain texture patches. This makes a ResNet pretrained
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2044.png)

> **[图片提取文字 (image.png)]:**
> for image classification on ImageNet a suitable domain expert due to its texture bias. The samples demonstrate that textures are indeed faithfully preserved.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> transferred  $y = \Lambda(\tau(v|\Phi(x)))$ input x
> 
> ![](_page_0_Figure_2.jpeg)
> 
> (c) Edge-to-Image using stylized ResNet classifier.
> 
> ![](_page_0_Figure_4.jpeg)
> 
> transferred  $y = \Lambda(\tau(v|\Phi(x)))$ input x
> 
> (d) Inpainting using vanilla ResNet classifier.
> 
> Figure 3: Different Image-to-Image translation tasks solved with a single AE g and different experts f.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2046.png)

超分辨率重建：低分辨率自编码器的**编码结果**，**迁移**到高分辨率自编码器**解码输入**。

> **[图片提取文字 (image.png)]:**
> Furthermore, we can employ the same approach for generative superresolution. Fig. A shows the
> 
> resulting transfer when using our method for combining two autoencoders, which are trained on
> 
> different scales. More precisely, f is an autoencoder trained on images of size  $32 \times 32$ , while q is an
> 
> autoencoder of  $256 \times 256$  images. The samples show that the model captures the ambiguities w.r.t.
> 
> this translation and thereby enables efficient superresolution.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> to  $256 \times 256$  Animalfaces
> 
>  $16 \times 16$ 
> 
> Figure 4: Superresolution with Network-to-Network Translation. Here, we use our cINN to combine two autoencoders f and q to generatively combine two autoencoders living on image scales  $32 \times 32$  and  $256 \times 256$ .
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2048.png)

## CVPR22、NIPS21：Stable Diffusion（LDM）

CVPR22：High-Resolution Image Synthesis with Latent Diffusion Models

NIPS21：Score-based Generative Modeling in Latent Space

### LDM的动机

pixel空间的DM有高昂的训练开销和较慢的推理速度，LDM将**pixel space映射到latent space**后训练和推理，处理训练难推理慢的问题。

生成式模型一般方法是：

生成式对抗（GAN）**难以优化且训练过程不稳定**，难以捕捉完整的数据分布，出现**模式崩溃 (mode-collapse)** 现象，“结构”上生成空白；

变分自编码（VAE）**生成样本的质量通常不如GANs**，存在Blurry和Posteior Collapse问题；

自回归模型（ARM）对计算资源需求极高。串行采样，**推理速度缓慢**。如果不结合**压缩技术（编码，VQVAE+ARM）**，通常仅限于低分辨率图像；

扩散模型（Diffusion）在**pixel空间**操作，训练消耗巨大，早期会花费大量参数来建模**感知上不重要的细节**（如高频噪声），而这些细节对语义理解贡献较少，使用视觉Transformer (ViT) 架构直接处理像素时，随着维度的增加，**模型性能会迅速且灾难性地下降**。

> **[图片提取文字 (image.png)]:**
> Generative Models for Image Synthesis The high dimensional nature of images presents distinct challenges to generative modeling. Generative Adversarial Networks (GAN) [26] allow for efficient sampling of high resolution images with good perceptual quality [3, 41], but are diffi-
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2049.png)

> **[图片提取文字 (image.png)]:**
> cult to optimize [2, 27, 53] and struggle to capture the full data distribution [54]. In contrast, likelihood-based methods emphasize good density estimation which renders optimization more well-behaved. Variational autoencoders (VAE) [45] and flow-based models [18, 19] enable efficient synthesis of high resolution images [9, 43, 89], but sample quality is not on par with GANs. While autoregressive models (ARM) [6, 10, 91, 92] achieve strong performance in density estimation, computationally demanding architectures [94] and a sequential sampling process limit them to low resolution images. Because pixel based representations of images contain barely perceptible, high-frequency details [16,71], maximum-likelihood training spends a disproportionate amount of capacity on modeling them, resulting in long training times. To scale to higher resolutions, several two-stage approaches [23,65,97,99] use ARMs to model a compressed latent image space instead of raw pixels.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2050.png)

> **[图片提取文字 (image.png)]:**
> have achieved state-of-the-art results in density estimation [44] as well as in sample quality [15]. The generative power of these models stems from a natural fit to the inductive biases of image-like data when their underlying neural backbone is implemented as a UNet [15, 29, 69, 82]. The best synthesis quality is usually achieved when a reweighted objective [29] is used for training. In this case, the DM corresponds to a lossy compressor and allow to trade image quality for compression capabilities. Evaluating and optimizing these models in pixel space, however, has the downside of low inference speed and very high training costs. While the former can be partially adressed by advanced sampling strategies [46, 73, 81] and hierarchical approaches [30, 90], training on high-resolution image data always requires to calculate expensive gradients. We adress both drawbacks with our proposed LDMs, which work on a compressed latent space of lower dimensionality. This renders training
> 
> computationally cheaper and speeds up inference with al-
> 
> most no reduction in synthesis quality (see Fig. 1).
> 
> Recently, **Diffusion Probabilistic Models** (DM) [79],
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2051.png)

**高分辨**生成模型通过**2-stage方法（VQVAE+ARM）**组合模型，提高效率和性能：

VQVAE在stage 2采用**ARM Generator**学习生成图像的texture sequence（prior）；

VQGAN将VQVAE中stage-1中Compressor改为感知特征的距离**LPIPS**，并增加判别器**对抗**学习；

Net2Net**迁移**不同domain的“概念”表达来**拼接模型**；

Zero-Shot采用VAVAE+ARM架构对**caption-图片的pair**进行编码和解码生成。

**VQVAE+ARM的trade-off：**学习高压缩率的编码序列需要大量的参数，因为每个编码的图像信息很多，学习低压缩率的编码序列需要大量的计算（串行计算），延迟很高。

> **[图片提取文字 (image.png)]:**
> ings of individual generative approaches, a lot of research [11,23,65,68,97,99] has gone into combining the strengths of different methods into more efficient and performant models via a two stage approach. VQ-VAEs [65, 97] use autoregressive models to learn an expressive prior over a discretized latent space. [64] extend this approach to textto-image generation by learning a joint distributation over discretized image and text representations. More generally, [68] uses conditionally invertible networks to provide a generic transfer between latent spaces of diverse domains. Different from VQ-VAEs, VQGANs [23, 99] employ a first stage with an adversarial and perceptual objective to scale autoregressive transformers to larger images. However, the high compression rates required for feasible ARM training, which introduces billions of trainable pa-
> 
> rameters [23, 64], limit the overall performance of such ap-
> 
> Two-Stage Image Synthesis To mitigate the shortcom-
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2052.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> lent inductive biases for spatial data, we do not need the heavy spatial downsampling of related generative models in latent space, but can still greatly reduce the dimensionality of the data via suitable autoencoding models, see Sec. 3. Images are from the DIV2K [1] validation set, evaluated at  $512^2$  px. We denote the spatial downsampling factor by f. Reconstruction FIDs [28] and PSNR are calculated on ImageNet-val. [12]; see also Tab. 8.
> 
> Figure 1. Boosting the upper bound on achievable quality with
> 
> less agressive downsampling. Since diffusion models offer excel-
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2053.png)

之前的Latent Space Diffusion Denoising方法需要考虑VAE和Denoising的训练优先级，确保Encoder输出分布和Denoising输入分布接近（domain transfer）。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> from latent to data space using a decoder  $p(\mathbf{x}|\mathbf{z}_0)$ . The model is trained end-to-end. increased easily without sacrifices. Moreover, SDE-based generative models are currently defined for continuous data and cannot be applied effortlessly to binary, categorical, or graph-structured data.
> 
> distribution  $p(\mathbf{z}_1)$  and generates samples in latent space via denoising  $(\mathbf{z}_0 \leftarrow \mathbf{z}_1)$ . Then, the samples are mapped
> 
> Here, we propose the *Latent Score-based Generative Model* (LSGM), a new approach for learning SGMs in latent space, leveraging a variational autoencoder (VAE) framework [14, 15]. We map the input data to latent space and apply the score-based generative model there. The score-based model is then tasked with modeling the distribution over the embeddings of the data set. Novel data synthesis
> 
> iterative denoising, and then transforming this embedding via a decoder to data space (see Fig. 1). We can consider this model a VAE with an SGM prior. Our approach has several key advantages:
> 
> is achieved by first generating embeddings via drawing from a simple base distribution followed by
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2054.png)

> **[图片提取文字 (image.png)]:**
> ## Training with Different Weighting Mechanisms
> 
> The weighting term w(t) in Eq. 7 trains the prior with maximum likelihood. Similar to [1, 2], we observe that when w(t) is dropped
> 
> Mechanism Weights Weighted  $w_{ll}(t) = g(t)^2/\sigma_t^2$ Unweighted  $w_{\rm un}(t) = 1$ Reweighted  $w_{\rm re}(t) = g(t)^2$ 
> 
> Table 1: Weighting mechanisms
> 
> while training the SGM prior (i.e., w(t) = 1), LSGM often yields higher quality samples at a small cost in likelihood. However, in our case, we can only drop the weighting when training the prior. When updating the encoder parameters, we still need to use the maximum likelihood weighting to ensure
> 
> that the encoder  $q(\mathbf{z}_0|\mathbf{x})$  is brought closer to the true posterior  $p(\mathbf{z}_0|\mathbf{x})^4$ . Tab. 1 summarizes three weighting mechanisms we consider in this paper:  $w_{\rm ll}(t)$  corresponds to maximum likelihood,  $w_{\rm un}(t)$ is the unweighted objective used by [1, 2], and  $w_{\rm re}(t)$  is a variant obtained by dropping only  $1/\sigma_t^2$ . This weighting mechanism has a similar affect on the sample quality as  $w_{\rm un}(t)=1$ ; however, in
> 
> Sec. 3.4, we show that it is easier to define a variance reduction scheme for this weighting mechanism.
> 
> The following summarizes our training objectives (with  $t \sim \mathcal{U}[0, 1]$  and  $\epsilon \sim \mathcal{N}(\epsilon; \mathbf{0}, \mathbf{I})$ ):
> 
> $$\min_{\boldsymbol{\phi}, \boldsymbol{\psi}} \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}_{0}|\mathbf{x})} \left[ -\log p_{\boldsymbol{\psi}}(\mathbf{x}|\mathbf{z}_{0}) \right] + \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}_{0}|\mathbf{x})} \left[ \log q_{\boldsymbol{\phi}}(\mathbf{z}_{0}|\mathbf{x}) \right] + \mathbb{E}_{t, \boldsymbol{\epsilon}, q(\mathbf{z}_{t}|\mathbf{z}_{0}), q_{\boldsymbol{\phi}}(\mathbf{z}_{0}|\mathbf{x})} \left[ \frac{w_{\mathbb{I}}(t)}{2} ||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{z}_{t}, t)||_{2}^{2} \right]$$
> (8)
> 
> $$\min_{\boldsymbol{\theta}} \mathbb{E}_{t,\epsilon,q(\mathbf{z}_t|\mathbf{z}_0),q_{\boldsymbol{\phi}}(\mathbf{z}_0|\mathbf{x})} \left[ \frac{w_{\text{ll/un/re}}(t)}{2} || \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t,t) ||_2^2 \right] \quad \text{with} \quad q(\mathbf{z}_t|\mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \boldsymbol{\mu}_t(\mathbf{z}_0), \sigma_t^2 \mathbf{I}), \tag{9}$$
> 
> where Eq. 8 trains the VAE encoder and decoder parameters  $\{\phi, \psi\}$  using the variational bound  $\mathcal{L}(\mathbf{x}, \phi, \theta, \psi)$  from Eq. 6. Eq. 9 trains the prior with one of the three weighting mechanisms. Since the SGM prior participates in the objective only in the cross entropy term, we only consider this term when training the prior. Efficient algorithms for training with the objectives are presented in App. G.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2055.png)

### VQGAN+Conv-UNet

**LDM的改进**：使用Conv-UNet适当伸缩到高维的图像latent space，Conv-UNet+VQGAN端到端训练**调整感知压缩表达中的信息量**。

LDM的压缩方式：**VQGAN**（感知Loss、对抗学习、patch分块）。

> **[图片提取文字 (image.png)]:**
> proaches and less compression comes at the price of high computational cost [23, 64]. Our work prevents such tradeoffs, as our proposed LDMs scale more gently to higher dimensional latent spaces due to their convolutional backbone. Thus, we are free to choose the level of compression which optimally mediates between learning a powerful first stage, without leaving too much perceptual compression up to the generative diffusion model while guaranteeing highfidelity reconstructions (see Fig. 1). While approaches to jointly learn an encoding/decoding model together with a score-based prior exist [90], they still require a difficult weighting between reconstruction and generative capabilities [11] and are outperformed by our approach (Sec. 4).
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2056.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> bits of a digital image correspond to imperceptible details. While DMs allow to suppress this semantically meaningless information by minimizing the responsible loss term, gradients (during training) and the neural network backbone (training and inference) still need to be evaluated on all pixels, leading to superfluous computations and unnecessarily expensive optimization and inference.
> 
> Figure 2. Illustrating perceptual and semantic compression: Most
> 
> We propose *latent diffusion models* (*LDMs*) as an effective generative model and a separate mild compression stage that only eliminates imperceptible details. Data and images from [29].
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2057.png)

> **[图片提取文字 (image.png)]:**
> ## 3.1. Perceptual Image Compression
> 
> Our perceptual compression model is based on previous work [23] and consists of an autoencoder trained by com-
> 
> bination of a perceptual loss [102] and a patch-based [32] adversarial objective [20, 23, 99]. This ensures that the reconstructions are confined to the image manifold by enforc-
> 
> constructions are confined to the image manifold by enforcing local realism and avoids bluriness introduced by relying solely on pixel-space losses such as  $L_2$  or  $L_1$  objectives.
> 
> More precisely, given an image  $x \in \mathbb{R}^{H \times W \times 3}$  in PGR
> 
> More precisely, given an image  $x \in \mathbb{R}^{H \times W \times 3}$  in RGB space, the encoder  $\mathcal{E}$  encodes x into a latent representation  $z = \mathcal{E}(x)$ , and the decoder  $\mathcal{D}$  reconstructs the image from the latent, giving  $\tilde{x} = \mathcal{D}(z) = \mathcal{D}(\mathcal{E}(x))$ , where
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2058.png)

> **[图片提取文字 (image.png)]:**
> The first variant, KL-reg., imposes a slight KL-penalty towards a standard normal on the learned latent, similar to a VAE [45, 67], whereas VQ-reg. uses a vector quantization layer [93] within the decoder. This model can be interpreted as a VQGAN [23] but with the quantization layer absorbed by the decoder. Because our subsequent DM is designed to work with the two-dimensional structure of our learned latent space  $z = \mathcal{E}(x)$ , we can use relatively mild compres-
> 
>  $z \in \mathbb{R}^{h \times w \times c}$ . Importantly, the encoder downsamples the
> 
> image by a factor f = H/h = W/w, and we investigate
> 
> we experiment with two different kinds of regularizations.
> 
> In order to avoid arbitrarily high-variance latent spaces,
> 
> different downsampling factors  $f = 2^m$ , with  $m \in \mathbb{N}$ .
> 
> to work with the two-dimensional structure of our learned latent space  $z = \mathcal{E}(x)$ , we can use relatively mild compression rates and achieve very good reconstructions. This is in contrast to previous works [23, 64], which relied on an arbitrary 1D ordering of the learned space z to model its distribution autoregressively and thereby ignored much of the inherent structure of z. Hence, our compression model preserves details of x better (see Tab. 8). The full objective and training details can be found in the supplement.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2059.png)

LDM：latent space的训练方式和pixel space一致，将过程中**pixels x改为latent z**。

VQGAN+ARM采用Transformer生成高度压缩的patch code，VQGAN+Conv-UNet使用**面向图形特化**的Conv-UNet（Inductive Bias）生成含噪编码zt。

> **[图片提取文字 (image.png)]:**
> ## 3.2. Latent Diffusion Models
> 
> **Diffusion Models** [79] are probabilistic models designed to learn a data distribution p(x) by gradually denoising a normally distributed variable, which corresponds to learning the reverse process of a fixed Markov Chain of length T.
> 
> For image synthesis, the most successful models [15,29,70] rely on a reweighted variant of the variational lower bound on p(x), which mirrors denoising score-matching [82]. These models can be interpreted as an equally weighted sequence of denoising autoencoders  $\epsilon_{\theta}(x_t,t)$ ;  $t=1\ldots T$ , which are trained to predict a denoised variant of their input  $x_t$ , where  $x_t$  is a noisy version of the input  $x_t$ . The corresponding objective can be simplified to (Sec. A)
> 
> sponding objective can be simplified to (Sec. A) 
> $$L_{DM} = \mathbb{E}_{x,\epsilon \sim \mathcal{N}(0,1),t} \left[ \|\epsilon - \epsilon_{\theta}(x_t,t)\|_2^2 \right], \tag{1}$$
> 
> with t uniformly sampled from  $\{1, \ldots, T\}$ .
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2060.png)

> **[图片提取文字 (image.png)]:**
> and  $\mathcal{D}$ , we now have access to an efficient, low-dimensional latent space in which high-frequency, imperceptible details are abstracted away. Compared to the high-dimensional pixel space, this space is more suitable for likelihood-based generative models, as they can now (i) focus on the important, semantic bits of the data and (ii) train in a lower dimensional, computationally much more efficient space.
> 
> Generative Modeling of Latent Representations With
> 
> our trained perceptual compression models consisting of  ${\cal E}$ 
> 
> Unlike previous work that relied on autoregressive, attention-based transformer models in a highly compressed, discrete latent space [23,64,99], we can take advantage of image-specific inductive biases that our model offers. This includes the ability to build the underlying UNet primar-
> 
> ily from 2D convolutional layers, and further focusing the
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2061.png)

> **[图片提取文字 (image.png)]:**
> objective on the perceptually most relevant bits using the reweighted bound, which now reads  $L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t} \left| \|\epsilon - \epsilon_{\theta}(z_t, t)\|_2^2 \right|. \tag{2}$ 
> 
> The neural backbone  $\epsilon_{\theta}(\circ, t)$  of our model is realized as a time-conditional UNet [69]. Since the forward process is fixed,  $z_t$  can be efficiently obtained from  $\mathcal{E}$  during training, and samples from p(z) can be decoded to image space with a single pass through  $\mathcal{D}$ .
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2062.png)

Inductive Bias：模型构建使用的假设。

> **[图片提取文字 (image.png)]:**
> uses to prioritize certain solutions and generalize from limited training data to unseen examples. Without these built-in assumptions, a model would be unable to make predictions on new data better than
> 
> Inductive bias refers to the set of assumptions a learning algorithm
> 
> random quessing.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2063.png)

> **[图片提取文字 (image.png)]:**
> ## Examples in Machine Learning Algorithms Different algorithms have different inductive biases built into their
> 
> architecture or training process:
> 
> - Linear Regression: Assumes a linear relationship between input and output variables.
> - k-Nearest Neighbors (k-NN): Assumes that data points close to each other in feature space likely belong to the same class.
>    Decision Trees: Biased towards simpler, shorter tree structures and
> - axis-parallel decision boundaries.
> 
>   Support Vector Machines (SVMs): Assumes that distinct classes
> - tend to be separated by a wide boundary (maximum margin).
> 
>   Convolutional Neural Networks (CNNs): Designed with
>   - assumptions of **locality** (nearby pixels are related) and **translation equivariance/invariance** (patterns are relevant regardless of their position in an image), which is why they excel at image tasks.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2064.png)

> **[图片提取文字 (image.png)]:**
> - Recurrent Neural Networks (RNNs): Have a bias towards sequentiality and processing data in a specific temporal order, making them suitable for time-series and sequence data.
> 
>   Transformers: Generally considered to have weaker inductive
> - biases compared to CNNs or RNNs, which makes them more flexible and data-hungry, but capable of learning a wider range of patterns given sufficient data.
> 
> ## Strong vs. Weak Inductive Bias
> 
> underfitting if they are wrong.
> 
> overfitting.
> 
> - Strong Bias: The model makes specific assumptions about the data structure (e.g., assuming a strictly linear relationship). This works well with limited data if the assumptions are correct, but can lead to
> - Weak Bias: The model makes minimal assumptions (e.g., a large neural network). This is more flexible and can adapt to complex patterns but requires large amounts of data to avoid
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2065.png)

### 条件LDM

将图片caption或语义Map y、隐变量zt和时间t作为联合条件，和DDPM一致训练。

UNet + Attention将条件y引入Diffusion（而不是直接concate）：**τθ(y)**将条件y编码成隐变量参与K和V的计算，φi(zt)是预测噪声εθ的隐变量参与Q的计算，**条件y决定含噪表达zt的成分值域V和索引K**，**噪声εθ决定含噪表达zt从值域中组成的方式Q**。

> **[图片提取文字 (image.png)]:**
> ## 3.3. Conditioning Mechanisms Similar to other types of generative models [55, 80],
> 
> diffusion models are in principle capable of modeling conditional distributions of the form p(z|y). This can
> 
> be implemented with a conditional denoising autoencoder  $\epsilon_{\theta}(z_t, t, y)$  and paves the way to controlling the synthesis
> 
> process through inputs y such as text [66], semantic maps [32,59] or other image-to-image translation tasks [33]. In the context of image synthesis, however, combining the generative power of DMs with other types of conditionings beyond class-labels [15] or blurred variants of the input image [70] is so far an under-explored area of research.
> 
> We turn DMs into more flexible conditional image generators by augmenting their underlying UNet backbone with the cross-attention mechanism [94], which is effective for learning attention-based models of various input modalities [34,35]. To pre-process y from various modalities (such as language prompts) we introduce a domain specific encoder  $\tau_{\theta}$  that projects y to an intermediate representation  $\tau_{\theta}(y) \in \mathbb{R}^{M \times d_{\tau}}$ , which is then mapped to the intermediate
> 
> layers of the UNet via a cross-attention layer implementing Attention $(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$ , with
> 
>  $Q = W_Q^{(i)} \cdot \varphi_i(z_t), \ K = W_K^{(i)} \cdot \tau_\theta(y), \ V = W_V^{(i)} \cdot \tau_\theta(y).$ 
> 
> Here,  $\varphi_i(z_t) \in \mathbb{R}^{N \times d_\epsilon^i}$  denotes a (flattened) intermediate representation of the UNet implementing  $\epsilon_{\theta}$  and  $W_V^{(i)}$   $\in$  $\mathbb{R}^{d \times d_{\epsilon}^{i}}, W_{O}^{(i)} \in \mathbb{R}^{d \times d_{\tau}} \& W_{K}^{(i)} \in \mathbb{R}^{d \times d_{\tau}}$  are learnable pro-
> 
> jection matrices [35, 94]. See Fig. 3 for a visual depiction.
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2066.png)

> **[图片提取文字 (image.png)]:**
> Based on image-conditioning pairs, we then learn the conditional LDM via
> 
> $$L_{LDM} := \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_{\theta}(z_t, t, \tau_{\theta}(y))\|_2^2 \right], \quad (3)$$
> 
> where both  $\tau_{\theta}$  and  $\epsilon_{\theta}$  are jointly optimized via Eq. 3. This conditioning mechanism is flexible as  $\tau_{\theta}$  can be parameterized with domain-specific experts, *e.g.* (unmasked) transformers [94] when y are text prompts (see Sec. 4.3.1)
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2067.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3. We condition LDMs either via concatenation or by a more general cross-attention mechanism. See Sec. 3.3
![image.png](X2Img%E3%80%81VQGAN+ARM%E3%80%81LDM%EF%BC%9A19-22/image%2068.png)