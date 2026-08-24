## *LongLLMLingua*: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression

## Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu

Microsoft Corporation {hjiang,qianhuiwu,xufluo,dongsli,cyl,yuqyang,liliqiu}@microsoft.com

## Abstract

In long context scenarios, large language models (LLMs) face three main challenges: higher computational cost, performance reduction, and position bias. Research indicates that LLM performance hinges on the density and position of key information in the input prompt. Inspired by these findings, we propose LongLLM-Lingua for prompt compression towards improving LLMs' perception of the key information to simultaneously address the three challenges. Our extensive evaluation across various long context scenarios demonstrates that LongLLMLingua not only enhances performance but also significantly reduces costs and latency. For instance, in the NaturalQuestions benchmark, LongLLMLingua boosts performance by up to 21.4% with around 4x fewer tokens in GPT-3.5-Turbo, leading to substantial cost savings. It achieves a 94.0% cost reduction in the LooGLE benchmark. Moreover, when compressing prompts of about 10k tokens at ratios of 2x-6x, LongLLMLingua can accelerate end-to-end latency by 1.4x-2.6x.[1](#page-0-0)

## <span id="page-0-1"></span>1 Introduction

Large language models (LLMs) have revolutionized user-oriented language technologies and are serving as crucial components in more and more applications. Carefully designing prompts is necessary to achieve better performance in specific downstream tasks. The commonly used technologies such as In-Context Learning (ICL) [\(Min et al.,](#page-10-0) [2022;](#page-10-0) [Dong et al.,](#page-9-0) [2023\)](#page-9-0), Retrieval Augment Generation (RAG) [\(Lewis et al.,](#page-9-1) [2020;](#page-9-1) [Asai et al.,](#page-8-0) [2024\)](#page-8-0), and Multi-turn Agent [\(Shen et al.,](#page-10-1) [2024;](#page-10-1) [Park et al.,](#page-10-2) [2023;](#page-10-2) [Wu et al.,](#page-10-3) [2023a\)](#page-10-3) are driving prompts to be increasingly longer, even reaching thousands of tokens. Scenarios such as multi-document question answering, code completion, and document summarization also necessitate the processing of long contexts.

There are three main challenges when LLMs are used in long context scenarios: (1) Higher computational costs, encompassing both financial and latency expenses. (2) Longer prompts introduce irrelevant and redundant information, which can weaken LLMs' performance [\(Shi et al.,](#page-10-4) [2023\)](#page-10-4), as illustrated in Figure [1a.](#page-1-0) (3) LLMs exhibit position bias [\(Kamradt,](#page-9-2) [2023\)](#page-9-2), also known as the "lost in the middle" issue [\(Liu et al.,](#page-10-5) [2024\)](#page-10-5), suggesting that the placement of key information within the prompt significantly affects LLMs' performance. This is demonstrated by the purple curve in Figure [1b.](#page-1-1)

Inspired by these observations, we propose *LongLLMLingua* to address the three challenges. Specifically, we use LLMLingua [\(Jiang et al.,](#page-9-3) [2023a\)](#page-9-3) as the backbone for prompt compression to address the first challenge, *i.e.*, reduce cost and latency. However, in the case of long contexts, the distribution of question-relevant key information in the prompt is generally dynamic and sparse. Existing prompt compression methods like LLMLingua [\(Jiang et al.,](#page-9-3) [2023a\)](#page-9-3) and Selective-Context [\(Li](#page-9-4) [et al.,](#page-9-4) [2023c\)](#page-9-4) that often fail to consider question during compression, resulting in retention of excessive noise and decreased performance. LongLLM-Lingua aims to improve LLMs' perception of key information pertinent to the question, thereby overcoming the noise and position bias issues in long contexts, shown in Figure [1b.](#page-1-1) The underlying principle of LongLLMLingua is that small LM are inherently capable of capturing the distribution of key information relevant to a given question.

Our main contributions are five-fold: (1) We propose a question-aware coarse-to-fine compression method to improve the key information density in the prompt (Sec. [4.1\)](#page-2-0); (2) We introduce a document reordering strategy to minimize position bias in LLMs. (Sec. [4.2\)](#page-3-0); (3) We establish dynamic compression ratios for precise control between coarse and fine compression levels (Sec. [4.3\)](#page-3-1); (4) We propose a post-compression

<span id="page-0-0"></span><sup>1</sup>Access our code at <https://aka.ms/LongLLMLingua>.

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> 100 Normalized Performance(%) 95 90 Multi-Document QA Code Completion 85 Summarization 20 10 Document Number in the Prompt
![](_page_1_Figure_0.jpeg)

> **[图片提取文字 (无描述)]:**
> 75 Accuracy(%) Original LongLLMLingua w/o Reorder (4x) 55 LongLLMLingua (4x) 1st 5th 10th 15th 20th Position of Document with the Answer
![](_page_1_Figure_1.jpeg)

(a) Performance v.s. Document Number

<span id="page-1-1"></span>(b) Performance v.s. Key Information Position

Figure 1: (a) LLMs' performance in downstream tasks decreases with increased noise in prompts. In this case, we keep k most relevant documents/paragraphs based on the ground-truth or LongLLMLingua  $r_k$ . A larger k implies more noise introduced into the prompt. To improve the key information density in the prompt, we present question-aware coarse-to-fine compression. (b) LLMs' ability to capture the relevant information depends on their positions in the prompt. To reduce information loss in the middle, we introduce a document reordering mechanism.

subsequence recovery strategy to improve the integrity of the key information (4.4). (5) We evaluate LongLLMLingua across five benchmarks, *i.e.*, NaturalQuestions (Liu et al., 2024), LongBench (Bai et al., 2023), ZeroSCROLLS (Shaham et al., 2023), MuSicQue (Trivedi et al., 2022), and LooGLE (Li et al., 2023b), covering a variety of long context scenarios. Experimental results reveal that LongLLMLingua's compressed prompts outperform original prompts in terms of performance, cost efficiency, and system latency.

#### 2 Problem Formulation

Following LLMLingua (Jiang et al., 2023a), we use  $\mathbf{x} = (\mathbf{x}^{\text{ins}}, \mathbf{x}_1^{\text{doc}}, \cdots, \mathbf{x}_K^{\text{doc}}, \mathbf{x}^{\text{que}})$  to represent a prompt, including the instruction  $\mathbf{x}^{\text{ins}}$ , K documents  $\mathbf{x}_i^{\text{doc}}$ , and the question  $\mathbf{x}^{\text{que}}$ . However, this definition can be adjusted for specific scenarios. The objective of a prompt compression system can be formulated as:

$$\min_{\widetilde{\mathbf{x}}} D_{\phi}(\mathbf{y}, \widetilde{\mathbf{y}}) + \lambda \|\widetilde{\mathbf{x}}\|_{0}, \tag{1}$$

where  $\widetilde{\mathbf{x}}$  represents the compressed prompt, a token-level subsequence of  $\mathbf{x}$ .  $\mathbf{y}$  and  $\widetilde{\mathbf{y}}$  represent the LLM-generated results from  $\mathbf{x}$  and  $\widetilde{\mathbf{x}}$ , respectively.  $D_{\phi}$  measures the distance function, such as KL divergence.  $\lambda$  serves as a hyper-parameter balancing the compression ratio. Additionally, this study explores a permutation operation space over the K documents  $(\mathbf{x}_1^{\mathrm{doc}}, \cdots, \mathbf{x}_K^{\mathrm{doc}})$  for joint optimization.

## 3 Preliminary: LLMLingua

LLMLingua (Jiang et al., 2023a) utilizes a small language model  $\mathcal{M}_S$  to evaluate the perplexity of each prompt token, removing those with lower perplexities. This method is premised on the idea that tokens with lower perplexities have a negligible effect on the language model's overall entropy gain, implying their removal slightly impacts the LLMs' contextual understanding. This process is viewed as an application of "LM is Compression" (Delétang et al., 2023). LLMLingua include three key components: budget controller, iterative token-level prompt compression, and distribution alignment, highlighted by italic text in Figure 2. The budget controller assigns varying compression ratios to different parts of the prompt (i.e., instruction, demonstrations, question), implementing coarse-level prompt compression. Subsequent steps involve dividing intermediate results into segments and applying token-level compression iteratively, where each token's perplexity based on preceding compressed segments. To aware different target LLMs, LLMLingua fine-tunes  $\mathcal{M}_S$  using data from the target LLM.

#### 4 LongLLMLingua

LongLLMLingua builds on LLMLingua to better compress prompts in long context scenorias. It tackles three main issues in handling lengthy contexts, as introduced in Sec. 1. This approach focuses on making LLMs more effective at recognizing key

<span id="page-2-1"></span>> **[图片提取文字 (无描述)]:**
> **Original Prompt** LongLLMLingua Black-box LLMs Instruction: Answer the question based I Budget Controller on the given passages. Only give me the **Ouestion-aware Coarse-Grained** answer and do not output any other Compression words. The following are given w/ document reordering passages. Document 1: Alberic III of Dammartin Alberic III of Dammartin (Aubry de Subsequence III Execution with Dammartin) (c. 1138 – 19 September Distribution Recovery Compressed Prompt 1200) was a French count and son of Alianment Alberic II, Count of Dammartin, and Clémence de Bar, daughter of Reginald **Compressed Prompt** I. Count of Bar... Small Document 2: Answer the question based on the given Model Response passages. ... Passage 4:was a Roman **Document N:** Pope Agapetus II noblewo who was the alleged mistress Pope Agapetus II (died 8 November 955) of Pope Sergius III and was given the was the bishop of Rome and ruler of the unprecedented titles senatrix Papal States from 10 May 946 to his ("senatoress") and patricia of Rome by death. A nominee of the princeps of Pope John X., when Ottosys, the of and, Rome, Alberic II of Spoleto, his II Iterative Token-level were to to discusss Rome and other pontificate occurred during ... Question-aware Fine-Grained more important, were us was to the Question: Who gave the mother of dispute the the of Re... Who gave the Compression Alberic II of Spoleto the title "patricia" mother of Alberic II of Spoleto the title w/ dynamic compression ratio of Rome? "patricia" of Rome? ~13k tokens ~2k tokens
![](_page_2_Figure_0.jpeg)

Figure 2: Framework of LongLLMLingua. Gray Italic content: As in LLMLingua.

information related to the question in the prompt. It encompasses three perspectives and further incorporates a subsequence recovery strategy, as shown in Figure 2, to enhance the accuracy and reliability of the information provided to users. In this section, we detail how each part of LongLLMLingua works to improve the LLMs deal with long context.

## <span id="page-2-0"></span>4.1 How to improve key information density in the prompt?

### **Question-Aware Coarse-Grained Compression**

In coarse-grained compression, we aim to figure out a metric  $r_k$  to evaluate the importance of each document  $\mathbf{x}_k^{\mathrm{doc}} = \{x_{k,i}^{\mathrm{doc}}\}_{i=1}^{N_k}$ , where  $N_k$  is the number of tokens in  $\mathbf{x}_k^{\mathrm{doc}}$ . We only keep  $\mathbf{x}_k^{\mathrm{doc}}$  with higher  $r_k$  as the intermediate compressed results. One approach to improve key information density in the compressed prompts is to calculate document-level perplexity conditioned on the question  $p(\mathbf{x}_k^{\mathrm{doc}}|\mathbf{x}^{\mathrm{que}})$ . However, this method may not be effective because documents often contain a significant amount of irrelevant information. Even when conditioned on  $\mathbf{x}^{\mathrm{que}}$ , the perplexity scores computed for entire documents may not be sufficiently distinct, rendering them an inadequate metric for document-level compression.

We propose to use the perplexity of the question  $\mathbf{x}^{\text{que}}$  conditioned on different contexts  $\mathbf{x}_k^{\text{doc}}$   $p(\mathbf{x}^{\text{que}}|\mathbf{x}_k^{\text{doc}})$  to represent the association between them. We also append a restrictive statement<sup>2</sup>  $\mathbf{x}^{\text{restrict}}$  after  $\mathbf{x}^{\text{que}}$  to strengthen the interconnection

of  $\mathbf{x}^{\text{que}}$  and  $\mathbf{x}_k^{\text{doc}}$ . It can be regarded as a regularization term that mitigates the impact of hallucinations. This can be formulated as:

<span id="page-2-5"></span>
$$r_k = -\frac{1}{N_c} \sum_{i}^{N_c} \log p(x_i^{\text{que,restrict}} | \mathbf{x}_k^{\text{doc}}), \quad (2)$$
$$k \in \{1, 2, \dots, K\},$$

where  $x_i^{\text{que},\text{restrict}}$  is the i-th token in the concatenated sequence of  $\mathbf{x}^{\text{que}}$  and  $\mathbf{x}^{\text{restrict}}$  and  $N_c$  in the number of tokens.

Figure 3a displays the recall distribution of different retrieval methods, including traditional relevance methos (BM25, Gzip (Jiang et al., 2023b)), embedding-based methods (OpenAI-embedding, Voyageai³, BGE-large-en v1.5 (Xiao et al., 2023), Sentence-BERT (Reimers and Gurevych, 2019), Jina (Günther et al., 2023)), and reranker methods (Cohere-Rerank⁴, BGE-llmembeder, BGE-Rankerlarge), which demonstrates that our coarse-level compression approach achieves the highest recall with different numbers of retained documents, suggesting that it preserves the most key information from the contexts in the compressed results.

## Question-Aware Fine-Grained Compression In fine-grained compression, we assess the importance of each token in the instruction $\mathbf{x}^{ins}$ , the ques-

tance of each token in the instruction  $\mathbf{x}^{\text{ins}}$ , the question  $\mathbf{x}^{\text{que}}$ , and K' documents  $\{\mathbf{x}_i^{\text{doc}}\}_{i=1}^{K'}$  retained after coarse-grained compression. We incorporate the

<span id="page-2-2"></span><sup>&</sup>lt;sup>2</sup>Specifically, "We can get the answer to this question in the given documents".

<span id="page-2-3"></span><sup>3</sup>https://www.voyageai.com/

<span id="page-2-4"></span><sup>4</sup>https://cohere.com/rerank

<span id="page-3-2"></span>> **[图片提取文字 (无描述)]:**
> 100 LLMLingua BM25 OpenAI 80 Voyageai BGE-large-en v1.5 Recall(%) 60 SBERT Gzip Cohere-Rerank 40 BGE-Ilmembeder Jina LongLLMLingua  $r_k$ 20 w/o restrict BGE-Ranker-large LongLLMLingua rk 0 10 20 Number of Retained Documents (a) Recall Distribution
![](_page_3_Figure_0.jpeg)

> **[图片提取文字 (无描述)]:**
> 1.0 Perplexity Document Avg. Perplexity 70 9.0 9.0 8.0 8.0 Contrastive Perplexity 0.0 20 Document Position in the Prompt (1) D 1 ', D', '1 ,' (5.1)
![](_page_3_Figure_1.jpeg)

<span id="page-3-3"></span>(b) Perplexity Distribution (5th)

Figure 3: (a) Comparison of recall on NaturalQuestions Multi-documemnt QA dataset, which increases from top to bottom in terms of Recall@1. Different colors represent different types of methods. Among them, yellow represents traditional relevance methods, green signifies embedding-based methods, and red denotes rerank-based methods. (b) Comparison between perplexities and contrastive perplexities of tokens in the prompt from Multi-documemnt QA dataset. The document containing the ground-truth information is located in the 5th position. More results on position can be found in the Appendix C.1.

iterative compression mechanism following LLM-Lingua and directly calculate token perplexities to compress  $\mathbf{x}^{\text{ins}}$  and  $\mathbf{x}^{\text{que}}$ . In this section, we investigate how to make the fine-grained token-level compression over  $\{\mathbf{x}_k^{\text{doc}}\}_{k=1}^{K'}$  aware of the question  $\mathbf{x}^{\text{que}}$ , so that the compressed results could contain more question-relevant key information.

A straightforward solution for the awareness of  $\mathbf{x}^{\text{que}}$  is to simply concatenate it at the beginning of the whole context. However, this will result in low perplexities of relevant tokens in the context following the condition of question  $\mathbf{x}^{\text{que}}$ , further reducing their differentiation from other tokens.

In this paper, we propose *contrastive perplexity*, *i.e.*, the distribution shift caused by the condition of the question, to represent the association between the token and the question. The contrastive perplexity based importance metric  $s_i$  for each token  $x_i$  in  $\{\mathbf{x}_k^{\text{doc}}\}_{k=1}^{K'}$  can be formulated as:

<span id="page-3-4"></span>
$$s_i = \operatorname{perplexity}(x_i|x_{< i}) - \operatorname{perplexity}(x_i|x^{\operatorname{que}}, x_{< i}). \eqno(3)$$

Additionally, we provide the derivation of its mathematical significance in the Appendix A, concluding that it is equivalent to conditional pointwise mutual information (Church and Hanks, 1989).

Figure 3b illustrates the difference between perplexities and contrastive perplexities. The distribution of perplexities appears random, making it challenging to extract information related to the question. However, tokens with high contrastive perplexities tend to cluster near the ground-truth

document, which contains information relevant to the question. This suggests that the proposed contrastive perplexity can better distinguish tokens relevant to the question, thus improving the key information density in the compressed results.

## <span id="page-3-0"></span>**4.2** How to reduce information loss in the middle?

As demonstrated in Figure 1b, LLM achieves the highest performance when relevant information occurs at the beginning and significantly degrades if relevant information is located in the middle of long contexts. After the coarse-grained compression, we have obtained a set of documents  $\{\mathbf{x}_k^{\mathrm{doc}}\}_{k=1}^{K'}$  with their corresponding importance scores  $\{r_k\}_{k=1}^{K'}$  indicating their association with the question  $\mathbf{x}^{\mathrm{que}}$ . Therefore, we reorder documents using their importance scores to better leverage LLMs' information perception difference in positions:

$$(\mathbf{x}^{\text{ins}}, \mathbf{x}_{1}^{\text{doc}}, \cdots, \mathbf{x}_{K'}^{\text{doc}}, \mathbf{x}^{\text{que}}) \xrightarrow{r_{k}} (\mathbf{x}^{\text{ins}}, \mathbf{x}_{r1}^{\text{doc}}, \cdots, \mathbf{x}_{rK'}^{\text{doc}}, \mathbf{x}^{\text{que}})$$

$$(4)$$

# <span id="page-3-1"></span>**4.3** How to achieve adaptive granular control during compression?

In fine-grained compression, LLMLingua applies the same compression ratio over all documents obtained from budget controller. However, the key information density of different documents is different. The more relevant to the question a document <span id="page-4-2"></span>Document [1](Title: List of Nobel laureates in Physics) The first Nobel Prize in Physics was awarded in 1901 to {Wilhelm Conrad Röntgen}{Wilhelm Con rad Röntgen}, of Germany,...

Original Prompt

Document [1](Title: List of Nobelates in Physics) The first Nobel1
{Wilhelmgen}{Wilhelm gen}, of, who received, .... Compressed Prompt

{Wilhelmgen} {Wilhelm gen}

LLMs' Response

Figure 4: The example of Subsequence Recovery, the red text represents the original text, and the blue text is the result after using the LLaMA 2-7B tokenizer.

is, the more budget (i.e., lower compression ratio) we should allocate to it. Therefore, we bridge coarse-grained compression to fine-grained compression and use the importance scores  $\{r_k\}_{k=1}^{K'}$  obtained from coarse-grained compression to guide the budget allocation in fine-grained compression. In this way, we can achieve adaptive granular control on the whole.

Specifically, we first determine the initial budget for the retained documents  $^{5}$   $\tau^{\rm doc}$  using the budget controller of LLMLingua. During fine-grained compression, we follow the iterative token-level compression algorithm in LLMLingua but dynamically assign the compression budget  $\tau^{\rm doc}_{k}$  to each document  ${\bf x}^{\rm doc}_{k}$  according to the ranking index  $I(r_{k})$  (e.g., 0, 1) of its importance score from the coarsegrained compression. In this paper, we employ a linear scheduler for the adaptive allocation. Budget of each token  $x_{i}$  can be formulated as:

$$\tau_i = \tau_k^{\text{doc}}, \quad \forall x_i \in \mathbf{x}_k^{\text{doc}},$$
  
$$\tau_k^{\text{doc}} = \max(\min((1 - \frac{2I(r_k)}{K'})\delta\tau + \tau^{\text{doc}}, 1), 0),$$
  
(5)

where i and k is the index of token and document, K' denotes the number of documents, and  $\delta \tau$  is a hyper-parameter that controls the overall budget for dynamic allocation.

## <span id="page-4-0"></span>4.4 How to improve the integrity of key information?

During the generation process, LLMs tend to replicate entities found in the prompt, such as names, places, and organizations. Compressing these entities at the token level doesn't affect the LLMs' understanding of semantic content but can lead to errors in the generated content.

Therefore, we propose a subsequence recovery method to restore the original content in LLMs' responses. This method relies on the subsequence relationship among tokens in the original prompt, compressed prompt, and LLMs' response, as shown in Figure 4.

The overall procedure includes: i) Iterate through tokens  $y_l$  in LLMs' response and select the longest substring  $\widetilde{\boldsymbol{y}}_{\text{key},l} = \{y_l, y_{l+1}, ..., y_r\}$  that appears in the compressed prompt  $\widetilde{\boldsymbol{x}}$ . ii) Find the maximum common shortest subsequence  $\boldsymbol{x}_{i,j} = \{x_i, x_{i+1}, ..., x_j\}$  in the original prompt  $\boldsymbol{x}$ , corresponding to the representation  $\widetilde{\boldsymbol{y}}_{\text{key},l}$  in the original prompt (accelerated using prefix trees or sequence automata). iii) Replace the matched tokens  $\widetilde{\boldsymbol{y}}_{\text{key},l}$  in LLMs' response with the corresponding subsequence  $\boldsymbol{x}_{i,j}$  from the original prompt. For more details, please refer to Algorithm 1.

## <span id="page-4-3"></span>**Algorithm 1** Token-level Subsquence Recovery Algorithm

**Input**: The original prompt x; the compressed prompt  $\widetilde{x}$ ; the generation response of LLMs y.

```
1: Set the final response list y_{\text{rec}} = \phi, the left token index of subsquence l to 0.
```

```
2: while l < y.len() do
          if Substring y_l \in \widetilde{x} then
              Find the longer substring \widetilde{\mathbf{y}}_{\text{key},l} = \{y_l, y_{l+1}, 
 4:
      ..., y_r\} \in \widetilde{\boldsymbol{x}}.
               Find the maximum common shortest subsequence
     \boldsymbol{x}_{i,j} = \{x_i, x_{i+1}, ..., x_j\} in the original prompt \boldsymbol{x}.
 6:
               Add the subsequence x_{i,j} = \{x_i, x_{i+1}, ..., x_j\}
     to the response y_{rec}.
 7:
               Set the left index l to r+1.
 8:
 9:
               Add the token y_l to the response y_{rec}.
10:
               Set the left index l to l + 1.
          end if
12. end while
Output: The final response list y_{rec}.
```

## 5 Experiments

Here, we investigate: (1) How *effective* is LongLLMLingua? (2) How *efficient* is LongLLM-Lingua?

**Implementation details** In this paper, we use GPT-3.5-Turbo-0613<sup>6</sup> and LongChat-13B-16k as the target LLMs, both accessible via OpenAI<sup>7</sup> and HuggingFace<sup>8</sup>. To ensure stable and reproducible

<span id="page-4-1"></span><sup>&</sup>lt;sup>5</sup>In LLMLingua, it is  $\tau^{\text{dems}}$  for demonstrations.

<span id="page-4-4"></span><sup>&</sup>lt;sup>6</sup>For experiments with original prompts exceeding 4k tokens, we utilize GPT-3.5-Turbo-16k-0613.

<span id="page-4-5"></span><sup>&</sup>lt;sup>7</sup>https://platform.openai.com

<span id="page-4-6"></span><sup>8</sup>https://huggingface.co/lmsys/longchat-13b-16k

results, we employ greedy decoding and set the temperature to 0 in all experiments. For the small language models used for compression, we apply LLaMA-2-7B-Chat[9](#page-5-0) , which has been aligned by supervised fine-tuning and RLHF. We implement our approach with PyTorch 1.13.1 and Hugging-Face Transformers. We set up hyperparameters following LLMLingua except for the segment size used in iterative token-level compression set to 200 here. More details are provided in Appendix [B.](#page-11-1)

Dataset & evaluation metric We use NaturalQuestions for the multi-document QA task, and use LongBench and ZeroSCROLLS for general long context scenarios. We also test on multihop QA tasks using MuSiQue dataset [\(Trivedi](#page-10-7) [et al.,](#page-10-7) [2022\)](#page-10-7), and long dependency QA tasks using LooGLE benchmark [\(Li et al.,](#page-9-5) [2023b\)](#page-9-5). Please refer to Appendix [C](#page-12-1) for more details on datasets.

Baselines We include two sets of baselines in following experiments:

*(i) Retrieval-based Methods.* We assess the question-document association in the prompt using five SoTA retrieval methods: BM25, Gzip [\(Jiang](#page-9-7) [et al.,](#page-9-7) [2023b\)](#page-9-7), SentenceBERT [\(Reimers and](#page-10-9) [Gurevych,](#page-10-9) [2019\)](#page-10-9), OpenAI Embedding, and the LongLLMLingua ranker's important metric r<sup>k</sup> for coarse-grained compression. Notably, embedding model-based compression mirrors the method in [Xu et al.](#page-11-2) [\(2024\)](#page-11-2). We remove low-relevance sentences or paragraphs to meet compression limits, maintaining the original document sequence.

*(ii) Compression-based Methods.* We compare our approach with two state-of-art methods for prompt compression, *i.e.*, Selective Context [\(Li](#page-9-4) [et al.,](#page-9-4) [2023c\)](#page-9-4) and LLMLingua [\(Jiang et al.,](#page-9-3) [2023a\)](#page-9-3). Both methods employ LLaMA-2-7B-Chat as the small language model for compression. In LLM-Lingua, a coarse-to-fine approach is used to handle constraints of compression ratio: the original prompt is first compressed to k times the constraint at a coarse level, where k is the granular control coefficient; token-level is then performed to reach the overall constraint. Our method follows the same coarse-to-fine logic to achieve the constraint.

Main results Table [1](#page-6-0) and [2](#page-7-0) present the performance of various methods under different compression constraints. There are multiple observa-

tions and conclusions: (1) Our LongLLMLingua achieves the best performance across different tasks and constraints of compression ratios. Compared to the original prompt, our compressed prompt can derive higher performance with much lower cost. For example, LongLLMLingua gains a performance boost of 21.4% on NaturalQuestions with the ground-truth document at the 10th position, while the number of tokens input to GPT3.5-Turbo is ∼4x less. (2) Compression-based methods like Selective Context [\(Li et al.,](#page-9-4) [2023c\)](#page-9-4) and LLMLingua [\(Jiang et al.,](#page-9-3) [2023a\)](#page-9-3) perform poorly on most tasks, especially those with abundant irrelevant information in the original prompt. This is due to their pure information entropy based compression mechanism, which includes too much noise in the compressed results and even leads to performance worse than the zero-shot setting, *e.g.*, on NaturalQuestions. (3) Retrieval-based methods work well with low compression ratios. However, their performance declines as the compression progresses, *e.g.*, 2x → 4x; 3000 tokens → 2000 tokens. This may be caused by the decreased recall. Figure [3a](#page-3-2) is the illustration of cases on NaturalQuestions. (4) LongLLMLingua as well as our coarse-grained compression metric r<sup>k</sup> only is much more robust than all other baselines under different tasks and compression constraints. With the increase of the compression ratio, *e.g.*, 2x → 4x, LongLLMLingua even achieves a little performance gain. We mainly owe this to the question-aware coarse-to-fine compression, which can better figure out the key information and reach a higher key information density with a higher compression ratio. (5) The proposed reordering method helps in not only our approach but also other baselines, well demonstrating its effectiveness. (6) Compared to the results with a 2,000 tokens constraint, overall performance of 3,000 tokens has improved. LongLLMLingua sees an increase of 1.2 points in average score and a 1.6x speedup in end-to-end latency. In this scenario, the recall rates of retrieval-based methods have increased, leading to a significant improvement in their accuracy. For example, BM25 achieves an average score of 48.9.

In addition, we also present experimental results on datasets such as MuSicQue, LooGLE, ZERO-SCROLLS, etc., in Appendix [C.](#page-12-1)

Ablation study To evaluate the contributions of different components in LongLLMLingua, we

<span id="page-5-0"></span><sup>9</sup> https://ai.meta.com/llama/

<sup>9</sup> https://python.langchain.com/docs/modules/data\_connecti on/document\_transformers/post\_retrieval/long\_context\_reorder

<span id="page-6-0"></span>

| Methods                                  |        | GPT3.5-Turbo |      |      |      |         |      |        | Long   | Chat- | 13b  |         | Leng   | th       | Late    | ency    |
|------------------------------------------|--------|--------------|------|------|------|---------|------|--------|--------|-------|------|---------|--------|----------|---------|---------|
| Methods                                  | 1st    | 5th          | 10th | 15th | 20th | Reorder | 1st  | 5th    | 10th   | 15th  | 20th | Reorder | Tokens | $1/\tau$ | Latency | Speedup |
|                                          |        |              |      |      |      |         | 23   | c cons | traint |       |      |         |        |          |         |         |
| Retrieval-based Meth                     | ods    |              |      |      |      |         |      |        |        |       |      |         |        |          |         |         |
| BM25                                     | 53.7   | 49.3         | 47.9 | 49.9 | 46.9 | 50.3    | 50.9 | 44.9   | 44.1   | 42.9  | 43.2 | 46.0    | 1,545  | 1.9x     | 2.1     | 1.9x    |
| Gzip                                     | 64.6   | 63.8         | 60.5 | 58.3 | 57.3 | 64.4    | 61.9 | 55.7   | 52.7   | 50.8  | 50.9 | 59.3    | 1,567  | 1.9x     | 2.1     | 1.9x    |
| SBERT                                    | 72.5   | 67.9         | 63.3 | 65.0 | 66.2 | 68.7    | 65.8 | 57.5   | 54.9   | 53.4  | 55.7 | 61.4    | 1,549  | 1.9x     | 2.2     | 1.9x    |
| OpenAI                                   | 73.0   | 65.6         | 66.5 | 65.4 | 65.5 | 69.9    | 65.9 | 57.5   | 56.2   | 54.2  | 55.7 | 61.7    | 1,550  | 1.9x     | 4.9     | 0.8x    |
| Long<br>LLM<br>Lingua $\boldsymbol{r}_k$ | 73.9   | 67.7         | 68.7 | 66.0 | 65.6 | 74.3    | 68.5 | 59.1   | 56.8   | 55.3  | 56.9 | 65.2    | 1,548  | 1.9x     | 2.3     | 1.8x    |
| Compression-based M                      | 1ethod | ls           |      |      |      |         |      |        |        |       |      |         |        |          |         |         |
| Selective-Context                        | 45.4   | 39.0         | 33.8 | 33.5 | 41.5 | -       | 53.2 | 26.3   | 25.4   | 24.2  | 33.3 | -       | 1,478  | 2.0x     | 7.4     | 0.6x    |
| LLMLingua                                | 39.7   | 39.5         | 40.4 | 37.1 | 42.3 | 41.5    | 38.7 | 37.3   | 35.7   | 34.1  | 37.5 | 37.1    | 1,410  | 2.1x     | 2.8     | 1.5x    |
| LongLLMLingua                            | 77.2   | 72.9         | 70.8 | 70.5 | 70.6 | 76.2    | 68.7 | 59.4   | 57.3   | 55.9  | 58.4 | 66.1    | 1,429  | 2.1x     | 2.9     | 1.4x    |
|                                          |        |              |      |      |      |         | 43   | cons   | traint |       |      |         |        |          |         |         |
| Retrieval-based Meth                     | ods    |              |      |      |      |         |      |        |        |       |      |         |        |          |         |         |
| BM25                                     | 40.6   | 38.6         | 38.2 | 37.4 | 36.6 | 36.3    | 39.5 | 37.5   | 36.8   | 36.4  | 35.5 | 37.7    | 798    | 3.7x     | 1.5     | 2.7x    |
| Gzip                                     | 63.1   | 61.0         | 59.8 | 61.1 | 60.1 | 62.3    | 57.6 | 52.9   | 51.0   | 50.1  | 50.4 | 57.2    | 824    | 3.6x     | 1.5     | 2.7x    |
| SBERT                                    | 66.9   | 61.1         | 59.0 | 61.2 | 60.3 | 64.4    | 62.6 | 56.6   | 55.1   | 53.9  | 55.0 | 59.1    | 808    | 3.6x     | 1.6     | 2.5x    |
| OpenAI                                   | 63.8   | 64.6         | 65.4 | 64.1 | 63.7 | 63.7    | 61.2 | 56.0   | 55.1   | 54.4  | 55.0 | 58.8    | 804    | 3.7x     | 4.3     | 1.0x    |
| LongLLMLingua $r_k$                      | 71.1   | 70.7         | 69.3 | 68.7 | 68.5 | 71.5    | 67.8 | 59.4   | 57.7   | 57.7  | 58.6 | 64.0    | 807    | 3.7x     | 1.7     | 2.4x    |
| Compression-based M                      | 1ethod | ls           |      |      |      |         |      |        |        |       |      |         |        |          |         |         |
| Selective-Context                        | 31.4   | 19.5         | 24.7 | 24.1 | 43.8 | -       | 38.2 | 17.2   | 15.9   | 16.0  | 27.3 | -       | 791    | 3.7x     | 6.8     | 0.6x    |
| LLMLingua                                | 25.5   | 27.5         | 23.5 | 26.5 | 30.0 | 27.0    | 32.1 | 30.8   | 29.9   | 28.9  | 32.4 | 30.5    | 775    | 3.8x     | 1.8     | 2.2x    |
| LongLLMLingua                            | 75.0   | 71.8         | 71.2 | 71.2 | 74.7 | 75.5    | 68.7 | 60.5   | 59.3   | 58.3  | 61.3 | 66.7    | 748    | 3.9x     | 2.1     | 2.0x    |
| Original Prompt                          | 75.7   | 57.3         | 54.1 | 55.4 | 63.1 | -       | 68.6 | 57.4   | 55.3   | 52.5  | 55.0 | -       | 2,946  | -        | 4.1     | -       |
| Zero-shot                                |        |              | 56   | 5.1  |      |         |      |        | 35     | 5.0   |      |         | 15     | 196x     | 1.1     | 3.7x    |

Table 1: Performance of different methods with different compression ratios (raw size / compressed size =  $1/\tau$ ) on NaturalQuestions (20 documents) (Liu et al., 2024). Reorder: we reorder the documents with relevance metrics of different baselines as our document reordering strategy described in Sec. 4.2. In the case of OpenAI, it corresponds to LongContextReorder<sup>9</sup> in the LangChain framework (Chase, 2022). For results reported under 1st to 20th, we do not use the reordering strategy for all methods.

introduce following variants of it for ablation study. (1) Variants about Question-aware Coarsegrained Compression, include: ours w/o Questionawareness, which calculates question-text relevance  $r_k$  using information entropy in LLMLingua, ours w/ SBERT, which employs SBERT to compute  $r_k$ , ours w/  $p(\mathbf{x}_k^{\text{doc}}|x_i^{\text{que,restrict}})$ , which replace  $p(x_i^{\text{que,restrict}}|\mathbf{x}_k^{\text{doc}})$  with  $p(\mathbf{x}_k^{\text{doc}}|x_i^{\text{que,restrict}})$  in Eq. (2), and ours w/o restrict, which only calculates the conditional probability corresponding to  $x^{\text{que}}$ . (2) Ours w/o Question-aware Fine-grained, which disregards Eq. (3) and only applies Iterative Token-level Prompt Compression as LLMLingua. (3) Ours w/o Dynamic Compression Ratio, where all documents share the same compression ratio in fine-grained compression. (4) Ours w/o and (5) LLMLingua w/ Subsequence Recovery, which either removes or adds the post-processing subsequence recovery strategy. (6) Ours w/ GPT2-small, which uses the GPT2-small model as the  $\mathcal{M}_S$ .

Table 3, 4, and 7 shows the results of the ablation study in difference tasks. In summary, removing any component proposed for LongLLMLingua will

lead to a performance drop regardless of the position of the ground-truth answer. This well validates the necessity and effectiveness of the proposed question-aware mechanism during coarse-to-fine compression, the dynamic compression ratio, and the subsequence recovery strategy. It also shows that applying SBERT for coarse-grained compression will result in inferior performance, which implies the superiority of our question-aware importance metric in Eq. (2) over SBERT. In addition, replacing  $p(x_i^{\text{que,restrict}}|\mathbf{x}_k^{\text{doc}})$  with  $p(\mathbf{x}_k^{\text{doc}}|x_i^{\text{que,restrict}})$ can greatly affect performance due to the large noise in calculating  $p(\mathbf{x}_k^{\text{doc}})$  since the perplexity of document depends on many other information besides the question. Removing the restrictive statement can increase the hallucination of small language models, leading to a decrease in performance. Moreover, our subsequence recovery strategy can also bring performance gains for LLMLingua. However, without our question-aware mechanism, results from LLMLingua are still less satisfactory. For more detailed cases, please go to Appendix E.

<span id="page-7-0"></span>

| Methods                   | SingleDoc | MultiDoc | Summ. | FewShot   | Synth.   | Code | AVG  | Tokens | $1/\tau$ | Latency | Speedup |
|---------------------------|-----------|----------|-------|-----------|----------|------|------|--------|----------|---------|---------|
|                           |           |          | 3,000 | tokens co | nstraint |      |      |        |          |         |         |
| Retrieval-based Meth      | ods       |          |       |           |          |      |      |        |          |         |         |
| BM25                      | 32.3      | 34.3     | 25.3  | 57.9      | 45.1     | 48.9 | 40.6 | 3,417  | 3x       | 7.5     | 2.1x    |
| SBERT                     | 35.3      | 37.4     | 26.7  | 63.4      | 51.0     | 34.5 | 41.4 | 3,399  | 3x       | 7.7     | 2.0x    |
| OpenAI                    | 34.5      | 38.6     | 26.8  | 63.4      | 49.6     | 37.6 | 41.7 | 3,421  | 3x       | 13.3    | 1.2x    |
| LongLLMLingua $r_k$       | 37.6      | 42.9     | 26.9  | 68.2      | 49.9     | 53.4 | 46.5 | 3,424  | 3x       | 8.2     | 1.9x    |
| Compression-based Methods |           |          |       |           |          |      |      |        |          |         |         |
| Selective-Context         | 23.3      | 39.2     | 25.0  | 23.8      | 27.5     | 53.1 | 32.0 | 3,328  | 3x       | 50.6    | 0.3x    |
| LLMLingua                 | 31.8      | 37.5     | 26.2  | 67.2      | 8.3      | 53.2 | 37.4 | 3,421  | 3x       | 9.2     | 1.7x    |
| LongLLMLingua             | 40.7      | 46.2     | 27.2  | 70.6      | 53.0     | 55.2 | 48.8 | 3,283  | 3x       | 10.0    | 1.6x    |
| 2,000 tokens constraint   |           |          |       |           |          |      |      |        |          |         |         |
| Retrieval-based Meth      | nods      |          |       |           |          |      |      |        |          |         |         |
| BM25                      | 30.1      | 29.4     | 21.2  | 19.5      | 12.4     | 29.1 | 23.6 | 1,985  | 5x       | 4.6     | 3.4x    |
| SBERT                     | 33.8      | 35.9     | 25.9  | 23.5      | 18.0     | 17.8 | 25.8 | 1,947  | 5x       | 4.8     | 3.4x    |
| OpenAI                    | 34.3      | 36.3     | 24.7  | 32.4      | 26.3     | 24.8 | 29.8 | 1,991  | 5x       | 10.4    | 1.5x    |
| LongLLMLingua $r_k$       | 37.8      | 41.7     | 26.9  | 66.3      | 53.0     | 52.4 | 46.3 | 1,960  | 5x       | 4.7     | 3.3x    |
| Compression-based I       | Methods   |          |       |           |          |      |      |        |          |         |         |
| Selective-Context         | 16.2      | 34.8     | 24.4  | 15.7      | 8.4      | 49.2 | 24.8 | 1,925  | 5x       | 47.1    | 0.3x    |
| LLMLingua                 | 22.4      | 32.1     | 24.5  | 61.2      | 10.4     | 56.8 | 34.6 | 1,950  | 5x       | 5.9     | 2.6x    |
| LongLLMLingua             | 39.9      | 43.2     | 27.4  | 69.8      | 53.0     | 56.7 | 48.3 | 1,822  | 6x       | 6.1     | 2.6x    |
| Original Prompt           | 39.7      | 38.7     | 26.5  | 67.0      | 37.8     | 54.2 | 44.0 | 10,295 | -        | 15.6    | -       |
| Zero-shot                 | 15.6      | 31.3     | 15.6  | 40.7      | 1.6      | 36.2 | 23.5 | 214    | 48x      | 1.6     | 9.8x    |

Table 2: Performance of different methods under different compression ratios on LongBench (Bai et al., 2023) using GPT-3.5-Turbo in 2,000 tokens constraint.

<span id="page-7-1"></span>

|                                                                      | 1st  | 5th  | 10th | 15th | 20th |
|----------------------------------------------------------------------|------|------|------|------|------|
| LongLLMLingua                                                        | 77.2 | 72.9 | 70.8 | 70.5 | 70.6 |
| Question-aware Coarse-grained                                        |      |      |      |      |      |
| - w/o Question-awareness                                             | 42.1 | 40.3 | 39.7 | 40.1 | 40.3 |
| - w/ SBERT                                                           | 73.2 | 68.5 | 65.7 | 66.1 | 66.7 |
| - w/ $p(\mathbf{x}_k^{\text{doc}} x_i^{\text{que},\text{restrict}})$ | 56.0 | 52.6 | 53.4 | 51.6 | 51.1 |
| - w/o restrict                                                       | 75.1 | 72.2 | 70.3 | 70.3 | 70.2 |
| - w/o Question-aware Fine-grained                                    | 75.8 | 71.0 | 68.9 | 68.4 | 69.3 |
| - w/o Dynamic Compression Ratio                                      | 74.4 | 70.7 | 68.7 | 67.9 | 68.1 |
| - w/o Subsequence Recovery                                           | 76.7 | 71.7 | 69.4 | 69.3 | 69.7 |
| - w/ Document Reordering                                             | 76.2 | 76.2 | 76.2 | 76.2 | 76.2 |
| - w/ GPT2-small                                                      | 74.6 | 71.7 | 70.1 | 69.8 | 68.5 |
| LLMLingua                                                            | 39.7 | 39.5 | 40.4 | 37.1 | 42.3 |
| - w/ Subsequence Recovery                                            | 43.8 | 44.1 | 43.5 | 43.3 | 44.4 |

Table 3: Ablation study on NaturalQuestions with 2x constraint using GPT-3.5-Turbo.

Latency evaluation We conducte end-to-end latency testing on a V100-32G, using the prompts from Multi-document QA, LongBench, and Zero-SCROLLS in the API call, and results are shown in Table 1, 2 and 6. The latency includes the time cost for prompt compression and the request time for LLMs, with multiple measurements taken and averaged over. Results demonstrate that LongLLM-Lingua does indeed speed up the overall inference

under different compression ratios and scenarios. Moreover, with the compression ratio increasing, the acceleration effect becomes more pronounced up to 2.6x. However, the OpenAI embedding and Selective-Context results in longer latency time, due to repeated API calls and the sequential entropy calculation of semantic units, respectively.

#### **6** Related Works

Long context for LLMs. Recent research has focused on expanding the window size of LLMs. Main approaches include: (1) Staged pre-training (Nijkamp et al., 2023) which gradually increases the context window; (2) Modifying (Press et al., 2022) or interpolating position embeddings (Chen et al., 2023; Peng et al., 2024); (3) Using linear or sparse attention mechanisms (Ding et al., 2023; Sun et al., 2023); (4) Utilizing external memory modules for context storage (Bertsch et al., 2023; Tworkowski et al., 2023). While these methods address context window expansion, their impact on downstream task performance has yet to be discussed.

**Information distribution in prompt**. Recent empirical experiments have shown that LLM performance decreases with less effective information

in a prompt [\(Bai et al.,](#page-8-1) [2023;](#page-8-1) [Li et al.,](#page-9-13) [2023a;](#page-9-13) [Shi](#page-10-4) [et al.,](#page-10-4) [2023\)](#page-10-4). Moreover, the position of relevant information in a prompt has a significant impact on performance [\(Wu et al.,](#page-10-15) [2023b\)](#page-10-15). [Liu et al.](#page-10-5) [\(2024\)](#page-10-5) suggests that LLMs have more difficulty comprehending information located in the middle of a prompt compared to those at the edges.

Retrieval methods can be categorized as dense or sparse retrieval methods. Sparse retrieval methods, like BM25, determine the relevance between queries and documents based on n-gram information. Conversely, dense retrieval methods assess the relevance between queries and documents in latent space using embedding model [\(Reimers and](#page-10-9) [Gurevych,](#page-10-9) [2019;](#page-10-9) [Xiao et al.,](#page-10-8) [2023;](#page-10-8) [Günther et al.,](#page-9-8) [2023\)](#page-9-8) and reranker model [\(Xiao et al.,](#page-10-8) [2023\)](#page-10-8). Recently, [Jiang et al.](#page-9-7) [\(2023b\)](#page-9-7) proposed an unsupervised dense retrieval method that leverages traditional compression algorithms, such as gzip, and k-nearest neighbors.

Prompt compression methods can be grouped into three main categories: (1) Token pruning [\(Goyal et al.,](#page-9-14) [2020;](#page-9-14) [Kim and Cho,](#page-9-15) [2021;](#page-9-15) [Modar](#page-10-16)[ressi et al.,](#page-10-16) [2022\)](#page-10-16) and token merging [\(Bolya et al.,](#page-8-3) [2023\)](#page-8-3), which need model fine-tuning or intermediate results during inference and have been used with BERT-scale models. (2) Soft prompt tuning methods like GIST [\(Mu et al.,](#page-10-17) [2023\)](#page-10-17), AutoCompressor [\(Chevalier et al.,](#page-9-16) [2023\)](#page-9-16), and ICAE [\(Ge](#page-9-17) [et al.,](#page-9-17) [2024\)](#page-9-17), which require LLMs' parameter finetuning, making them suitable for specific domains but not directly applicable to black-box LLMs. (3) Information-entropy-based approaches such as Selective Context [\(Li et al.,](#page-9-4) [2023c\)](#page-9-4) and LLMLingua [\(Jiang et al.,](#page-9-3) [2023a\)](#page-9-3), which use a small language model to calculate the self-information or perplexity of each token in the original prompt and then remove tokens with lower perplexities.

## 7 Conclusion

We propose LongLLMLingua to address the three challenges, *i.e.*, higher computational cost, performance reduction, and position bias for LLMs in long context scenarios. We develop LongLLMLingua from the perspective of efficient prompt compression, thus reducing computational cost. We further design four components, *i.e.*, a questionaware coarse-to-fine compression method, a document reordering mechanism, dynamic compression ratios, and a subsequence recovery strategy to improve LLMs' perception of the key information, with which LongLLMLingua demonstrate superior performance. Experiments on the multidocument QA, multi-hop QA, and long context benchmarks demonstrate that LongLLMLingua compressed prompt can derive higher performance than original prompts while both API costs for inference and the end-to-end system latency are largely reduced.

## Limitation

Although previous experiments demonstrate LongLLMLingua's effectiveness and efficiency across a broad range of tasks, the method still has the following limitations: 1) LongLLMLingua is a question-aware approach, meaning it requires re-compression for different questions, even with the same context, preventing caching of the context. Moreover, in terms of computational cost, LongLLMLingua increases the computation by twice as much as LLMLingua. This can lead to greater overhead in real-world applications. However, this issue can be mitigated by extending the question-aware approach to a task-aware approach, allowing for reuse and caching. 2) While the effectiveness of LongLLMLingua has been tested on a wide range of tasks, especially on the multi-hop QA dataset MuSicQue [\(Trivedi](#page-10-7) [et al.,](#page-10-7) [2022\)](#page-10-7), its effectiveness might be impacted when the relationship between context and prompt is more complex and subtle due to the coarse-level question-aware approach.

## References

<span id="page-8-0"></span>Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. [Self-RAG: Learning to](https://openreview.net/forum?id=hSyW5go0v8) [retrieve, generate, and critique through self-reflection.](https://openreview.net/forum?id=hSyW5go0v8) In *The Twelfth International Conference on Learning Representations*.

<span id="page-8-1"></span>Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. 2023. [Longbench:](https://arxiv.org/abs/2308.14508) [A bilingual, multitask benchmark for long context](https://arxiv.org/abs/2308.14508) [understanding.](https://arxiv.org/abs/2308.14508) *ArXiv preprint*, abs/2308.14508.

<span id="page-8-2"></span>Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew R. Gormley. 2023. [Unlimiformer: Long](https://openreview.net/forum?id=lJWUJWLCJo)[range transformers with unlimited length input.](https://openreview.net/forum?id=lJWUJWLCJo) In *Thirty-seventh Conference on Neural Information Processing Systems*.

<span id="page-8-3"></span>Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2023. [Token merging: Your vit but faster.](https://openreview.net/forum?id=JroZRaRw7Eu) In *The*

- *Eleventh International Conference on Learning Representations*.
- <span id="page-9-10"></span>Harrison Chase. 2022. [LangChain.](https://github.com/hwchase17/langchain)
- <span id="page-9-11"></span>Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. [Extending context window of](https://arxiv.org/abs/2306.15595) [large language models via positional interpolation.](https://arxiv.org/abs/2306.15595) *ArXiv preprint*, abs/2306.15595.
- <span id="page-9-16"></span>Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. [Adapting language models to](https://doi.org/10.18653/v1/2023.emnlp-main.232) [compress contexts.](https://doi.org/10.18653/v1/2023.emnlp-main.232) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 3829–3846, Singapore. Association for Computational Linguistics.
- <span id="page-9-9"></span>Kenneth Ward Church and Patrick Hanks. 1989. [Word](https://doi.org/10.3115/981623.981633) [association norms, mutual information, and lexicog](https://doi.org/10.3115/981623.981633)[raphy.](https://doi.org/10.3115/981623.981633) In *27th Annual Meeting of the Association for Computational Linguistics*, pages 76–83, Vancouver, British Columbia, Canada. Association for Computational Linguistics.
- <span id="page-9-6"></span>Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, et al. 2023. [Lan](https://arxiv.org/abs/2309.10668)[guage modeling is compression.](https://arxiv.org/abs/2309.10668) *ArXiv preprint*, abs/2309.10668.
- <span id="page-9-12"></span>Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, and Furu Wei. 2023. [Longnet: Scaling transformers to 1,000,000,000 to](https://arxiv.org/abs/2307.02486)[kens.](https://arxiv.org/abs/2307.02486) *ArXiv preprint*, abs/2307.02486.
- <span id="page-9-0"></span>Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2023. [A survey for in-context learning.](https://arxiv.org/abs/2301.00234) *ArXiv preprint*, abs/2301.00234.
- <span id="page-9-17"></span>Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. [In-context autoencoder for con](https://openreview.net/forum?id=uREj4ZuGJE)[text compression in a large language model.](https://openreview.net/forum?id=uREj4ZuGJE) In *The Twelfth International Conference on Learning Representations*.
- <span id="page-9-14"></span>Saurabh Goyal, Anamitra Roy Choudhury, Saurabh Raje, Venkatesan T. Chakaravarthy, Yogish Sabharwal, and Ashish Verma. 2020. [Power-bert: Accel](http://proceedings.mlr.press/v119/goyal20a.html)[erating BERT inference via progressive word-vector](http://proceedings.mlr.press/v119/goyal20a.html) [elimination.](http://proceedings.mlr.press/v119/goyal20a.html) In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pages 3690–3699. PMLR.
- <span id="page-9-8"></span>Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, Maximilian Werk, Nan Wang, and Han Xiao. 2023. [Jina embeddings 2: 8192-token](https://arxiv.org/abs/2310.19923) [general-purpose text embeddings for long documents.](https://arxiv.org/abs/2310.19923) *ArXiv preprint*, abs/2310.19923.

- <span id="page-9-19"></span>Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. [Unsupervised dense informa](https://openreview.net/forum?id=jKN1pXi7b0)[tion retrieval with contrastive learning.](https://openreview.net/forum?id=jKN1pXi7b0) *Transactions on Machine Learning Research*.
- <span id="page-9-3"></span>Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023a. [LLMLingua: Compress](https://doi.org/10.18653/v1/2023.emnlp-main.825)[ing prompts for accelerated inference of large lan](https://doi.org/10.18653/v1/2023.emnlp-main.825)[guage models.](https://doi.org/10.18653/v1/2023.emnlp-main.825) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 13358–13376. Association for Computational Linguistics.
- <span id="page-9-7"></span>Zhiying Jiang, Matthew Yang, Mikhail Tsirlin, Raphael Tang, Yiqin Dai, and Jimmy Lin. 2023b. ["low](https://doi.org/10.18653/v1/2023.findings-acl.426)[resource" text classification: A parameter-free clas](https://doi.org/10.18653/v1/2023.findings-acl.426)[sification method with compressors.](https://doi.org/10.18653/v1/2023.findings-acl.426) In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 6810–6828, Toronto, Canada. Association for Computational Linguistics.
- <span id="page-9-2"></span>Greg Kamradt. 2023. [Needle In A Haystack - Pressure](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) [Testing LLMs.](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
- <span id="page-9-15"></span>Gyuwan Kim and Kyunghyun Cho. 2021. [Length](https://doi.org/10.18653/v1/2021.acl-long.508)[adaptive transformer: Train once with length drop,](https://doi.org/10.18653/v1/2021.acl-long.508) [use anytime with search.](https://doi.org/10.18653/v1/2021.acl-long.508) In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 6501–6511, Online. Association for Computational Linguistics.
- <span id="page-9-18"></span>Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. [Natu](https://doi.org/10.1162/tacl_a_00276)[ral questions: A benchmark for question answering](https://doi.org/10.1162/tacl_a_00276) [research.](https://doi.org/10.1162/tacl_a_00276) *Transactions of the Association for Computational Linguistics*, 7:452–466.
- <span id="page-9-1"></span>Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. [Retrieval-augmented generation for](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) [knowledge-intensive NLP tasks.](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*.
- <span id="page-9-13"></span>Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang. 2023a. [How long can open](https://lmsys.org/blog/2023-06-29-longchat)[source llms truly promise on context length?](https://lmsys.org/blog/2023-06-29-longchat)
- <span id="page-9-5"></span>Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. 2023b. [Loogle: Can long-context language](https://arxiv.org/abs/2311.04939) [models understand long contexts?](https://arxiv.org/abs/2311.04939) *ArXiv preprint*, abs/2311.04939.
- <span id="page-9-4"></span>Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023c. [Compressing context to enhance inference](https://doi.org/10.18653/v1/2023.emnlp-main.391)

- [efficiency of large language models.](https://doi.org/10.18653/v1/2023.emnlp-main.391) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 6342–6353, Singapore. Association for Computational Linguistics.
- <span id="page-10-5"></span>Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. [Lost in the Middle: How Language](https://doi.org/10.1162/tacl_a_00638) [Models Use Long Contexts.](https://doi.org/10.1162/tacl_a_00638) *Transactions of the Association for Computational Linguistics*, 12:157–173.
- <span id="page-10-0"></span>Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. [MetaICL: Learning to learn](https://doi.org/10.18653/v1/2022.naacl-main.201) [in context.](https://doi.org/10.18653/v1/2022.naacl-main.201) In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2791–2809, Seattle, United States. Association for Computational Linguistics.
- <span id="page-10-16"></span>Ali Modarressi, Hosein Mohebbi, and Mohammad Taher Pilehvar. 2022. [AdapLeR: Speeding up](https://doi.org/10.18653/v1/2022.acl-long.1) [inference by adaptive length reduction.](https://doi.org/10.18653/v1/2022.acl-long.1) In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1–15, Dublin, Ireland. Association for Computational Linguistics.
- <span id="page-10-17"></span>Jesse Mu, Xiang Lisa Li, and Noah Goodman. 2023. [Learning to compress prompts with gist tokens.](https://openreview.net/forum?id=2DtxPCL3T5) In *Thirty-seventh Conference on Neural Information Processing Systems*.
- <span id="page-10-10"></span>Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Krysci ´ nski, Lidiya Mu- ´ rakhovs'ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, Chien-Sheng Wu, Silvio Savarese, Yingbo Zhou, Shafiq Joty, and Caiming Xiong. 2023. [Xgen-7b technical](https://arxiv.org/abs/2309.03450) [report.](https://arxiv.org/abs/2309.03450) *ArXiv preprint*, abs/2309.03450.
- <span id="page-10-2"></span>Joon Sung Park, Joseph O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. [Generative agents: Interactive simulacra](https://doi.org/10.1145/3586183.3606763) [of human behavior.](https://doi.org/10.1145/3586183.3606763) In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*, UIST '23, New York, NY, USA. Association for Computing Machinery.
- <span id="page-10-12"></span>Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. [YaRN: Efficient context window ex](https://openreview.net/forum?id=wHBfxhZu1u)[tension of large language models.](https://openreview.net/forum?id=wHBfxhZu1u) In *The Twelfth International Conference on Learning Representations*.
- <span id="page-10-11"></span>Ofir Press, Noah A. Smith, and Mike Lewis. 2022. [Train](https://openreview.net/forum?id=R8sQPpGCv0) [short, test long: Attention with linear biases enables](https://openreview.net/forum?id=R8sQPpGCv0) [input length extrapolation.](https://openreview.net/forum?id=R8sQPpGCv0) In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net.
- <span id="page-10-9"></span>Nils Reimers and Iryna Gurevych. 2019. [Sentence-](https://doi.org/10.18653/v1/D19-1410)[BERT: Sentence embeddings using Siamese BERT](https://doi.org/10.18653/v1/D19-1410)[networks.](https://doi.org/10.18653/v1/D19-1410) In *Proceedings of the 2019 Conference on*

- *Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.
- <span id="page-10-6"></span>Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. 2023. [ZeroSCROLLS: A zero-shot](https://doi.org/10.18653/v1/2023.findings-emnlp.536) [benchmark for long text understanding.](https://doi.org/10.18653/v1/2023.findings-emnlp.536) In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 7977–7989, Singapore. Association for Computational Linguistics.
- <span id="page-10-1"></span>Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2024. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-10-4"></span>Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In *International Conference on Machine Learning*, pages 31210–31227. PMLR.
- <span id="page-10-13"></span>Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. 2023. [Retentive network: A successor to trans](https://arxiv.org/abs/2307.08621)[former for large language models.](https://arxiv.org/abs/2307.08621) *ArXiv preprint*, abs/2307.08621.
- <span id="page-10-7"></span>Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. [MuSiQue: Multi](https://doi.org/10.1162/tacl_a_00475)[hop questions via single-hop question composition.](https://doi.org/10.1162/tacl_a_00475) *Transactions of the Association for Computational Linguistics*, 10:539–554.
- <span id="page-10-14"></span>Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miłos. 2023. ´ [Focused transformer: Contrastive train](https://openreview.net/forum?id=s1FjXzJ0jy)[ing for context scaling.](https://openreview.net/forum?id=s1FjXzJ0jy) In *Thirty-seventh Conference on Neural Information Processing Systems*.
- <span id="page-10-3"></span>Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. 2023a. [Autogen: Enabling next-gen llm applica](https://arxiv.org/abs/2308.08155)[tions via multi-agent conversation framework.](https://arxiv.org/abs/2308.08155) *ArXiv preprint*, abs/2308.08155.
- <span id="page-10-15"></span>Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2023b. [Self-adaptive in-context learn](https://doi.org/10.18653/v1/2023.acl-long.79)[ing: An information compression perspective for in](https://doi.org/10.18653/v1/2023.acl-long.79)[context example selection and ordering.](https://doi.org/10.18653/v1/2023.acl-long.79) In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1423–1436, Toronto, Canada. Association for Computational Linguistics.
- <span id="page-10-8"></span>Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. [C-pack: Packaged resources to](https://arxiv.org/abs/2309.07597) [advance general chinese embedding.](https://arxiv.org/abs/2309.07597) *ArXiv preprint*, abs/2309.07597.

<span id="page-11-2"></span>Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. 2024. [Retrieval meets long context large lan](https://openreview.net/forum?id=xw5nxFWMlo)[guage models.](https://openreview.net/forum?id=xw5nxFWMlo) In *The Twelfth International Conference on Learning Representations*.

## <span id="page-11-0"></span>A Derivation Of Question-Aware Fine-Grained Compression

Based on the definition of Eq. [\(3\)](#page-3-4), we can derive that,

$$s_{i} = \operatorname{perplexity}(x_{i}|x_{< i}) - \operatorname{perplexity}(x_{i}|x^{\operatorname{que}}, x_{< i})$$

$$= q(x_{i}) \log p(x_{i}|x^{\operatorname{que}}, x_{< i}) - q(x_{i}) \log p(x_{i}|x_{< i})$$

$$= q(x_{i}) \log \frac{p(x_{i}|x^{\operatorname{que}}, x_{< i})}{p(x_{i}|x_{< i})}$$
(6)

In the actual calculation of perplexity, a log operation is performed to avoid overflow, and q(xi) represents the probability distribution of the groundtruth.

At the same time, we can derive the following expanded expression based on Bayes' theorem.

<span id="page-11-3"></span>
$$p(x^{\text{que}}|x_{i}, x_{< i}) = \frac{p(x_{i}|x^{\text{que}}, x_{< i})p(x^{\text{que}})}{p(x_{i}|x_{< i})}$$

$$= p(x^{\text{que}})\frac{p(x_{i}|x^{\text{que}}, x_{< i})}{p(x_{i}|x_{< i})}$$
(7)

The probability distribution p(x que) of the question and the ground-truth distribution q(xi) of x<sup>i</sup> are constants, hence s<sup>i</sup> can be considered as the representation of Eq. [\(7\)](#page-11-3).

$$s_i \propto p(x^{\text{que}}|x_i, x_{< i})$$
 (8)

So we can utilize Eq. [\(3\)](#page-3-4) to represent the probability distribution p(x que|x<sup>i</sup> , x<i), which represents the condition likelihood of generating x que given the token x<sup>i</sup> . Therefore, we can represent the tokenlevel sensitive distribution for the question x que using just a single inference. For tokens that are unrelated to x que, such as the tokens on the right side of Figure [3b,](#page-3-3) their original amount of information may be high, but the contrastive perplexity remains at a relatively low level. Finally, we observe that the form of contrastive perplexity is equivalent to conditional pointwise mutual information [\(Church](#page-9-9) [and Hanks,](#page-9-9) [1989\)](#page-9-9).

## <span id="page-11-1"></span>B Experiment Details

## B.1 Dataset Details

We use NaturalQuestions [\(Liu et al.,](#page-10-5) [2024\)](#page-10-5) for the multi-document QA task, MuSicQue [\(Trivedi et al.,](#page-10-7) [2022\)](#page-10-7) for the multi-hop QA task, and use Long-Bench [\(Bai et al.,](#page-8-1) [2023\)](#page-8-1), ZeroSCROLLS [\(Shaham](#page-10-6) [et al.,](#page-10-6) [2023\)](#page-10-6), LooGLE [\(Li et al.,](#page-9-5) [2023b\)](#page-9-5) for general long context scenarios. The specific details of the dataset are as follows:

NaturalQuestions multi-document QA A multi-document question-answering dataset, comprising 2,655 problems, was built by [\(Liu](#page-10-5) [et al.,](#page-10-5) [2024\)](#page-10-5) based on the NaturalQuestions dataset [\(Kwiatkowski et al.,](#page-9-18) [2019\)](#page-9-18). This dataset provides a realistic retrieval-augmented generation setup that closely resembles commercial search and question-answering applications (e.g., Bing Chat). Each example in the dataset contains a question and k related documents, utilizing the Contriever retrieval system [\(Izacard et al.,](#page-9-19) [2022\)](#page-9-19), one of which includes a document with the correct answer. To perform this task, the model must access the document containing the answer within its input context and use it to answer the question. The dataset's data is sourced from the NaturalQuestions dataset, which contains historical queries issued to the Google search engine and human-annotated answers extracted from Wikipedia. The average prompt token length in this benchmark is 2,946. For our experiments, we used the version provided by [\(Liu et al.,](#page-10-5) [2024\)](#page-10-5) that includes 20 documents[10](#page-11-4). The dataset comprises five different ground truth document position settings in the prompt: 1st, 5th, 10th, 15th, and 20th.

LongBench A multi-task long context benchmark consists of 3,750 problems in English and includes six categories with a total of 16 tasks. These tasks encompass key long-text application scenarios, such as single-document QA, multi-document QA, summarization, few-shot learning, synthetic tasks, and code completion. The average prompt token length in this benchmark is 10,289. For our experiments, we used the English dataset and evaluation scripts provided by [\(Bai et al.,](#page-8-1) [2023\)](#page-8-1) for this benchmark[11](#page-11-5) .

ZeroSCROLLS The multi-task long context benchmark consists of 4,378 problems, including four categories with a total of 10 tasks. These tasks cover summarization, question answering, aggregated sentiment classification, and information reordering. The average prompt token length in this

<span id="page-11-4"></span><sup>10</sup>https://github.com/nelson-liu/lost-in-the-middle

<span id="page-11-5"></span><sup>11</sup>https://github.com/THUDM/LongBench

benchmark is 9,788. For our experiments, we used the validation set and evaluation scripts provided by [\(Shaham et al.,](#page-10-6) [2023\)](#page-10-6) for this dataset[12](#page-12-2) .

MuSiQue The multi-hop question-answer dataset is composed of 39,876, 4,834, and 4,918 problems in the training, validation, and testing datasets, respectively. This dataset requires the language model to conduct multiple inferences based on the content of several documents and provide corresponding answers, thereby necessitating a certain capability for global information processing. The average token length for prompts in this dataset is 2,477. For our experiments, we utilized the validation set and evaluation scripts provided by [\(Trivedi et al.,](#page-10-7) [2022\)](#page-10-7) for this dataset[13](#page-12-3) .

LooGLE The multi-task long context benchmark comprises 6,448 problems, divided into three categories: summarization, short dependency question answering, and long dependency question answering. The average prompt token length in this benchmark stands at 24,005. For our experiments, we focused on the long dependency question answering subset, which includes four types of tasks: information retrieval, timeline reordering, computation, and comprehension. This subset contains 1,101 problems. We utilized the evaluation scripts provided by [\(Li et al.,](#page-9-5) [2023b\)](#page-9-5) for this dataset[14](#page-12-4) .

#### B.2 Other Implementation Details

All experiments were conducted using a Tesla V100 (32GB). We use tiktoken[15](#page-12-5) and GPT-3.5- Turbo model to count all the tokens. We set the granular control coefficient k to 2. We use the pre-defined compression rates τins = 0.85 and τque = 0.9 for instructions and questions. The segment size used in the iterative token-level compression is set to 200. The δτ used in dynamic compression ratio is set to 0.3. For a fair comparison, we only used reordering in the NaturalQuestions Multi-document QA and noted this in Table [1.](#page-6-0) We use "*We can get the answer to this question in the given documents.*" as the guideline sentence in Eq. [\(3\)](#page-3-4).

For the baselines experiment, we use the currently recommended strongest model, all-mpnetbase-v2[16](#page-12-6), as the dense representation model for

SentenceBERT. We use the recommended "textembedding-ada-002" as the embedding model for OpenAI Embedding[17](#page-12-7). We use the GPT2-dolly[18](#page-12-8) as the small language model in w/ GPT2-small ablation experiments.

## <span id="page-12-1"></span>C Additional Experimental Results

## <span id="page-12-0"></span>C.1 Empirical Study of Question-aware Fine-grained Compression

Figure [5](#page-13-1) shows the distribution of the document's average perplexity when the ground-truth is located at more positions within the prompt. As can be observed, as the context length increases, the original perplexity curve remains relatively stable. In unrelated documents, a higher perplexity is still retained, making it easier to remove relevant tokens from the related documents in the prompt compression process, thereby damaging the corresponding semantic information. Contrarily, contrastive perplexity shows an increase in perplexity in documents related to the question. According to the theoretical derivation in Appendix [A,](#page-11-0) it's known that contrastive perplexity characterizes the conditional probability of tokens corresponding to the question. The higher the relevance, the higher the contrastive perplexity, thereby retaining key information in the prompt compression process.

#### C.2 Ablation in LongBench

Table [4](#page-13-0) presents the results from the ablation experiment in the LongBench long context benchmark. It can be observed that in various long context tasks: 1) Removing the question-aware coarse-grained, question-aware fine-grained, dynamic compression ratio, document reordering, and subsequence recovery proposed by LongLLMLingua all result in different degrees of performance drop. 2) Among these, question-aware coarse-grained is particularly important for document-based QA and synthetic tasks, with the maximum drop being 35.8 points; its impact on summarization and code tasks is relatively smaller. 3) The design of the conditional probability in the question-aware coarse-grained module improves the results in all tasks, including code completion, single-document questionanswer, and synthetic tasks. Changing the order of conditional probabilities or removing the restrict prompt both lead to varying degrees of performance decline. 4) Removing question-aware fine-

<span id="page-12-2"></span><sup>12</sup>https://www.zero.scrolls-benchmark.com/

<span id="page-12-3"></span><sup>13</sup>https://github.com/stonybrooknlp/musique

<span id="page-12-4"></span><sup>14</sup>https://github.com/bigai-nlco/LooGLE

<span id="page-12-5"></span><sup>15</sup>https://github.com/openai/tiktoken

<span id="page-12-6"></span><sup>16</sup>https://www.sbert.net/docs/pretrained\_models.html

<span id="page-12-7"></span><sup>17</sup>https://platform.openai.com/docs/guides/embeddings/

<span id="page-12-8"></span><sup>18</sup>https://huggingface.co/lgaalves/gpt2-dolly

<span id="page-13-1"></span>> **[图片提取文字 (无描述)]:**
> 1.0 Perplexity 8.0 Document Avg. Perplexity 0.0 0.0 8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0. Perpley Document Avg. P 0.0 0.0 0.0 0.6غ Document A 0.0 0.0 0.0 Perplexity Perplexity Perplexity Contrastive Perplexity Contrastive Perplexity Contrastive Perplexity 20 20 20 Document Position in the Prompt Document Position in the Prompt Document Position in the Prompt (c) 15th (a) 1st (b) 10th
![](_page_13_Figure_0.jpeg)

Figure 5: The distribution of document-level average perplexity when the ground-truth document is in different positions.

<span id="page-13-0"></span>

| Methods                                                       | SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | AVG  | Tokens | $1/\tau$ |
|---------------------------------------------------------------|-----------|----------|-------|---------|--------|------|------|--------|----------|
| LongLLMLingua                                                 | 39.9      | 43.2     | 27.4  | 69.8    | 53.0   | 56.7 | 48.3 | 1,822  | 6x       |
| Question-aware Coarse-grained                                 |           |          |       |         |        |      |      |        |          |
| <ul> <li>w/o Question-awareness</li> </ul>                    | 27.1      | 38.7     | 25.4  | 62.0    | 18.0   | 53.3 | 37.4 | 1,945  | 5x       |
| - w/ SBERT                                                    | 34.0      | 38.7     | 24.1  | 57.9    | 32.5   | 31.1 | 36.4 | 1,790  | 6x       |
| - w/ $p(\mathbf{x}_k^{\text{doc}} x_i^{\text{que,restrict}})$ | 22.5      | 28.9     | 23.2  | 53.0    | 22.5   | 33.3 | 30.6 | 1,794  | 6x       |
| - w/o restrict                                                | 37.8      | 39.5     | 26.4  | 64.8    | 52.5   | 55.8 | 46.1 | 1,834  | 6x       |
| - w/o Question-aware Fine-grained                             | 35.7      | 41.1     | 26.4  | 62.9    | 44.5   | 54.8 | 44.2 | 1,807  | 6x       |
| - w/o Dynamic Compression Ratio                               | 36.1      | 40.6     | 26.9  | 67.2    | 48.0   | 55.8 | 45.7 | 1,851  | 6x       |
| <ul> <li>w/o Subsequence Recovery</li> </ul>                  | 38.6      | 41.8     | 27.3  | 69.0    | 53.8   | 56.6 | 47.8 | 1,809  | 6x       |
| - w/o Document Reordering                                     | 39.0      | 42.2     | 27.4  | 69.3    | 53.8   | 56.6 | 48.0 | 1,809  | 6x       |
| - w/ GPT2-small                                               | 35.9      | 39.4     | 25.0  | 60.6    | 42.0   | 55.4 | 43.0 | 1,892  | 5x       |

Table 4: Ablation on LongBench (Bai et al., 2023) using GPT-3.5-Turbo in 2,000 tokens constraint.

grained, dynamic compression ratio has a more significant impact on document-based QA and synthetic tasks. 5) The subsequence recovery module can enhance reference-based tasks, but its improvement on tasks like summarization, code, synthetic, etc., is relatively smaller. 6) Document reordering is effective for all types of tasks. Reordering at the document level does not affect LLMs' understanding of context information, even for timelinerelated tasks (see timeline reorder in LooGLE, Table 8). On the contrary, reordering can effectively alleviate the "lost in the middle" issue, thereby improving LLMs performance. 7) Using GPT2-small reduces the capture of effective tokens, but it can still achieve results close to or even slightly better than the original prompt.

#### C.3 LongBench Using LongChat-13b-16k

Table 5 presents the experiment results in the Long-Bench long context benchmark using LongChat-13b-16k. It can be seen that the compressed prompt can also achieve good results on other LLMs, such as LongChat-13b-16k. Specifically, 1) there is a maximum improvement of 15.5 points in synthetic tasks. Except for a slight drop in few-shot Learning, there is an improvement of 3-5 points in other

tasks. 2) The performance trends of retrieval-based and compressed-based baselines are similar to the results in GPT-3.5-Turbo.

#### C.4 ZeroSCROLLS

Table 6 presents a detailed performance breakdown on the ZeroSCROLLS benchmark. It can be observed that in the four summarization tasks - GvRp, SSFD, QMsm, SQAL, LongLLMLingua closely matches or slightly surpasses the original results under two compression constraints. Meanwhile, in the four long context QA tasks - Qsqr, Nrtv, QALT, MuSQ, there is a significant improvement. Notably, in the MuSiQue task, which is based on a question-answering dataset from books and movie scripts, there is a 2.1 point increase even under a 2,000 tokens constraint. It's worth mentioning that MuSiQue is a multi-hop question-answering dataset that requires LLMs to utilize global information for long dependency QA. LongLLMLingua can also improve by 3.5 points under a 6x compression ratio. In the two ordering tasks, SpDg and BkSS, LongLLMLingua can better retain globally sensitive information, resulting in a 3.0 point improvement in BkSS after prompt compression.

It's important to note that although the Zero-

<span id="page-14-1"></span>

| Methods                 | SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | AVG  | Tokens | $1/\tau$ |  |  |  |
|-------------------------|-----------|----------|-------|---------|--------|------|------|--------|----------|--|--|--|
| Original Prompt         | 27.4      | 30.3     | 20.3  | 49.9    | 12.5   | 42.5 | 30.5 | 10,295 | -        |  |  |  |
| Retrieval-based Methods |           |          |       |         |        |      |      |        |          |  |  |  |
| BM25                    | 2.4       | 2.6      | 16.4  | 8.7     | 0.0    | 44.7 | 12.5 | 1,985  | 5x       |  |  |  |
| SBERT                   | 11.6      | 13.7     | 21.1  | 16.2    | 7.5    | 30.0 | 16.7 | 1,947  | 5x       |  |  |  |
| LongLLMLingua $r_k$     | 30.3      | 32.4     | 24.5  | 41.0    | 27.5   | 38.1 | 32.3 | 1,960  | 5x       |  |  |  |
| Compression-based 1     | Methods   |          |       |         |        |      |      |        |          |  |  |  |
| Selective-Context       | 16.1      | 23.5     | 21.8  | 21.4    | 2.5    | 35.9 | 20.2 | 1,925  | 5x       |  |  |  |
| LLMLingua               | 20.6      | 22.3     | 22.4  | 35.6    | 0.0    | 35.4 | 22.7 | 1,950  | 5x       |  |  |  |
| LongLLMLingua           | 31.3      | 34.6     | 24.6  | 46.1    | 27.8   | 48.8 | 35.5 | 1,822  | 6x       |  |  |  |

Table 5: Performance of different methods under different compression ratios on LongBench (Bai et al., 2023) using LongChat-13b in 2,000 tokens constraint.

<span id="page-14-0"></span>

| Methods                                     | GvRp                    | SSFD | QMsm | SQAL | QALT | Nrtv    | Qspr   | MuSQ    | SpDg | BkSS | AVG  | Tokens | $1/\tau$ | Latency | Speedup |
|---------------------------------------------|-------------------------|------|------|------|------|---------|--------|---------|------|------|------|--------|----------|---------|---------|
|                                             |                         |      |      |      | 3,00 | 00 toke | ns con | straint |      |      |      |        |          |         |         |
| Retrieval-based Meth                        | ods                     |      |      |      |      |         |        |         |      |      |      |        |          |         |         |
| BM25                                        | 9.7                     | 3.4  | 11.7 | 14.3 | 57.1 | 5.9     | 25.7   | 11.2    | 29.6 | 29.6 | 19.8 | 3,379  | 3x       | 5.5     | 2.2x    |
| SBERT                                       | 16.5                    | 9.8  | 12.3 | 15.2 | 60.0 | 14.6    | 23.4   | 12.1    | 39.4 | 36.4 | 24.0 | 3,340  | 3x       | 5.9     | 2.1x    |
| OpenAI                                      | 14.3                    | 8.3  | 12.0 | 15.3 | 66.7 | 13.3    | 24.3   | 11.7    | 31.2 | 26.4 | 22.4 | 3,362  | 3x       | 11.7    | 1.0x    |
| LongLLMLingua $r_k$                         | 19.5                    | 11.6 | 14.7 | 15.5 | 66.7 | 20.5    | 27.6   | 13.0    | 60.8 | 43.4 | 29.3 | 3,350  | 3x       | 6.2     | 2.0x    |
| Compression-based Methods                   |                         |      |      |      |      |         |        |         |      |      |      |        |          |         |         |
| Selective-Context                           | 20.8                    | 9.1  | 11.7 | 13.4 | 50.0 | 9.8     | 26.1   | 11.0    | 46.0 | 9.5  | 20.7 | 3,460  | 3x       | 54.2    | 0.2x    |
| LLMLingua                                   | 18.7                    | 10.0 | 14.9 | 16.8 | 61.9 | 26.9    | 27.2   | 23.4    | 62.9 | 44.5 | 30.7 | 3,366  | 3x       | 7.4     | 1.7x    |
| LongLLMLingua                               | 22.1                    | 12.8 | 15.9 | 17.1 | 67.0 | 27.8    | 31.3   | 23.9    | 65.8 | 46.5 | 33.0 | 3,431  | 3x       | 8.2     | 1.5x    |
|                                             | 2,000 tokens constraint |      |      |      |      |         |        |         |      |      |      |        |          |         |         |
| Retrieval-based Meth                        | ods                     |      |      |      |      |         |        |         |      |      |      |        |          |         |         |
| BM25                                        | 8.8                     | 2.5  | 11.1 | 13.5 | 60.0 | 7.0     | 4.9    | 20.3    | 39.9 | 32.9 | 20.1 | 1,799  | 5x       | 3.8     | 3.2x    |
| SBERT                                       | 10.2                    | 7.9  | 13.7 | 13.2 | 60.0 | 8.1     | 10.8   | 1.7     | 37.2 | 42.8 | 20.5 | 1,773  | 6x       | 4.1     | 3.0x    |
| OpenAI                                      | 11.1                    | 8.0  | 11.8 | 13.6 | 60.0 | 7.1     | 13.2   | 4.0     | 33.6 | 43.6 | 20.6 | 1,784  | 5x       | 9.9     | 1.2x    |
| Long<br>LLM<br>Lingua<br>$\boldsymbol{r}_k$ | 18.2                    | 9.8  | 12.3 | 15.9 | 57.1 | 10.1    | 17.8   | 7.3     | 57.7 | 42.3 | 24.9 | 1,771  | 6x       | 4.7     | 2.6x    |
| Compression-based I                         | Methods                 |      |      |      |      |         |        |         |      |      |      |        |          |         |         |
| Selective-Context                           | 19.0                    | 8.4  | 9.7  | 12.4 | 47.0 | 12.5    | 21.6   | 11.5    | 41.2 | 11.0 | 19.4 | 1,865  | 5x       | 47.5    | 0.3x    |
| LLMLingua                                   | 19.4                    | 11.9 | 13.1 | 16.0 | 62.1 | 23.7    | 24.0   | 22.4    | 33.9 | 44.9 | 27.2 | 1,862  | 5x       | 4.8     | 0.3x    |
| LongLLMLingua                               | 20.1                    | 12.4 | 14.9 | 16.5 | 65.1 | 27.7    | 30.7   | 23.6    | 68.5 | 47.2 | 32.7 | 1,826  | 6x       | 5.2     | 2.3x    |
| Original Prompt                             | 21.8                    | 12.1 | 17.9 | 17.4 | 66.7 | 25.3    | 29.8   | 20.0    | 69.7 | 44.1 | 32.5 | 9,788  | -        | 12.2    | -       |
| Zero-shot                                   | 9.4                     | 3.0  | 8.6  | 11.4 | 42.9 | 10.6    | 12.4   | 5.5     | 4.2  | 0.0  | 12.8 | 32     | 306x     | 1.0     | 12.2x   |

Table 6: Performance breakdown of different methods under different compression ratios on ZeroSCROLLS (Shaham et al., 2023) using GPT-3.5-Turbo.

Scrolls validation dataset is relatively small, it still demonstrates conclusions similar to previous experimental observations across various methods and tasks. Furthermore, this study conducted an indepth analysis of the multi-hop QA task - MuSiQue, and another long context benchmark - LooGLE. The results can be found in Appendix C.5 and Appendix C.6.

### <span id="page-14-2"></span>C.5 MuSiQue

Table 7 presents the results from the MuSiQue multi-hop question-answer dataset. From the table, it can be observed that in the multi-hop QA task, requiring global information: 1) LongLLMLingua can reduce noise in the prompt by eliminating irrelevant information and putting more related informa-

tion at the beginning or end of the prompt, thereby improving performance by 5.4 points. 2) The performance drop is more pronounced for retrievalbased methods, particularly for n-gram-based methods like BM25. Due to long dependencies, direct matching information is lost, resulting in less relevant information being recalled. 3) The performance of compression-based methods is slightly different. Selective-Context does not distinguish between different modules' sensitivity, resulting in a loss of question and instruction-related information, thereby leading to poorer performance. However, LLMLingua can still retain relevant key information at around a 2x compression ratio. 4) The ablation experiments show that every module designed in LongLLMLingua plays a role in the

<span id="page-15-0"></span>

| Methods                                                       | F1   | Tokens | $1/\tau$ |
|---------------------------------------------------------------|------|--------|----------|
| Original Prompt                                               | 45.8 | 2,427  | -        |
| BM25                                                          | 28.5 | 1,295  | 1.9x     |
| SBERT                                                         | 36.2 | 1,288  | 1.9x     |
| LongLLMLingua $r_k$                                           | 46.3 | 1,295  | 1.9x     |
| Selective-Context                                             | 19.6 | 1,141  | 2.1x     |
| LLMLingua                                                     | 40.1 | 1,110  | 2.2x     |
| LongLLMLingua                                                 | 51.2 | 1,077  | 2.3x     |
| Question-aware Coarse-grained                                 |      |        |          |
| - w/o Question-awareness                                      | 43.2 | 1,076  | 2.3x     |
| - w/ SBERT                                                    | 47.3 | 1,070  | 2.3x     |
| - w/ $p(\mathbf{x}_k^{\text{doc}} x_i^{\text{que,restrict}})$ | 44.0 | 1,066  | 2.3x     |
| - w/o restrict                                                | 49.2 | 1,078  | 2.3x     |
| - w/o Question-aware Fine-grained                             | 48.4 | 1,118  | 2.2x     |
| - w/o Dynamic Compression Ratio                               | 48.2 | 1,090  | 2.2x     |
| - w/o Subsequence Recovery                                    | 50.7 | 1,077  | 2.3x     |
| - w/o Document Reordering                                     | 49.2 | 1,077  | 2.3x     |
| - w/ GPT2-small                                               | 48.4 | 1,095  | 2.2x     |

Table 7: Performance of different methods and ablation study on MuSicQue (Trivedi et al., 2022) with 2x constraint using GPT-3.5-Turbo.

multi-hop task. The removal of the question-aware coarse-grained and w/  $p(\mathbf{x}_k^{\text{doc}}|x_i^{\text{que,restrict}})$  modules, which have difficulty in perceiving the importance distribution of corresponding questions, can cause a drop of up to 8 points. Removing the restrict prompt in the question-aware coarse module can also cause a 2-point drop due to the hallucination issue of small LLM. In addition, removing question-aware fine-grained, dynamic compression ratio, and document reordering can all cause a drop of 0.5-2.8 points. 5) Moreover, if the small language model in LongLLMLingua is replaced with GPT2-small, it can further improve the acceleration ratio and still achieve a result that is 2.6 points better than the original prompt.

#### <span id="page-15-2"></span>C.6 LooGLE

Table 8 presents the experiment results in the LooGLE long dependency benchmark, which features longer prompts (~30k) and more global dependencies. From the table, we can observe that:

1) LongLLMLingua can effectively improve the performance of long context tasks by compressing prompts, even for long dependency tasks. The results show that LongLLMLingua significantly improves performance in tasks such as retrieval, timeline reorder, and computation, with the maximum improvement reaching 15.9 points. 2) The document reorder in LongLLMLingua is effective in all types of tasks, even in tasks highly related to the

timeline, it can effectively improve performance by alleviating the "lost in the middle" issue. 3) Retrieval-based methods tend to lose performance in tasks that have longer dependencies, such as computation and reasoning. 4) For compressionbased methods, due to the difficulty in perceiving question information, there tends to be a larger performance loss in retrieval tasks within long contexts.

#### D Economic Cost

Table 9 presents the estimated per 1,000 samples inference costs for various datasets, encompassing input prompts and generated output text, based on GPT-3.5-Turbo pricing<sup>19</sup>. Our approach demonstrates substantial savings in computational resources and monetary expenses, particularly in long context situations. Cost reductions of \$3.3 (71.7%), \$28.5 (90.5%), \$27.4 (89.5%), \$2.0 (52.6%), and \$88.0 (94.0%) per 1,000 samples are observed for Multi-document QA, LongBench, ZeroScrolls, MuSiQue, and LooGLE, respectively.

## <span id="page-15-1"></span>**E** Ablation Analysis

Figure 6 illustrates the compressed prompts from the Multi-document QA dataset, comparing the use of contrastive perplexity at a high compression ratio (30x). It shows that without question-aware token-level prompt compression, LongLLMLingua tends to compress key information, a tendency that becomes more pronounced at higher compression ratios. Conversely, employing contrastive perplexity allows for better detection of key information related to the question within the context, thus preserving key information within the compressed prompt.

## F Cases Study

Figures 7, 8, and 9 display the outcomes before and after compression, as well as the LLMs' responses in various scenarios.

<span id="page-15-3"></span><sup>19</sup>https://openai.com/pricing

<span id="page-16-0"></span>

| Methods                   |      | Retrieval Timeline Reorder Computation Reasoning AVG Tokens |      |      |      |        | 1/τ  |
|---------------------------|------|-------------------------------------------------------------|------|------|------|--------|------|
| Retrieval-based Methods   |      |                                                             |      |      |      |        |      |
| BM25                      | 20.4 | 21.7                                                        | 8.2  | 26.3 | 19.2 | 3,185  | 10x  |
| SBERT                     | 28.9 | 21.1                                                        | 10.7 | 27.2 | 22.0 | 3,169  | 10x  |
| LongLLMLingua rk          | 38.6 | 32.2                                                        | 16.2 | 26.3 | 28.3 | 3,158  | 10x  |
| Compression-based Methods |      |                                                             |      |      |      |        |      |
| Selective-Context         | 16.7 | 5.0                                                         | 2.3  | 17.6 | 10.4 | 3,710  | 8x   |
| LLMLingua                 | 10.0 | 25.0                                                        | 13.3 | 21.1 | 17.3 | 3,404  | 9x   |
| LongLLMLingua             | 40.0 | 35.0                                                        | 19.7 | 33.6 | 32.1 | 3,121  | 10x  |
| LongLLMLingua w/o Reorder | 39.3 | 33.8                                                        | 18.7 | 31.6 | 30.9 | 3,119  | 10x  |
| Original Prompt           | 24.1 | 20.9                                                        | 13.5 | 32.1 | 22.6 | 30,546 | -    |
| Zero-shot                 | 8.7  | 6.3                                                         | 1.2  | 14.5 | 7.7  | 43     | 710x |

Table 8: Performance of different methods on LooGLE [\(Li et al.,](#page-9-5) [2023b\)](#page-9-5) long dependency QA.

<span id="page-16-1"></span>

|          | Multi-document QA | LongBench    | ZeroScolls   | MuSicQue     | LooGLE       |
|----------|-------------------|--------------|--------------|--------------|--------------|
| Original | 4.6               | 31.5         | 30.6         | 3.8          | 93.6         |
| Ours     | 1.3 (↓71.7%)      | 3.0 (↓90.5%) | 3.2 (↓89.5%) | 1.8 (↓52.6%) | 5.6 (↓94.0%) |

Table 9: The inference costs \$ (per 1,000 samples) for various datasets using GPT-3.5-Turbo.

#### <span id="page-16-2"></span>*Ours w/o Token-level Question-aware:*

### Compressed Prompt:

*Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).*

Document [1](: Physics)gen" who received2K, which is ,73,0 in0. Johnen only to twice6. Mariaie won, for.g was, until1estate he. Two:Mayer (1963). As of 2017, the prize has been awarded *Question: who got the first nobel prize in physics Answer:*

## LLMs' Response:

No answer found in the given search results.

#### *Ours w/ Token-level Question-aware:*

#### Compressed Prompt:

*Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).*

1Title: List of Nobelates in The first Nobel Prize was1 to Wilhelmrad , of who received 1582 which,70 in0 en the prize. Skska also won two Nobeles for physics3g01, theate he women prize:ertMayer (1963). As of 2017, the prize has been awarded

*Question: who got the first nobel prize in physics*

#### *Answer:*

#### LLMs' Response:

Wilhelmrad

#### LLMs' Response after Subsquence Recovery:

Wilhelm Conrad Röntgen

## Ground Truth:

Wilhelm Conrad Röntgen

Figure 6: Comparing the compressed prompt and LLMs' response before and after using Question-aware Finegrained Compression and Subsequence Recovery(1/τ = 30x, high compression ratio setting) from NaturalQuestions Multi-document QA [\(Liu et al.,](#page-10-5) [2024\)](#page-10-5) using GPT-3.5-Turbo.

## <span id="page-17-0"></span>Original Prompt:

...

Document [1](Title: Dancing on Ice) It was confirmed on 25 January 2018, that Dancing on Ice had been recommissioned for an eleventh series to air in 2019 .

...

### Compressed Prompt:

*Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).*

1Title: Dancing on was confirmed on 2 January 2018 that Dancing on had been recommissioned for an eleventh series air in 209 .

Document [2Title: Dan on) Dan on Ice Dancing on British presented by Phillip Schof alongside Holly Willough from 26 to 2011, and Christine Bleakley from 2012 to 204 The show consists of celebrit and professional partners figure skating in front of a panel of judges The, broadcast on ITV, started on January 2006 and ended on 9 March 2014 after showcontract not renewed by ITV ´ On 4 September 2017, it was announced that rev series would on I 7 January 201 Sch and Willby returning as a

5(: on ( on () The third series of a from January to168TV. The from Saturdays, with Holby present Kar,y Sliner Robin Cins returned to Panel", with Ruth H joining the panel as replacement for Natalia Bestova. The commission of the was confirmed by at the07 announcedova depart the series Robinen Bar,ater and Jasoniner announced

7( on ( )) Dan 2 second of Dan on a from January to1207 ITV It presented Phillip Sch Holly Willough, and judged the "I P consisting Nicky Slater, Nataliaian Karenres Jason Gardiner Karen Barber and Robin Cousins Jaynevill and Christopher Dean co and trained the contestants In this series, cele to ten in first series. The series was won former Kyran Bracken, with Mel Lambert the winner. It announced thatenresge

Document []( on Ice on 08 on TV edition started 8 TV2 The Russian version "анду) being on channel0, and renamed in8 to " Ice" (). Its counterpart called "Ice Age (, "Stars on Ice on Channel Oneak IceHviezdyl'J. The Turkish version" is called Dans" ("ance on

Document1 on Ice its, all,é () and Sje Chris de In series.2 edition

](: on Ice world) Dan Ice is a made competition world format, and been subsequently Italy Chile where titled after series There have a, the show was broadcast on Channel 13 as a

Document [17](Title: Dancing on Ice) the insight to the training of the celebrities over the last week. It was presented by television presenter Ben Shephard and former contestant and "Loose Women" star Coleen Nolan. The show was broadcast from 8 pm to 8.30 pm on Friday evenings on ITV throughout the duration of the main shows season. STV who broadcast the main show did not broadcast this on the Friday evening but after repeating the previous weeks main show on the ´ following Saturday afternoon. Due to poor ratings, "Dancing on Ice Friday" was axed prior to the 2011 series. The show was based in the

*Question: when is dancing on ice on the tv*

*Answer:*

LLMs' Response:

209

LLMs' Response after Subsquence Recovery:

2019

Ground Truth:

2019

Figure 7: Cases study on NaturalQuestions Multi-document QA dataset [\(Liu et al.,](#page-10-5) [2024\)](#page-10-5) in 4x constraint using GPT-3.5-Turbo.

```
Compressed Prompt:
Please complete the code given below.
public class MessageArchiveManagement
    private static final long MILLISECONDS_IN_DAY = 24 * 00 *0;
    public static final long_CUP = MCON_DAY
    /.../
          .("",.getStart
          add
 ifget() >0
           Node end("
            end.("
            endNode.Value("", Util.getTimestamp(query.getEnd
addNode
        } if (.withid null && contact null && !isference
           Node with(" .with
           .Value("valuewith
           .(
        // queryMessageive(connection, nextQuery
            final(connectionProtocol(), query
            synchronized (eries)
            // queries.add(nextQuery } }
    public boolean queryInProgress( contact, OnLoaded
    moreMessagesLoadedListener)
       ized (eries)
            (Query query : queries)
                if(query.getWith().equals(contact.getUserId()))
    if (query.onMoreMessagesLoaded == null &&MessagesListener
    null) query.setOnMoreMessagesLoaded(Listener}
                    return true;}} return false;}}
    private void finalizeQuery(Protocol protocol, Query query) {
        synchronized (queries) {
            .remove(query); }
        Contact contact = null;
        if (query.getWith() != null) {
            contact = protocol.getItemByUID(query.getWith()); }
        if (contact != null) {
Next line of code:
LLMs' Response:
        contact.setLastMessageTransmitted(query.getEnd());\n
Ground Truth:
        if (contact.setLastMessageTransmitted(query.getEnd())) {
Zero-shot LLMs' Response:
        contact.removeQuery(query);\n
```

Figure 8: Cases study on lcc code completion task in LongBench benchmark [\(Bai et al.,](#page-8-1) [2023\)](#page-8-1) in 2,000 constraint using GPT-3.5-Turbo.

```
Compressed Prompt:
Please determine the Type of the question below. Here are some examples of questions.
Question: How is energy created ? Type Manner of an action
Question: What is chocolate ? Type: Definition of something
Question: What is a bone marrow transplant ? Type: Definition of something
Question: What is fear of odors , body , ? Type Disease and medicine
Question: What was the Vietnam War ? Type: Definition of something
Question: was education system in 16s ? Type: Other entity
Question: What is IP address ? Type: Definition of something
Question: are the differences in Catholic Methodist religions ? Type of something
...
Question: When was San fire ? : Date
Question: CNN began broadcasting in what year ? Type: Date
Type: Manner of an action
Question: What the l behind the ir in the eye called ? Type Equ term
Type: Date
Question: What the former name of Zimbabwe ? Type: termType something
Question: What is troilism ? Type: Definition of something
: What is origin of the word , Type: of something
: do you name to social security number ? Type Manner of an action
: that of an employee Universal and Export ? Type Individual
: anesthetic did Queen Victoria allow to be for the birth of her seventh , in 183 ? Type: Disease
and medicine
: Where isyer 's rock ? Type location
Question: What isymnophobia ? Type: Definition of something
...
Type burns the most calories ?
Type Sport
: In what book I find story of Aladdin ? Type In, book and piece an have sex ?
Type: Manner of an action: What is the acron for rating forer ?
Type Abbreviation
: are the Baltic States ? Type: Definition of something
: What is appearance , that violates the standards of sexual mor ? Type
: Where did the May people live ? : location
: What population Kansas ? Type number
: was the hurr ? Type: Event
: 's a score aymnast exercise ? Type: number
: year become a state ? Type: Date
do go school ? Type Reason
...
Question: What is a fuel cell ?
Type:
LLMs' Response:
Definition of something
LLMs' Response after Subsquence Recovery:
Definition of something
Ground Truth:
Definition of something
```

Figure 9: Cases study on trec few-show learning in LongBench benchmark [\(Bai et al.,](#page-8-1) [2023\)](#page-8-1) in 2,000 constraint using GPT-3.5-Turbo.