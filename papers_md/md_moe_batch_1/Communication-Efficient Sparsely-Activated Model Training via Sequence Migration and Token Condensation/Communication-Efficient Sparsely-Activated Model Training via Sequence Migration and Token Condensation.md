# Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

Fahao Chen, Peng Li, *Senior Member, IEEE,* Zicong Hong, Zhou Su, *Senior Member, IEEE,* Song Guo, *Fellow, IEEE,*

*Abstract*—Mixture-of-Experts (MoE) is an emerging technique for scaling large models with sparse activation. MoE models are typically trained in a distributed manner with an *expert parallelism* scheme, where experts in each MoE layer are distributed across multiple GPUs. However, the default expert parallelism suffers from the heavy network burden due to the allto-all intermediate data exchange among GPUs before and after the expert run. Some existing works have proposed to reduce intermediate data exchanges by transferring experts to reduce the network loads, however, which would decrease parallelism level of expert execution and make computation inefficient. The weaknesses of existing works motivate us to explore whether it is possible to reduce inter-GPU traffic while maintaining a high degree of expert parallelism. This paper gives a positive response by presenting LUFFY, a communication-efficient distributed MoE training system with two new techniques. First, LUFFY migrates sequences among GPUs to hide heavy token pulling paths within GPUs and avoid copying experts over GPUs. Second, we propose token condensation that identifies similar tokens and then eliminates redundant transmissions. We implement LUFFY based on PyTorch and evaluate its performance on a testbed of 16 V100 GPUs. LUFFY system can achieve a speedup of up to 2.73× compared to state-of-the-art MoE training systems.

*Index Terms*—Mixture-of-Experts, Distributed Training, Parallelism.

# I. INTRODUCTION

The recent success of large-scale language models (LLM), e.g., GPT, has demonstrated that model capability often increases with the growth of model sizes [\[1\]](#page-10-0), [\[2\]](#page-10-1), [\[3\]](#page-10-2), [\[4\]](#page-10-3), however, which comes with a huge computational cost. Mixture-of-Experts (MoE) has been proposed as one of the most popular LLM structures, thanks to its unique sparse activation feature with great promises in reducing computational overhead [\[5\]](#page-10-4), [\[6\]](#page-10-5). It decomposes the dense part of the model into multiple *experts*. Input sentences, also referred to as sequences, are divided into tokens as basic processing units. A gate network routes these tokens to only a few experts instead of all experts.

Fahao Chen is with the University of Aizu, Aizuwakamatsu, Japan (e-mail: d8232101@u-aizu.ac.jp).

Peng Li is with the School of Cyber Science and Engineering, Xi'an Jiaotong University, Xi'an, China (e-mail: pengli@xjtu.edu.cn).

Zicong Hong with the Department of Computing, The Hong Kong Polytechnic University, Hong Kong, China (e-mail: zicong.hong@connect.polyu.hk).

Zhou Su is with the School of Cyber Science and Engineering, Xi'an Jiaotong University, Xi'an, China (e-mail: zhousu@ieee.org).

Song Guo is with the Department of Computer Science and Engineering, The Hong Kong University of Science and Technology, Hong Kong, China (e-mail: songguo@cse.ust.hk).

Despite the promises of MoE, how to efficiently train MoE models is still an open challenge, mainly because their giant model sizes could easily exceed the memory limit of a single GPU. Therefore, distributed MoE training over multiple GPUs has become one of the hottest topics in AI system research [\[7\]](#page-10-6), [\[8\]](#page-10-7). Since MoE models have giant sizes and unique structures, some recent works [\[5\]](#page-10-4), [\[6\]](#page-10-5) have recognized that traditional parallelism schemes, e.g., data parallelism and model parallelism, can be hardly applied to distributed MoE training. Recently, *expert parallelism* [\[5\]](#page-10-4) has been proposed as a novel parallelism scheme dedicated to MoE. As shown in [Figure 1\(a\),](#page-1-0) experts, which are usually with large sizes, in each MoE layer are distributed across different GPUs, while nonexpert components (e.g., multi-head attention) with moderate sizes are replicated over all GPUs. Expert parallelism can maximize GPU resource utilization and thus has become the mainstream of distributed MoE training [\[9\]](#page-10-8), [\[10\]](#page-10-9), [\[11\]](#page-10-10). Several popular open-source model training frameworks, such as Microsoft's DeepSpeed-MoE [\[7\]](#page-10-6) and Tutel [\[8\]](#page-10-7), have supported expert parallelism.

1

However, the default expert parallelism, as shown in [Fig](#page-1-0)[ure 1\(a\),](#page-1-0) suffers from a high network burden because of the all-to-all intermediate data exchange among GPUs before and after experts. Many tokens in a GPU may be pushed to experts located at others by the gate network. This process is usually termed as the dispatch phase. After being processed by experts, tokens are pulled back to original GPUs to revert into sequences, in a so-called combine phase. GPUs need to transmit massive tokens through the inter-GPU network in both dispatch and combine phases, which significantly degrades the efficiency of distributed MoE training. Existing works have revealed that the communication costs become the main bottleneck of the whole system [\[12\]](#page-10-11), [\[10\]](#page-10-9), [\[11\]](#page-10-10). Some recent works, e.g., Lina [\[11\]](#page-10-10), have proposed to hide this bottleneck by overlapping token transmission and expert computation, but they cannot reduce the size of intermediate data and are still far from fundamentally solving the communication challenge.

Another line of works have proposed to copy remote experts to local GPUs, if there is too much intermediate data pushed out. An example is shown in [Figure 1\(b\).](#page-1-1) Janus [\[10\]](#page-10-9) follows this idea and designs sophisticated algorithms to decide when and how to fetch remote experts. FasterMoE [\[13\]](#page-10-12) has proposed a dynamic expert fetching scheme guided by a performance model. However, experts could be large and incur high transmission cost. Moreover, copying more experts to local GPUs

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

<span id="page-1-1"></span>Fig. 1: Comparison between existing works and our ideas. We assume that each GPU holds one expert (e.g., GPU k holds Expert k). A: attention computation; E: expert computation; D: token dispatch; C: token combine; T: expert transfer. Each rectangle represents a part of a sequence, and the width of a rectangle indicates its length. Arrows with solid and dashed lines represent intra-GPU and inter-GPU traffic, respectively, and the arrow width indicates the amount of traffic. The red hatched rectangles in (d) represent the tokens eliminated by token condensation. The block index is denoted by b.

intensifies GPU resource competition. The improved communication efficiency is traded with reduced parallelism levels of expert running. In addition, all existing works mainly focus on experts, without much study about multi-head attention, which is the most compute-intensive component of MoE [\[14\]](#page-10-13), [\[15\]](#page-10-14).

The above facts motivate us to explore whether it is possible to reduce inter-GPU traffic while maintaining a high degree of expert parallelism. This paper gives a positive response by conducting a holistic system study of jointly optimizing communication and computation. We present LUFFY, a distributed MoE training system that can significantly improve time-toaccuracy by two key techniques. First, instead of moving experts, LUFFY migrates sequences among GPUs to reduce cross-GPU token pulling in the combine phase. As shown in [Figure 1\(c\),](#page-1-2) if many tokens of a sequence are pushed to a remote GPU, this sequence can be migrated to that GPU. After migration, this sequence is reverted by pulling its tokens to the new location, followed by being fed to the multi-head attention of the next block. Such a sequence migration can hide heavy token pulling paths within GPUs, thus reducing inter-GPU traffic. Meanwhile, it avoids copying experts over GPUs, so that we can save GPU memory and maintain high degree of expert parallelism. Furthermore, sequence migration provides a new chance to optimize attention computation by gathering sequences of similar lengths, so that we can reduce padded zeros when aligning them for batch processing.

We then shift our focus from the combine phase to the dispatch phase, where the second technique, token condensation, can be applied to further reduce inter-GPU traffic. Token condensation is based on an important observation that a considerable number of tokens pushed to the same expert exhibit high similarity, which has not yet been exploited by existing works. For example, when training the MoE-TransformerXL model, we find that about 62% tokens pushed to the same expert are very similar, and sending only one of them has almost no influence on the final training accuracy. <span id="page-1-3"></span><span id="page-1-2"></span>Therefore, we propose the token condensation, as shown in [Figure 1\(d\),](#page-1-3) to identify similar tokens and then to eliminate redundant transmissions. Note that token condensation can reduce traffic not only in the dispatch phase but also in the combine phase because we further find that token similarity can be preserved after passing experts.

Although sequence migration and token condensation are promising, it is non-trivial to bring them into full play in practical MoE training due to following technical challenges. First, sequence migration needs to decide which sequences should be migrated to which GPUs. LUFFY features an algorithm to make migration decisions by jointly considering the token pulling cost and efficiency of the subsequent attention computation. Note that attention computation is affected by two factors. One is about how many sequences should be handled by each GPU, and the other is about whether these sequences have similar lengths, so that we can reduce padded zeros. Second, token condensation should have a low overhead in identifying similar tokens. A straightforward pair-wise comparison is with high computational complexity and it can hardly work in practice. We equip LUFFY with a fast algorithm to identify similar tokens by fully exploiting token features during MoE training. In addition, token condensation needs to strike a balance between the amount of condensed tokens and training convergence. If more tokens are condensed, we can have lower communication costs but may miss important differences between tokens, which would slow down or even compromise the training convergence. Therefore, LUFFY uses a dynamic condensation policy that can adjust token condensation rate according to training convergence status.

We implement LUFFY using PyTorch and evaluate its performance on a testbed consisting of 16 V100 GPUs. The experimental results show that LUFFY can achieve a speedup of up to 2.73× compared to state-of-the-art MoE training systems.

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

Fig. 2: An illustration of MoE.

| Model             | Expert=4, Batch = 8 |        |       | Expert=4, Batch = 16 |        |       | Expert=8, Batch = 8 |        |       |
|-------------------|---------------------|--------|-------|----------------------|--------|-------|---------------------|--------|-------|
|                   | S (GB)              | C (ms) | R (%) | S (GB)               | C (ms) | R (%) | S (GB)              | C (ms) | R (%) |
| MoE-TransformerXL | 3.19                | 327    | 18.1  | 6.15                 | 507    | 14.8  | 3.98                | 381    | 30.5  |
| MoE-BERT-Large    | 6.73                | 439    | 36.6  | 13.07                | 859    | 40.3  | 7.92                | 477    | 47.5  |
| MoE-GPT2          | 6.53                | 411    | 34.6  | 12.13                | 707    | 35.9  | 7.52                | 452    | 45.9  |

TABLE I: Communication bottleneck for distributed MoE training. The total number of experts is set equal to the number of GPUs. **S** means the data transfer size. **C** and **R** mean the time of all-to-all communication and its ratio in a batch of training.

#### II. BACKGROUND AND MOTIVATION

### A. MoE and Distributed MoE Training

Transformer [1] emerges as the primary architecture for many complex tasks in natural language processing (NLP), computer vision, and beyond. As Figure 2 depicts, the Transformer architecture consists of multiple blocks, each of which includes a multi-head attention layer and a Feed-Forward Network (FFN) layer. MoE structure has been widely used to scale up Transformer-based models [16], [6], [17] by replacing the FFN layer with an MoE layer, consisting of a gate network and multiple expert networks that are essentially FFNs. The instinct behind the MoE is that each expert could be trained to handle a specific kind of input. Thus, the whole model capability becomes stronger as more experts are integrated.

However, more experts significantly enlarge the MoE model size, with higher running costs. For example, Switch Transformer [6] with 256 experts per block requires more than 50GB memory to accommodate the model parameters. However, the current mainstream GPUs peak at 48GB, while many ranging from 12GB to 24GB. Thus, traditional data parallelism and pipeline parallelism could hardly be applied for efficient distributed training or inference over multiple GPUs. Recently, expert parallelism [6], [5] has been proposed as a novel hybrid parallelism dedicated to MoE by exploiting its unique features [18], [19], [20], [13]. Specifically, multiple experts are distributed across different GPUs, while other components, such as attention layers, are replicated on GPUs. The gate network selects experts for each input token and dispatches them to corresponding GPUs containing the required experts. This process needs an all-to-all communication between GPUs and is called *dispatch phase*. After the expert computation, another all-to-all operation is initiated to combine tokens into sequences for the execution of subsequent layers, which is called combine phase. In this paper, we mainly focus on optimizing the expert-parallelism-based distributed MoE training. Although there are some research works about hybrid parallelism strategies [21], [8], they are orthogonal to our work.

#### B. Motivations

**Traffic pattern of MoE training**: We first conduct experiments to study the communication cost incurred by the all-to-all token push and pull operations in the default expert parallelism<sup>1</sup>. Table I shows the data transfer size of all-to-all

<span id="page-2-1"></span><sup>1</sup>We focus on the communication cost, which is mainly caused by the all-to-all operations for tokens, excluding gradient synchronization.

operations in a training batch of MoE-TransformerXL [22], MoE-BERT-Large [2], and MoE-GPT2 [3]. We here use top-2 gating and other preliminary experiments follow the same setting. When training the MoE-BERT-Large with 4 experts per block by setting the batch size as 8, the total data transfer size reaches 6.73GB. Moreover, we can see that the amount of data transmission increases with larger batch sizes and more experts. The communication time becomes a significant portion of the total training time. When training MoE-BERT-Large with 4 experts per block and a batch size of 8, the communication time is 439ms, which is 36.6% of the total training time. When the number of experts increases to 8, this ratio grows to 47.5%. There are similar observations for other models.

In addition, we find that each sequence activates very a few experts. We randomly select some sequences and show the portion of tokens pushed to different experts in Figure 3. Statistically, more than half of sequences use no more than 3 experts when training MoE-TransformerXL and MoE-BERT-Large. MoE-GPT2 has a stronger bias in expert activation, and more than 80% sequences use only 1 or 2 experts. The biased expert activation is mainly attributed to the data distribution in each sequence, which results in a non-uniform output of the gate network. This characteristic exists throughout the training process, even at the beginning of the training with a randomly initialized gate network. A similar observation has been reported in [13].

To reduce the all-to-all communication cost, some recent works [10], [13], [9] have proposed to transfer experts instead of intermediate data between GPUs, which are effective when the expert size is smaller than volume of tokens. However, this method cannot fundamentally address the communication bottleneck, especially when facing large expert sizes. For example, recent Mixtral 8×7B [23] has 256 experts in total, and each expert is with a size of about 300MB. In addition, the execution of experts co-located on the same GPU cannot be well parallelized because of resource competition. To demonstrate this issue, we migrate different numbers of experts to a single GPU and measure the corresponding running time. As shown in Figure 4, the computation time increases as more experts are migrated to the same GPU. For example, as we increase the number of experts from 1 to 3 for MoE-BERT-Large, its expert computation time grows to  $1.88 \times$ .

**Token Similarity**: We collect token embeddings routed to the same experts and compare their cosine similarity, a common metric to evaluate embedding similarity [24], [25], [26]. We here use a normalized cosine similarity and its value ranges

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

Fig. 3: Biased expert activation for sequences under different models after 30 training iterations, where a training iteration indicates the training on a batch of data. Different colors represent hotness values, which indicate the portions of tokens routed to different experts.

<span id="page-3-1"></span>![](_page_3_Figure_3.jpeg)

Fig. 4: Batch time on one GPU with different number of experts. The batch size is set as 1.

from [0,1]. A larger value means higher similarity. Due to space limits, we selectively report the results of three blocks (i.e., block 1, block 3, and block 6) of studied models. The results are presented in Figure 5(a), which reveals a significant prevalence of similar tokens across different models. For example, about 25% of token pairs in the first block of the MoE-TransformerXL model have a similarity greater than 0.75. Similarly, in the sixth block of the MoE-BERT-Large model, about 57% of token pairs have a similarity greater than 0.55. In addition, we notice that token pairs tend to show increased similarity in deeper levels. For example, in the first block of the MoE-TransformerXL model, only 25% of token pairs have a similarity greater than 0.75, but this proportion increases to 85% in the sixth block. In the case of MoE-GPT2, the proportion of token pairs with similarity above 0.50 increases from 18% in the first block to 50% in the sixth block. This trend of higher similarity in deeper layers can be attributed to the significant reduction in the rank of

<span id="page-3-2"></span>![](_page_3_Figure_6.jpeg)

(a) Token similarity over different blocks.

<span id="page-3-3"></span>![](_page_3_Figure_8.jpeg)

(b) Token similarity change after the expert execution.

Fig. 5: Token similarity and the change after the expert execution after 30 training iterations. All results are shown over block 1 (**left**), block 3 (**middle**), and block 6 (**right**).

the embedding matrix [27]. The above observations suggest a great chance of reducing communication costs by eliminating the transmission of similar tokens.

We further check whether token similarity could be preserved after they pass through the same experts. If similar tokens would be quite different after passing experts, this idea does not work because once we select to transmit only one of similar tokens, their differences are dismissed in the subsequent expert computation. To verify similarity preservation, we select some token pairs whose similarity exceeds a certain threshold, which is set to 0.75 for MoE-TransformerXL, 0.55 for MoE-BERT-Large, and 0.50 for MoE-GPT2, and show their similarity changes in Figure 5(b). We can see that the similarity of token pairs has changed only slightly after passing through experts. For example, in the first block of the MoE TransformerXL, about 95% of token pairs have a similarity change of less than 0.2. Similarly, in the MoE-BERT-Large model, about 36% of token pairs in the third block have a similarity change below 0.10, and this proportion increases to 98% in the sixth block.

## III. SYSTEM OVERVIEW

LUFFY is designed with the goal of improving time-to-accuracy of distributed MoE training across multiple GPUs. To achieve this goal, LUFFY follows several important design principles. First, inter-GPU traffic should be minimized. The all-to-all token push and pull operations have been recognized as the main system bottleneck by existing works. The main solution adopted by existing works is to move experts or to exploit network resources with complex scheduling algorithms. LUFFY explores the possibility of fundamentally solving this challenge by reducing number of tokens sent over the network. Second, the computation part, involving not only experts but also attention layers, of MoE training should also be efficient. Many existing works have excessive concerns about the all-to-all token communication, but share

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

Fig. 6: Architecture overview of LUFFY.

little consideration of expert or attention computation. Third, LUFFY should not compromise the training convergence, and thus preserve the quality of the final MoE model. LUFFY can allow a certain level of computational approximation during training for acceleration, but approximation errors should be constrained.

An overview of LUFFY's design is shown in [Figure 6.](#page-4-0) LUFFY is based on expert parallelism that each GPU has one or a few experts and a full copy of attention layers. An important design choice that makes LUFFY different from existing work is that it does not allow expert movement during training, so that it can parallelize expert running at the maximum level. LUFFY's superiority stems from two novel designs: *sequence migration* and *token condensation*. Sequence migration works in the combine phase, to strategically decide locations where sequences should be re-constructed and then be fed to the subsequent attention layers. An algorithm is designed to make migration decisions by jointly considering token pulling cost and running efficiency of attention layers. The second key design, token condensation, can be applied for both dispatch and combine phases. It eliminates the transmission of similar tokens to reduce inter-GPU traffic. Moreover, since only one of similar tokens reaches and goes through the corresponding expert, we can save a lot of expert computation. A fast heuristic algorithm is proposed to quickly identify similar tokens. In addition, the approximation error during expert computation can be constrained to guarantee training convergence.

The system workflow is as follows. As shown in [Figure 6,](#page-4-0) ① LUFFY profiles training information (e.g., batch sizes and sequence lengths) and hardware information (e.g., GPU speed) at the start of each training iteration. Here an iteration is defined as the process of training a batch of data. After profiling, GPUs load training data and proceed to train the MoE model. ② GPUs run attention computations locally, and then send the output to the token condensation module that identifies similar tokens and reduces redundant transmissions. ③ Expert computation is launched after receiving tokens. Meanwhile, the sequence migration module decides where tokens should be pulled back for re-construction. After that, LUFFY follows these decisions to collect tokens in the combine phase. ④ Finally, GPUs compute gradients based on the loss function and synchronize parameters of attention layers and

```
Algorithm 1: Sequence Migration Algorithm
```

```
Input: The set of sequences N, the set of GPUs M;
1 For sequence i, we estimate traffic fi,j supposing to
   migrate i to GPU j;
```

<sup>2</sup> Put top-q GPUs with minimum traffic into a candidate set H<sup>i</sup> ;

```
3 for each sequence i do
4 for each GPU j ∈ Hi do
5 si,j = Tatt(Bj←i
                         , Lj←i) − Tatt(Bj , Lj );
6 Migrate sequence i to the GPU j
                                     ∗ with maximum
      si,j if it has sufficient capacity.
7 return j
          ∗
            for each sequence;
```

experts for the next iteration.

## IV. SEQUENCE MIGRATION

To minimize token transmissions in the combine phase, an intuitive idea is to migrate each sequence to the GPU accommodating the most of its tokens, so that only a few other ones need to be pushed out and pulled back over the network. However, this approach performs poorly in practice because some GPUs would be assigned too many sequences, leading to a serious workload imbalance of following attention computation. In addition, if sequences of different lengths are batched together, short sequences must be padded with zeros to align with other long ones, leading to GPU memory waste and additional computation [\[28\]](#page-10-27). In LUFFY, we design a sequence migration algorithm that holistically considers the communication cost of the combine phase and the computational efficiency of subsequent blocks. We first present the algorithm design in [§ IV-A](#page-4-1) and then give details of a cost model that plays an important role in the algorithm in [§ IV-B.](#page-5-0)

# <span id="page-4-1"></span>*A. Migration Algorithm Design*

The sequence migration problem could be difficult because of dual optimization objectives (i.e., both communication and computation cost) and a large optimization space (i.e., there could be lots of possible combinations of sequences and GPUs). To tackle this, we propose a heuristic approach consisting of two steps: one focuses on finding out some candidate GPUs for each sequence with less traffic, and the other step then gathers sequences with similar lengths by migrating them to one of candidate GPUs.

The pseudo codes of the proposed algorithm are shown in [algorithm 1.](#page-4-2) For each sequence i, we estimate its token pulling traffic if it is migrated to different GPUs. We choose the top-q GPUs with minimum traffic as candidate locations, and include them into set H<sup>i</sup> . A large q can provide more flexibility when we gather sequences with similar lengths in the next step, but may come with higher traffic cost. Setting q = 1 goes to the other extreme that we put all efforts in minimizing traffic, without considering attention computation.

After getting all H<sup>i</sup> , we continue to find a GPU for each sequence from its candidate set. Similar sequences are expected to stay at the same GPU to reduce padded zeros. Note that we do not directly count the number of padded zeros as the metric for decision making, as it does not accurately reflect the impact on computational cost. Suppose there is a sequence a of length 11 that needs to be migrated. There are two GPUs as candidates, GPU1 with a sequence of length 1, and GPU2 with two sequences of length 6. Migrating sequence a to either GPU results in 10 padded zeros. However, migrating sequence a to GPU2 is a better choice with lower cost because of less attention computation among tokens.

Therefore, we propose a cost model to estimate the attention computation time. Given B sequences and the maximum sequence length L, the cost model can be described as a function  $T_{att}(B,L)$ , whose details are presented in § IV-B. As shown in lines 3-6, for sequence i, we select a GPU with the minimum cost growth from its candidate set. Meanwhile, we also consider the capacity constraints of GPUs. A GPU can accommodate more short sequences but less long ones.

#### <span id="page-5-0"></span>B. Cost Model

The performance of the migration algorithm depends on the accurate estimation of attention computation cost. As the attention operation is typically compute-bound [9], we can estimate the cost of each attention layer as follows.

$$T_{att}(B, L) = \left\{ \underbrace{\frac{3BLd^2}{P}}_{\text{Linear projection}} + \underbrace{\frac{2BL^2d}{P}}_{\text{Dot-product}} \right\}, \tag{1}$$

where d is the feature dimension and P denotes GPU speed. The rationale of this model is explained as follows. Each attention layer consists of two key operations. First, the input tokens are linearly projected into corresponding queries, keys, and values, denoted by Q, K, and V, respectively. Second, a scaled dot product operation is performed to compute the attention scores and weighted outputs as  $Attention = softmax(\frac{QK^T}{\sqrt{d}})V$ .

According to the above workflow, the total operations for each linear projection is  $BLd^2$ . Note that there are three linear projections to generate Q, K, and V, respectively. The scaled dot product operation for each sequence has two parts, where one is the dot product of Q and K, resulting in a total operations of  $BL^2d$ . Then, the softmax is applied, followed by the multiplication of V and attention scores, with  $BL^2d$  operations. Thus, the number of computational operations for the scaled dot product operation is  $2BL^2d$ . We ignore the cost of the softmax operation since it is significantly lower than that of matrix multiplication, which has been also verified by [29]. The speed P is profiled by running an attention layer several times with varying B and L.

#### V. TOKEN CONDENSATION

Token condensation contains two main tasks, measuring token similarity and deciding "how similar" are tokens to be condensed. An intuitive idea is to compute the pairwise similarity (e.g., cosine similarity) between each pair of tokens and use a predefined similarity threshold to decide which

<span id="page-5-2"></span>![](_page_5_Figure_10.jpeg)

<span id="page-5-3"></span>![](_page_5_Figure_11.jpeg)

(a) Similarity change for token pairs (b) Similarity change for token pairs in block b and have  $s_b>0.8$ . in block b and have  $s_b<0.2$ .

Fig. 7: Similarity change across consecutive blocks for the MoE-TransformerXL model.

tokens should be condensed. However, this approach can hardly work in practice for the following reasons.

First, computing pairwise similarity between every two tokens has a high overhead, because of the massive number of tokens involved in MoE training and their high-dimensional embeddings (e.g., each token of MoE-TransformerXL has 1024 dimensions). The high overhead would counteract the benefit of token condensation in reducing communication cost. To address this challenge, we propose a fast similarity measurement algorithm to efficiently compute the similarity between tokens by strategically skipping unnecessary similarity computations in § V-A.

Second, condensing more tokens means less network traffic, but would increase the risk of compromising convergence due to computational errors. A trade-off between communication efficiency and training convergence should be considered to improve the time-to-accuracy. Moreover, we find that tokens become more similar as the training proceeds, which implies that we can condense more tokens in later training rounds. Motivated by the above observations, we design an adaptive condensation strategy to dynamically set the similarity threshold in § V-B.

#### <span id="page-5-1"></span>A. Fast Similarity Measurement

Our design goal here is to make the measurement process fast by reducing the pairwise similarity computation between tokens. To better present our idea, we model tokens and their similarity relationship as a fully connected graph. By default, all edge weights, which represent similarity levels, in the graph need to be measured, leading to high computational overhead. However, we find that some tokens are obviously similar or dissimilar according to expert activation and historical knowledge. We can skip their similarity calculation by directly assigning the weights of corresponding edges as 0 or 1. Specifically, our fast similarity measurement algorithm works as follows.

1) Measuring similarity by expert activation. We first find out edges whose associated tokens are pushed to different experts, and set their weights as 0. The design rationale is as follows. Experts of an MoE model are often designed to process different kinds of input. Thus, it is expected that similar tokens are routed to and processed by the same expert. In other words, tokens assigned to different experts are highly unlikely to be similar. Note that tokens going to the same expert are

not definitely similar, and we need to continue to check them in the next step.

- 2) Measuring similarity by historical similarity. We then identify obviously dissimilar and similar tokens according to historical information. This design is based on the observation that token pairs that are extremely similar or dissimilar in the previous block always tend to keep this pattern. As shown in Figure 7(a), we randomly select some token pairs whose similarity values are greater than 0.8 and check their value changes in several consecutive blocks. We can see that about 90% of token pairs still keep their similarity of greater than 0.8. We also check dissimilar token pairs whose similarity values are less than 0.2 and show results in Figure 7(b). Based on this observation, in the b-th block, we identify tokens as similar ones (whose link weights are set to 1) if  $s_{b-1} > S_1$ , or dissimilar ones (whose link weights are 0) if  $s_{b-1} < S_2$ , where  $S_1$  and  $S_2$  are two adjustable system parameters.
- 3) Calculating similarity of rest token pairs. Up to now, similarity of most of token pairs has been decided. The rest ones are highly uncertain and we measure them by conducting real cosine similarity calculation. Note that similarity calculations in this step could be quick because they can be easily parallelized.

## <span id="page-6-0"></span>B. Adaptive Token Condensation

After token similarity measurement, we need to decide which tokens should be condensed, i.e., not being transmitted in the dispatch phase. Specifically, we delete the edges in the graph whose weights are below a given threshold, which generates a sparse graph composed of multiple subgraphs. For each subgraph, we keep the token with the highest degree for transmission and condense its neighboring tokens. We repeat this process until all tokens are condensed in subgraphs.

To achieve a trade-off between communication efficiency and training convergence, we introduce an adaptive token condensation strategy, which adaptively generates a condensation threshold to ensure the training convergence. The basic idea is as follows. The early stage of MoE training is often accompanied by unstable training convergence. Thus, we need a high threshold to prevent most tokens from being condensed to maintain convergence. As the training progresses, the model tends to converge, and we should lower the threshold to condense more tokens so that the data transmission can be reduced as much as possible. Specifically, we set the threshold  $h_t$  for training iteration t according to the loss value in the previous iteration:

$$h_t = \frac{1}{1 + \exp(l_{norm})}, \quad l_{norm} = \frac{l_{ini} - l_{t-1}}{l_{ini}},$$
 (2)

where  $l_{ini}$  and  $l_{t-1}$  are the loss values in the first and previous training iterations, respectively. The normalized loss decrease in the training iteration t-1 is denoted by  $l_{norm}$ . The early training stages have a small normalized loss decrease  $l_{norm}$ , and thus we obtain a large condensation threshold to keep tokens as much as possible. As training progresses,  $l_{norm}$  becomes larger, indicating that the training tends to be stable.

<span id="page-6-1"></span>

| Model name     | Experts | Layers | $d_{model}$ | $d_{hidden}$ | len  | Size         |  |
|----------------|---------|--------|-------------|--------------|------|--------------|--|
| MoE-           | 2, 4    | 18     | 1024        | 4096         | 250  | 0.44B, 0.74B |  |
| Transformer-XL | 8, 16   | 10     | 1024        | 4090         | 230  | 1.34B, 2.55B |  |
| MoE-           | 2, 4    | 24     | 768         | 3072         | 512  | 0.54B, 0.94B |  |
| BERT-Large     | 8, 16   | 24     | 708         | 3072         | 312  | 1.74B, 3.36B |  |
| MoE-GPT2       | 2, 4    | 12     | 768         | 3072         | 1024 | 0.18B, 0.29B |  |
|                | 8, 16   | 12     |             |              |      | 0.52B, 0.97B |  |

TABLE II: Specifications of models for evaluation.  $d_{model}$  refers to the dimension of token embedding while  $d_{hidden}$  refers to the hidden dimension of FFN layer, i.e., an expert. B is the abbreviation of billion.

Thus, we lower the threshold  $h_t$  to eliminate more tokens. We use an exponential function here to make the threshold unbiased since the loss decrease becomes trivial in the stable stages of the training.

#### VI. IMPLEMENTATION

LUFFY is implemented using PyTorch [30] by adding about 4.5K lines of codes. The developers can easily invoke LUFFY as a plug-in-play plugin.

Sequence Migration Controller. In LUFFY, a machine is selected to run the sequence migration module and it is called controller. It collects all the information needed for algorithm input and makes migration decisions. The required information used in the migration algorithm, e.g., which GPU each token is dispatched to for expert running, is distributed across GPUs and will be gathered by the controller. This operation can be parallelized with expert running. To guide the sequence migration, the controller maintains three hash tables: token\_to\_sequence, token\_to\_gpu, and sequence\_to\_gpu, to record the information of execution location for tokens and sequences. For instance, sequence\_to\_gpu maintains the information about which GPU each token should be sent to for sequence combining, and it will be updated by the controller according to the migration algorithm outputs. Both tables token to sequence and sequence to gpu will guide GPUs on how to exchange tokens for sequence combining with torch.distributed.rpc APIs.

**Token Condensation Scheduler.** Each GPU maintains a CUDA stream as the token condensation scheduler, which calculates token similarities and conducts token condensation. The scheduler creates a token graph with DGL [31] APIs. Each node represents a token, endowed with two features: the corresponding expert index generated by the gate and the token embedding. Initially, we generate edge features based on the historical similarity between the connected tokens. We define an edge-wise function edge\_sim\_calculation to calculate the similarity between tokens efficiently. To achieve token condensation, the scheduler maintains a hash table  $token\_to\_token$  to indicate how tokens are condensed. For example,  $token\_to\_token(i) = j$  represents that token i is condensed and we need to use the expert output of token j to replace it.

## VII. EVALUATION

## A. Setup

**Testbed.** We evaluate LUFFY on a testbed of 16 NVIDIA V100 GPUs with 16GB memory and PCIe connections, aligned with

the settings in [\[13\]](#page-10-12). We use Ubuntu 20.04 with Linux kernel version 5.15, NVIDIA driver 525.85, CUDA 11.7, and cuDNN 8.6.0.

Models. We consider three popular MoE models, including (1) MoE-TransformerXL [\[22\]](#page-10-21), a 18-block decoder model; (2) MoE-BERT-Large [\[2\]](#page-10-1), a 24-block encoder model; and (3) MoE-GPT2 [\[3\]](#page-10-2), a 12-block decoder model. The number of experts in each MoE layer varies from 2, 4, 8, to 16. All models are equipped with top-2 gate networks. We set the batch size as 64 for all models. We set the number of experts equal to the number of GPUs, similar to the common practice [\[6\]](#page-10-5), [\[11\]](#page-10-10). More details of the models used in our experiments are shown in Table [II.](#page-6-1)

Baselines. We compare LUFFY with the following baselines. (1) Vanilla: the MoE implementation with expert parallelism, adopted by DeepSpeed [\[7\]](#page-10-6); (2) EXT (Expert Transfer): an MoE training paradigm that optimizes the communication cost by transferring activated experts across GPUs, instead of dispatching and combining tokens, which is adopted in Janus [\[10\]](#page-10-9); and (3) HYT (Hybrid of Token and Expert Transfer): it improves the end-to-end MoE training efficiency by strategically transferring popular experts to all GPUs, which is adopted in FasterMoE [\[13\]](#page-10-12).

## *B. End-to-End Performance*

[Figure 8](#page-8-0) shows the overall speedup by normalizing the average iteration time over that of Vanilla, where an iteration indicates the training on a batch of data. All methods use the same configurations when training on the same model, such as sequence length and batch size. LUFFY outperforms other baselines under all models and its superiority becomes clearer when there are more experts. For instance, with a number of experts ranging from 4 to 16, LUFFY provides a speedup from 1.51× to 2.73× over the vanilla on the MoE-TransformerXL model. This is because more experts means more all-to-all traffic, but LUFFY has stronger capability to reduce the total number of data transfers through token condensation and sequence migration, resulting in a higher speedup.

EXT and HYT also show obvious improvement over the vanilla solution. However, it's important to note that they may introduce significant competition for GPU resources during model computation, which can compromise the benefits of expert transfer by reducing parallelism. As a result, their performance improvement may be limited. LUFFY optimizes training efficiency by jointly considering communication costs and computation efficiency, resulting in higher performance gains. Compared to EXT and HYT, LUFFY achieves up to 1.65× and 1.46× speedup, respectively. More details about the results are as follows.

MoE-TransformerXL. The MoE-TransformerXL has larger experts, which can result in higher communication cost when transferring them. EXT copies remote experts to local GPUs once they are activated, which may not always be the best choice. Thus, it has smaller speedup under MoE-TransformerXL than other models. In contrast, HYT and LUFFY well consider expert sizes and have achieved higher performance.

MoE-BERT-Large. The MoE-BERT-Large model has more experts because of its large number of MoE blocks. When transferring these experts, there may be competition for GPU resources, which can result in longer computation times for expert running. Furthermore, the negative impact of expert transfer is amplified with more MoE blocks. Our LUFFY model achieves approximately 1.61× and 1.80× speedup compared to EXT and HYT, respectively.

MoE-GPT2. The MoE-GPT2 model has a small number of experts but a large number of tokens in each MoE block due to the long sequence length. Therefore, both EXT and HYT provide a high speedup with expert transfer. However, our LUFFY still achieves respective 1.33× and 1.53× speedups compared to EXT and HYT.

Moreover, the speedup provided by LUFFY is from not only communication reduction but also computation savings. This is because token condensation decreases the number of tokens processed by experts, thereby reducing computation costs. We show more results and analysis of communication and computation improvements in [§ VII-C.](#page-7-0)

## <span id="page-7-0"></span>*C. Performance Breakdown*

We then break down the batch training time into three parts (computation time for attention and expert running, and communication time for token and expert transfer) and show the results in [Table III.](#page-8-1) We report the time cost in milliseconds for each part as well as the speedup, compared to Vanilla. EXT and HYT can significantly reduce the communication cost by about 3.84× and 4.45× compared to Vanilla. However, they sacrifice the parallelism level of expert running by introducing intensive GPU resource contention. The computation cost is increased by about 1.81× and 1.62× with EXT and HYT, respectively. The computation costs of EXT and HYT increase with more experts. For the MoE-GPT2 model with 16 experts, the computation costs of EXT and HYT are 3.57× and 3.13× higher than that of Vanilla. In contrast, our LUFFY optimizes communication efficiency without sacrificing the parallelism level of expert running. In addition, the sequence migration and token condensation modules introduced by LUFFY can also optimize the efficiency in both expert and attention running computation. As a result, LUFFY respectively achieves average speedups of 1.35× and 2.66× on communication and computation, compared to Vanilla.

## *D. Ablation Study*

We conduct ablation experiments to study the performance improvement of different components in LUFFY. We use Vanilla as a baseline and report the average speedup of different components in [Figure 9.](#page-8-2) We see that the token condensation and sequence migration modules provide different performance gains for different MoE models. Specifically, for the MoE-TransformerXL model, we find that token condensation provides a higher performance improvement compared to sequence migration. This is because MoE-TransformerXL model has more similar tokens (as demonstrated in [Figure 5\(a\)\)](#page-3-2) and thus token condensation can condense more tokens, leading to reduce inter-GPU traffic. When only the token condensation

<span id="page-8-0"></span>![](_page_8_Figure_1.jpeg)

Fig. 8: Batch training time speedup of different MoE training systems.

<span id="page-8-1"></span>

| Model              | Method  | #Experts=2          |                   | #Experts=4  |                   | #Experts=8         |                    | #Experts=16        |                    |
|--------------------|---------|---------------------|-------------------|-------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| Model              |         | Computation         | Communication     | Computation | Communication     | Computation        | Communication      | Computation        | Communication      |
|                    | Vanilla | 2169                | 843               | 2102        | 1522              | 1923               | 2548               | 1533               | 4599               |
| TransformerXL HY   | EXT     | 2403(0.92×)         | $209(4.03\times)$ | 2714(0.78×) | $370(4.11\times)$ | 3054(0.63×)        | $625(4.07\times)$  | 3699(0.41×)        | $1233(3.73\times)$ |
|                    | HYT     | $2265(0.96\times)$  | $197(4.28\times)$ | 2387(0.84×) | $357(4.26\times)$ | $2629(0.72\times)$ | $539(4.73\times)$  | $3204(0.48\times)$ | $1068(4.31\times)$ |
|                    | LUFFY   | 1521(1.43×)         | $480(1.76\times)$ | 1389(1.51×) | $851(1.79\times)$ | 1225(1.57×)        | $1043(2.35\times)$ | 1012(1.52×)        | $1238(3.72\times)$ |
| MoE-<br>BERT-Large | Vanilla | 973                 | 899               | 953         | 2122              | 918                | 3072               | 756                | 4284               |
|                    | EXT     | $1258(0.77\times)$  | $314(2.87\times)$ | 1989(0.48×) | $561(3.60\times)$ | 2011(0.45×)        | $1181(2.60\times)$ | $2112(0.36\times)$ | $1728(2.48\times)$ |
|                    | HYT     | $1123(0.87\times)$  | $281(3.21\times)$ | 1794(0.53×) | $506(3.99\times)$ | 1843(0.49×)        | $1083(2.84\times)$ | 1914(0.39×)        | $1386(3.09\times)$ |
|                    | LUFFY   | 784(1.24×)          | $404(2.23\times)$ | 728(1.31×)  | $672(3.01\times)$ | 638(1.44×)         | $1042(2.95\times)$ | 525(1.44×)         | $1225(3.49\times)$ |
| MoE-GPT2           | Vanilla | 955                 | 881               | 847         | 1573              | 774                | 2592               | 676                | 3834               |
|                    | EXT     | $1399(0.68 \times)$ | $209(4.22\times)$ | 1706(0.49×) | $374(4.21\times)$ | $2048(0.38\times)$ | $544(4.77\times)$  | $2402(0.28\times)$ | $718(5.34\times)$  |
|                    | HYT     | $1278(0.75\times)$  | $174(5.06\times)$ | 1509(0.56×) | $331(4.75\times)$ | 1741(0.45×)        | $435(5.96\times)$  | $2095(0.32\times)$ | 557(6.88×)         |
|                    | LUFFY   | $752(1.27\times)$   | $292(3.02\times)$ | 724(1.17×)  | $780(2.02\times)$ | 669(1.16×)         | $963(2.69\times)$  | 571(1.18×)         | 1330(2.88×)        |

TABLE III: Performance breakdown. We use Vanilla as the baseline. Blue values indicate efficiency improvements while red values indicate efficiency degradation.

<span id="page-8-2"></span>![](_page_8_Figure_5.jpeg)

Fig. 9: Performance improvements of the optimizations separately. We use Vanilla as the baseline and show speedups of different optimizations in LUFFY.

<span id="page-8-3"></span>

| Model name             | Dataset      | Metric    | Vanilla | Luffy $(h = 0.3)$ | Luffy $(h = 0.8)$ | LUFFY |
|------------------------|--------------|-----------|---------|-------------------|-------------------|-------|
| MoE-<br>Transformer-XL | WikiText-103 | PPL ↓     | 25.13   | 31.52             | 25.79             | 25.28 |
| MoE-<br>BERT-Large     | SQuAD        | F1 ↑      | 90.82   | 85.41             | 88.29             | 89.17 |
| MoE-GPT2               | SAMSum       | ROUGE-1 ↑ | 45.56   | 38.14             | 43.86             | 43.57 |

TABLE IV: The impact of token condensation on test accuracy.

is enabled, LUFFY provides about  $1.74\times$  speedup over the baseline. In contrast, in the MoE-GPT2 model, the tokens have less similarity and the benefit of token condensation is less than in the MoE-TransformerXL model, which brings only  $1.38\times$  speedup. On the other hand, MoE-GPT2 shows stronger biased expert activation and thus we have a large optimization space that can be exploited by the sequence migration, with about a  $1.72\times$  speedup. In the MoE-BERT-Large model, both token condensation and sequence migration provide high performance gains.

# E. Convergence Evaluation

We study the impact of token condensation on model quality by evaluating three models across different datasets and metrics. Specifically, the MoE-TransformerXL model is evaluated on the WikiText-103 dataset [32] using the perplexity (PPL) metric, where lower PPL values indicate better performance. For MoE-BERT-Large and MoE-GPT2, we evaluate them on the SQuAD [33] and SAMSum [34] datasets, using F1 and ROUGE-1 metrics, respectively. The larger F1 and ROUGE-1 values indicate better performance. The results are shown in Table IV. When a static threshold of 0.3 is applied, the test accuracy experiences a significant drop. For example, the F1 score of the MoE-BERT-Large model decreases from 90.82 to 85.41 under a threshold of 0.3. In contrast, our proposed LUFFY model, employing an adaptive condensation strategy, preserves the model's performance while delivering a significant training speedup.

## F. Sensitivity Analysis

We study the sensitivity of system parameters used in the sequence migration algorithm (§ IV-A) and the fast similarity measurement (§ V-A). We use the MoE-TransformerXL model to conduct the evaluation.

**Parameters of migration algorithm.** In the first step of sequence migration algorithm, top-q GPUs with minimum traffic are selected. We change the value of q and show the

<span id="page-9-0"></span>![](_page_9_Figure_1.jpeg)

![](_page_9_Figure_2.jpeg)

![](_page_9_Figure_3.jpeg)

![](_page_9_Figure_4.jpeg)

(a) Impact of different candidate

<span id="page-9-1"></span>(b) Accuracy of the cost model.

<span id="page-9-2"></span>(c) Impact of the fast similarity mea- (d) Training convergence with differsurement.

<span id="page-9-3"></span>ent configurations of fast similarity measurement.

Fig. 10: Sensitivity analysis on migration algorithm and fast similarity measurement.

corresponding traffic and computation time in Figure 10(a). We can see that more candidate GPUs can reduce the attention computation cost since each sequence has more choices to stay with others of similar lengths. In contrast, a small candidate size means we mainly focus on traffic optimization, and the cost of token transfer is minimized.

We also evaluate the effectiveness of the cost model of attention computation (§ IV-B). We collect the real costs of attention computation under different data inputs, e.g., number of sequences and sequence lengths. We compare the estimated cost with the real cost and report the results in Figure 10(b). It can be observed that our performance model introduces only a trivial error in the estimation of computation cost, with an average error of about 5% across all models.

Parameters of fast similarity measurement. We study the impact of the fast similarity measurement by setting different configurations of  $S_1$  and  $S_2$ . First, we study the impact of these parameters on the cost of similarity measurement. As shown in Figure 10(c), we can see that the measurement cost can be significantly reduced when  $S_1$  and  $S_2$  become close, because the similarity values of less pairs need to be re-calculated.

We then study the impact of  $S_1$  and  $S_2$  on the effectiveness of the convergence. The training loss over time is shown in Figure 10(d). When we increase  $S_2$ , more token pairs are directly regarded as dissimilar and assigned with a similarity of 0. In other words, fewer tokens are condensed in the dispatch and combine phases, and the training time is prolonged. In contrast, decreasing  $S_1$  makes more token pairs be assigned with a similarity of 1, indicating that more tokens can be condensed and the total training time is reduced. However, some token pairs may be wrongly estimated as similar, which brings a negative impact to the training convergence.

## VIII. RELATED WORK

MoE Models. Existing works show that model quality is strongly associated with the number of model parameters [2], [3], [35]. Recently, MoE has been widely applied as a promising solution to increase the model size and improve the model quality [36], [37], [38], [39], [6]. PaLM [40] and GLaM [41], proposed by Google, achieve surprising results in various language tasks, such as language modeling and machine translation. Recently, Mixtral 8×7B [23] has been released by Mistral AI, which achieve near state-of-the-art performance on various tasks. The success of this model inspires severl follow-up works, such as LLaMA-MoE [42], OpenMoE [43], and DeepSeekMoE [44].

Distributed MoE Training. The MoE models have giant model sizes and are typically trained with multiple GPUs, using an expert parallelism [45], [20], [8]. Existing works introduce a series of techniques to optimize the efficiency of distributed MoE training. BASE layers [46] implements expert parallelism based on FairSeq [47]. DeepSpeed-MoE [7] introduces a hierarchical all-to-all algorithm to reduce communication costs. Tutel [8] introduces the switchable parallelism and dynamic pipeline to handle unbalanced workloads of MoE. Followed by this work, PipeMoE [48] and MPipeMoE [49] study adaptive technologies to find optimal pipeline settings to improve the efficiency of pipeline parallelism for MoE training. SE-MoE [50] also adopts a hierarchical all-to-all algorithm to improve communication efficiency. Alpa [51] develops the automated parallelism for MoE models, considering both inter-operator and intra-operator parallelism. Smart-MoE [21] studies automated parallelism and decomposes the search space into static pools for efficient hybrid parallelism searching. Lina [11] systematically analyzes all-to-all overhead and designs a novel communication scheduling scheme to improve all-to-all efficiency. ScheMoE [52] introduces a framework to schedule communication and computation tasks in MoE training. However, these existing works cannot reduce the data transmission size for token push and pull operations, which is the main bottleneck for distributed MoE training. LUFFY introduces two novel techniques to significantly reduce the total data transmission, improving the MoE training efficiency.

Janus [10] adopts a data-centric paradigm to reduce communication costs by transferring experts, which typically have smaller sizes than tokens. FasterMoE [13] introduces a dynamic shadowing approach, which only transfers popular experts instead of tokens among GPUs, to reduce communication costs and achieve workload balance. Although Janus and FasterMoE can effectively reduce data transmission, they introduce intensive GPU resource competition and reduce the parallelism levels of expert running. In contrast, LUFFY reduces the data transmission by migrating sequences, instead of transferring experts, which always parallelizes expert running at the maximum level.

# IX. CONCLUSION

We present LUFFY, an efficient distributed MoE training system. LUFFY jointly optimizes the communication and computation efficiencies for MoE training via two novel designs.

First, LUFFY migrates sequences among GPUs to reduce the total data transmissions in the token combine phase. Second, we observe that there is a considerable number of tokens pushed to the same expert are similar, and we propose a token condensation technique to condense similar tokens in the dispatch phase, further reducing inter-GPU traffic. We implement LUFFY and perform extensive evaluation to show that LUFFY can significantly improve MoE training efficiency.

## REFERENCES

- <span id="page-10-0"></span>[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Advances in neural information processing systems*, vol. 30, 2017.
- <span id="page-10-1"></span>[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," *arXiv preprint arXiv:1810.04805*, 2018.
- <span id="page-10-2"></span>[3] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever *et al.*, "Language models are unsupervised multitask learners," *OpenAI blog*, vol. 1, no. 8, p. 9, 2019.
- <span id="page-10-3"></span>[4] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, "Scaling laws for neural language models," *arXiv preprint arXiv:2001.08361*, 2020.
- <span id="page-10-4"></span>[5] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," *arXiv preprint arXiv:2006.16668*, 2020.
- <span id="page-10-5"></span>[6] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *The Journal of Machine Learning Research*, vol. 23, no. 1, pp. 5232–5270, 2022.
- <span id="page-10-6"></span>[7] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "Deepspeed-moe: Advancing mixture-ofexperts inference and training to power next-generation ai scale," in *International Conference on Machine Learning*. PMLR, 2022, pp. 18 332–18 346.
- <span id="page-10-7"></span>[8] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram *et al.*, "Tutel: Adaptive mixture-of-experts at scale," *Proceedings of Machine Learning and Systems*, vol. 5, 2023.
- <span id="page-10-8"></span>[9] X. Nie, X. Miao, Z. Wang, Z. Yang, J. Xue, L. Ma, G. Cao, and B. Cui, "Flexmoe: Scaling large-scale sparse pre-trained model training via dynamic device placement," *Proceedings of the ACM on Management of Data*, vol. 1, no. 1, pp. 1–19, 2023.
- <span id="page-10-9"></span>[10] J. Liu, J. H. Wang, and Y. Jiang, "Janus: A unified distributed training framework for sparse mixture-of-experts models," in *Proceedings of the ACM SIGCOMM 2023 Conference*, 2023, pp. 486–498.
- <span id="page-10-10"></span>[11] J. Li, Y. Jiang, Y. Zhu, C. Wang, and H. Xu, "Accelerating distributed {MoE} training and inference with lina," in *2023 USENIX Annual Technical Conference (USENIX ATC 23)*, 2023, pp. 945–959.
- <span id="page-10-11"></span>[12] R. Liu, Y. J. Kim, A. Muzio, and H. Hassan, "Gating dropout: Communication-efficient regularization for sparsely activated transformers," in *International Conference on Machine Learning*. PMLR, 2022, pp. 13 782–13 792.
- <span id="page-10-12"></span>[13] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li, "Fastermoe: modeling and optimizing training of large-scale dynamic pretrained models," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2022, pp. 120–134.
- <span id="page-10-13"></span>[14] Z. Qu, L. Liu, F. Tu, Z. Chen, Y. Ding, and Y. Xie, "Dota: detect and omit weak attentions for scalable transformer acceleration," in *Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 2022, pp. 14–26.
- <span id="page-10-14"></span>[15] H. You, Z. Sun, H. Shi, Z. Yu, Y. Zhao, Y. Zhang, C. Li, B. Li, and Y. Lin, "Vitcod: Vision transformer acceleration via dedicated algorithm and accelerator co-design," in *2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*. IEEE, 2023, pp. 273–286.
- <span id="page-10-15"></span>[16] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, "Scaling vision with sparse mixture of experts," *Advances in Neural Information Processing Systems*, vol. 34, pp. 8583–8595, 2021.
- <span id="page-10-16"></span>[17] B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus, "Designing effective sparse expert models," *arXiv preprint arXiv:2202.08906*, vol. 2, 2022.

- <span id="page-10-17"></span>[18] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li, E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley *et al.*, "Deepspeed-inference: enabling efficient inference of transformer models at unprecedented scale," in *SC22: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2022, pp. 1–15.
- <span id="page-10-18"></span>[19] Z. Ma, J. He, J. Qiu, H. Cao, Y. Wang, Z. Sun, L. Zheng, H. Wang, S. Tang, T. Zheng *et al.*, "Bagualu: targeting brain scale pretrained models with over 37 million cores," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2022, pp. 192–204.
- <span id="page-10-19"></span>[20] J. He, J. Qiu, A. Zeng, Z. Yang, J. Zhai, and J. Tang, "Fastmoe: A fast mixture-of-expert training system," *arXiv preprint arXiv:2103.13262*, 2021.
- <span id="page-10-20"></span>[21] M. Zhai, J. He, Z. Ma, Z. Zong, R. Zhang, and J. Zhai, "{SmartMoE}: Efficiently training {Sparsely-Activated} models through combining offline and online parallelization," in *2023 USENIX Annual Technical Conference (USENIX ATC 23)*, 2023, pp. 961–975.
- <span id="page-10-21"></span>[22] Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdinov, "Transformer-xl: Attentive language models beyond a fixed-length context," *arXiv preprint arXiv:1901.02860*, 2019.
- <span id="page-10-22"></span>[23] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand *et al.*, "Mixtral of experts," *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-10-23"></span>[24] M. Antoniak and D. Mimno, "Evaluating the stability of embeddingbased word similarities," *Transactions of the Association for Computational Linguistics*, vol. 6, pp. 107–119, 2018.
- <span id="page-10-24"></span>[25] T. Thongtan and T. Phienthrakul, "Sentiment classification using document embeddings trained with cosine similarity," in *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*, 2019, pp. 407–414.
- <span id="page-10-25"></span>[26] K. Zhou, K. Ethayarajh, D. Card, and D. Jurafsky, "Problems with cosine as a measure of embedding similarity for high frequency words," in *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, 2022, pp. 401– 423.
- <span id="page-10-26"></span>[27] Y. Wang, H. Chen, Y. Tang, T. Guo, K. Han, Y. Nie, X. Wang, H. Hu, Z. Bai, Y. Wang *et al.*, "Pangu-π: Enhancing language model architectures via nonlinearity compensation," *arXiv preprint arXiv:2312.17276*, 2023.
- <span id="page-10-27"></span>[28] J. Fang, Y. Yu, C. Zhao, and J. Zhou, "Turbotransformers: an efficient gpu serving system for transformer models," in *Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2021, pp. 389–402.
- <span id="page-10-28"></span>[29] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Re, "Flashattention: Fast and ´ memory-efficient exact attention with io-awareness," *Advances in Neural Information Processing Systems*, vol. 35, pp. 16 344–16 359, 2022.
- <span id="page-10-29"></span>[30] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga *et al.*, "Pytorch: An imperative style, high-performance deep learning library," *Advances in neural information processing systems*, vol. 32, 2019.
- <span id="page-10-30"></span>[31] M. Y. Wang, "Deep graph library: Towards efficient and scalable deep learning on graphs," in *ICLR workshop on representation learning on graphs and manifolds*, 2019.
- <span id="page-10-31"></span>[32] S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer sentinel mixture models," in *International Conference on Learning Representations*, 2022.
- <span id="page-10-32"></span>[33] P. Rajpurkar, "Squad: 100,000+ questions for machine comprehension of text," *arXiv preprint arXiv:1606.05250*, 2016.
- <span id="page-10-33"></span>[34] B. Gliwa, I. Mochol, M. Biesek, and A. Wawer, "Samsum corpus: A human-annotated dialogue dataset for abstractive summarization," in *Proceedings of the 2nd Workshop on New Frontiers in Summarization*, 2019, pp. 70–79.
- <span id="page-10-34"></span>[35] Z. Wang, M. Li, R. Xu, L. Zhou, J. Lei, X. Lin, S. Wang, Z. Yang, C. Zhu, D. Hoiem *et al.*, "Language models with image descriptors are strong few-shot video-language learners," *Advances in Neural Information Processing Systems*, vol. 35, pp. 8483–8497, 2022.
- <span id="page-10-35"></span>[36] Z. Qin, Y. Cheng, Z. Zhao, Z. Chen, D. Metzler, and J. Qin, "Multitask mixture of sequential experts for user activity streams," in *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2020, pp. 3083–3091.
- <span id="page-10-36"></span>[37] S. Zuo, X. Liu, J. Jiao, Y. J. Kim, H. Hassan, R. Zhang, T. Zhao, and J. Gao, "Taming sparsely activated transformer with stochastic experts," *arXiv preprint arXiv:2110.04260*, 2021.
- <span id="page-10-37"></span>[38] Y. Dai, X. Li, J. Liu, Z. Tong, and L.-Y. Duan, "Generalizable person reidentification with relevance-aware mixture of experts," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2021, pp. 16 145–16 154.

- <span id="page-11-0"></span>[39] H. Bao, W. Wang, L. Dong, Q. Liu, O. K. Mohammed, K. Aggarwal, S. Som, S. Piao, and F. Wei, "Vlmo: Unified vision-language pre-training with mixture-of-modality-experts," *Advances in Neural Information Processing Systems*, vol. 35, pp. 32 897–32 912, 2022.
- <span id="page-11-1"></span>[40] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann *et al.*, "Palm: Scaling language modeling with pathways," *Journal of Machine Learning Research*, vol. 24, no. 240, pp. 1–113, 2023.
- <span id="page-11-2"></span>[41] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat *et al.*, "Glam: Efficient scaling of language models with mixture-of-experts," in *International Conference on Machine Learning*. PMLR, 2022, pp. 5547–5569.
- <span id="page-11-3"></span>[42] L.-M. Team, "Llama-moe: Building mixture-of-experts from llama with continual pre-training," 2023.
- <span id="page-11-4"></span>[43] F. Xue, Z. Zheng, Y. Fu, J. Ni, Z. Zheng, W. Zhou, and Y. You, "Openmoe: An early effort on open mixture-of-experts language models," *arXiv preprint arXiv:2402.01739*, 2024.
- <span id="page-11-5"></span>[44] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu *et al.*, "Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models," *arXiv preprint arXiv:2401.06066*, 2024.
- <span id="page-11-6"></span>[45] Y. Xu, H. Lee, D. Chen, B. Hechtman, Y. Huang, R. Joshi, M. Krikun, D. Lepikhin, A. Ly, M. Maggioni *et al.*, "Gspmd: general and scalable parallelization for ml computation graphs," *arXiv preprint arXiv:2105.04663*, 2021.
- <span id="page-11-7"></span>[46] M. Lewis, S. Bhosale, T. Dettmers, N. Goyal, and L. Zettlemoyer, "Base layers: Simplifying training of large, sparse models," in *International Conference on Machine Learning*. PMLR, 2021, pp. 6265–6274.
- <span id="page-11-8"></span>[47] M. Ott, S. Edunov, A. Baevski, A. Fan, S. Gross, N. Ng, D. Grangier, and M. Auli, "fairseq: A fast, extensible toolkit for sequence modeling," *arXiv preprint arXiv:1904.01038*, 2019.
- <span id="page-11-9"></span>[48] S. Shi, X. Pan, X. Chu, and B. Li, "Pipemoe: Accelerating mixtureof-experts through adaptive pipelining," in *IEEE INFOCOM 2023-IEEE Conference on Computer Communications*. IEEE, 2023, pp. 1–10.
- <span id="page-11-10"></span>[49] Z. Zhang, D. Yang, Y. Xia, L. Ding, D. Tao, X. Zhou, and D. Cheng, "Mpipemoe: Memory efficient moe for pre-trained models with adaptive pipeline parallelism," in *2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*. IEEE, 2023, pp. 167–177.
- <span id="page-11-11"></span>[50] L. Shen, Z. Wu, W. Gong, H. Hao, Y. Bai, H. Wu, X. Wu, J. Bian, H. Xiong, D. Yu *et al.*, "Se-moe: A scalable and efficient mixtureof-experts distributed training and inference system," *arXiv preprint arXiv:2205.10034*, 2022.
- <span id="page-11-12"></span>[51] L. Zheng, Z. Li, H. Zhang, Y. Zhuang, Z. Chen, Y. Huang, Y. Wang, Y. Xu, D. Zhuo, E. P. Xing *et al.*, "Alpa: Automating inter-and {Intra-Operator} parallelism for distributed deep learning," in *16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)*, 2022, pp. 559–578.
- <span id="page-11-13"></span>[52] S. Shi, X. Pan, Q. Wang, C. Liu, X. Ren, Z. Hu, Y. Yang, B. Li, and X. Chu, "Schemoe: An extensible mixture-of-experts distributed training system with tasks scheduling," in *Proceedings of the Nineteenth European Conference on Computer Systems*, 2024, pp. 236–249.