# Growing a Multi-head Twig via Distillation and Reinforcement Learning to Accelerate Large Vision-Language Models

Zhenwei Shao, Mingyang Wang, Weijun Zhang, Zhou Yu, Wenwen Pan, Yan Yang, Tao Wei, Hongyuan Zhang, Jun Yu

**Abstract**—Large vision-language models (VLMs) have demonstrated remarkable capabilities in open-world multimodal understanding, yet their high computational overheads pose great challenges for practical deployment. Some recent works have proposed methods to accelerate VLMs by pruning redundant visual tokens guided by the attention maps of VLM's early layers. Despite the success of these token pruning methods, they still suffer from two major shortcomings: (i) considerable accuracy drop due to insensitive attention signals in early layers, and (ii) limited speedup when generating long responses (*e.g.*, 30 tokens). To address the limitations above, we present TwigVLM—a simple and general architecture by "growing" a lightweight module, named *twig*, upon an early layer of the base VLM. Compared with most existing VLM acceleration methods purely based on visual token pruning, our TwigVLM not only achieves **better accuracy retention** by employing a twig-guided token pruning (TTP) strategy, but also yields **higher generation speed** by utilizing a self-speculative decoding (SSD) strategy. Taking LLaVA-1.5-7B as the base VLM, experimental results show that TwigVLM preserves 96% of the original performance after pruning 88.9% of visual tokens and achieves 154% speedup in generating long responses, delivering significantly better performance in terms of both accuracy and speed over the state-of-the-art VLM acceleration methods. Moreover, we extend TwigVLM to an improved TwigVLM++ variant by introducing a novel multi-head twig architecture with a specialized *pruning head*. TwigVLM++ improves pruning quality via a two-stage training paradigm combining a distillation learning stage and a pruning-oriented reinforcement learning stage, and further accelerates inference via a tree-based SSD strategy. TwigVLM++ outperforms TwigVLM by 1.7% accuracy improvements and 43% speedup under the same settings, showing both the efficacy and efficiency of the extended method.

**Index Terms**—Vision-language models, model acceleration, visual token pruning, speculative decoding, multimodal learning.

✦

## **1 INTRODUCTION**

T HE revolution in large language models (LLMs) has reshaped the landscape of artificial intelligence [1], [2]. Meanwhile, there has been growing interest in building large vision-language models (VLMs) [3], [4], [5], [6], [7], which can perform visual understanding and reasoning and be used in various vision-language tasks such as object grounding [8], [9], document understanding [10], [11], GUI automation [12], [13], and robotic control [14], [15]. Despite the remarkable progress achieved by these large VLMs, they are usually parametric-intensive and computational-heavy. This poses great challenges for deploying these models in low-latency scenarios. Therefore, the research on VLM acceleration is of practical and emergent demands.

- *This work was supported in part by the National Natural Science Foundation of China under Grants (62422204, 62125201, U24B20174, 62402152, 62406093), the Key Research and Development Program of Zhejiang Province (No. 2025C01026), in part by the Zhejiang Provincial Natural Science Foundation of China under Grants LRG26F020001, LQN25F020017, LQ24F020032, and in part by the Scientific Research Innovation Capability Support Project for Young Faculty. (Corresponding author: Zhou Yu.)*
- *Z. Shao, M. Wang, W. Zhang, Z. Yu, W. Pan, and Y. Yang are with the Zhejiang Key Laboratory of Space Information Sensing and Transmission, School of Computer Science, Hangzhou Dianzi University, China. (e-mail:* {*shaozw, kaka wangmy, zhangwj, yuz, panww, yangyan*}*@hdu.edu.cn)*
- *T. Wei and H. Zhang are with Li Auto Inc., China. (e-mail:* {*weitao, zhanghongyuan*}*@lixiang.com)*
- *J. Yu is with the School of Intelligence Science and Engineering, Harbin Institute of Technology (Shenzhen), China. (e-mail: yujun@hit.edu.cn)*
- *Work was done when Z. Shao and M. Wang were interns at Li Auto Inc.*

As the computational complexity of transformer architecture [16] increases quadratically with token sequence length, reducing the number of tokens offers an effective way to accelerate VLMs. Moreover, visual tokens usually contain much more redundant information than textual tokens [17], especially in high-resolution images and long videos. Therefore, a series of works on visual token pruning has been proposed [17], [18], [19], [20]. For instance, FastV [17] leverages attention signals from an early layer of the VLM to drop redundant tokens for subsequent layers. Many following works have extended FastV's idea to progressive token reduction across layers via pruning and merging operations [18], [19]. Meanwhile, some other works have explored using the attention maps from the visual encoder to guide the visual token pruning before being input into the VLM backbone [20], [21]. Despite the success they have achieved, they still suffer from the following two major shortcomings:

(i) To effectively reduce computational costs, a large proportion of visual tokens are usually pruned in the early layers, which is guided by the attention map of the same layer. However, the attention signals in the early layers are insensitive to the task, which restricts the quality of retained visual tokens and leads to a considerable accuracy drop of the accelerated VLM.

(ii) The VLM inference process consists of the prefilling and decoding stages. Most token pruning-based approaches only focus on the acceleration of the prefilling stage while paying less attention to the decoding stage, which is the most time-consuming part of the inference process. This results in a significant efficiency gap between the theoretical estimation and practical application, especially in the tasks that generate long responses (*e.g.*, more than 30 tokens).

We ask: Is it possible to address the two limitations above in one unified framework?

To this end, we present TwigVLM, a simple and general VLM acceleration approach by appending a lightweight *twig* block upon an early layer of the base VLM. After efficient post-training of the twig block only, the learned twig block is used in both the prefilling and decoding stages of inference. As shown in Fig. 1a, our TwigVLM not only achieves better accuracy retention by employing a twigguided token pruning (TTP) strategy, but also yields higher generation speed by utilizing a self-speculative decoding (SSD) strategy [22]. To summarize, TwigVLM is easy to deploy and enables holistic inference acceleration with maximum accuracy retention.

Extensive experiments on a wide range of VLM benchmarks validate the effectiveness of the proposed method. Without bells and whistles, TwigVLM preserves 96% of the original accuracy after pruning 88.9% of visual tokens and achieves 154% speedup in generating long responses. These results demonstrate significant improvements over previous state-of-the-art methods in terms of both accuracy and speed. Moreover, our study also discloses a new aspect for VLM acceleration, *i.e.*, long response generation, which is crucial in real-world scenarios.

A preliminary version of this manuscript was published in [24]. Based on that version, we have made the following contributions to extend TwigVLM to a more powerful variant TwigVLM++ (see Fig. 1b): (i) we introduce a novel multi-head twig architecture by adding a pruning head (P-Head) alongside the original decoding head (D-Head) in TwigVLM, which decouples the functions of pruning and next-token prediction and leaves design space for pruningoriented optimization; (ii) we propose a two-stage training paradigm that combines the twig training via distillation learning in the first stage and pruning optimization via reinforcement learning in the second stage; (iii) we adopt a treebased SSD strategy to replace the original sequence-based SSD, which verifies multiple candidate sequences in parallel to increase the number of accepted tokens per verification step. Extensive and intensive experiments on multiple base VLMs across both image and video benchmarks demonstrate that TwigVLM++ delivers consistent improvements over TwigVLM. As shown in Fig. 1c, TwigVLM++ outperforms TwigVLM by 1.7% in accuracy and 43% in generation speed under the same setting, showing both the efficacy and efficiency of the extended method.

The source code is made available here<sup>1</sup>. We hope our studies may inspire future research on visual token reduction and VLM acceleration.

![](_page_1_Figure_8.jpeg)

![](_page_1_Figure_9.jpeg)

(c) Comparisons on accuracy and generation speed.

Fig. 1: **TwigVLM** and **TwigVLM++** overview. (a) Given a base deep VLM, the TwigVLM is obtained by freezing the base VLM while training a shallow *twig* block upon its early layer. Compared to the VLM acceleration methods based on visual token pruning (*e.g.*, FastV [17]), TwigVLM not only achieves better accuracy retention but also yields higher generation speed. (b) TwigVLM++ extends the design of TwigVLM by further introducing a multi-head twig architecture and a tree-based speculative decoding strategy. (c) The results are evaluated on LLaVA-1.5-7B [23] (red dotted lines mark its original performance) with the same pruning ratio of 88.9%.

#### 2 RELATED WORK

**Inference acceleration for LLMs.** Accelerating the inference process of LLMs has attracted significant attention from both academia and industry. Although various efficient model architectures have been proposed successively [25], [26], [27], they often need to be trained from scratch, thus incurring substantial costs. Consequently, the research of accelerating off-the-shelf LLMs has been extensively investigated from the prefilling and decoding stages of inference, respectively. In the prefilling stage, acceleration is achieved by redundant token reduction [28], [29] and attention operator optimization [30], [31]. In the decoding stage, techniques such as dynamic reduction of KV-cache [32], [33] and speculative decoding [22], [34], [35], [36] are introduced. In particular, speculative decoding performs parallel verification of draft tokens to boost GPU utilization, thus significantly improving the speed.

Visual token reduction for VLMs. Unlike acceleration techniques for LLMs, token reduction methods for VLMs mainly address the problem of visual token redundancy to improve inference efficiency [37]. The pioneering work of FastV [17] first studies this redundancy phenomenon, revealing that pruning visual tokens at an early layer of the VLM can achieve acceleration while preserving the majority of the model's accuracy. Following this paradigm, subsequent studies have introduced adaptive or hierarchical pruning strategies [18], [19], [38], [39], [40], [41] and training-based adoption [42], [43], [44], [45] to further reduce the accuracy drop. Meanwhile, there is another line of studies

that performs more aggressive token reduction by pruning or merging visual tokens in the visual encoding stage [20], [21], [46], [47], [48]. However, the accuracy of these pruned models is suboptimal as they are unaware of the textual prompt when pruning visual tokens. Finally, although the above methods effectively accelerate the prefilling stage of VLMs, their speedup in the decoding stage is limited.

More recently, an emerging line of work has challenged the assumption that attention-based pruning, which tends to retain the most salient tokens, is optimal for minimizing information loss [49], [50], [51], [52]. Instead, these approaches prioritize token representational diversity and global spatial dispersion to reduce redundancy and retain more complementary visual information.

Collaborative decoding of small-large VLMs. Beyond visual token reduction, some recent works have explored another direction for VLM acceleration, *i.e.*, collaborative decoding between small and large VLMs [53], [54]. Specifically, Gagrani *et al.* introduce a multimodal speculative decoding method, where a small language-only LLM is used to efficiently generate draft tokens and a large VLM then performs verification [53]. Similarly, Zhao *et al.* use a small VLM for early-exit prediction, invoking the large VLM only when the prediction confidence is low [54]. However, these methods generally rely on a pre-existing yet capable small model with the same vocabulary as the large model, which severely restricts the choice of VLMs.

To the best of our knowledge, our TwigVLM [24] is the first work that systematically studies the combination of visual token reduction and speculative decoding within a unified framework. Following this line, several recent works have explored related designs, such as tree-based speculative verification, elastic draft-side visual compression, and verifier-guided video token pruning [55], [56], [57], [58]. Nevertheless, these methods either cannot effectively reduce computation throughout the holistic inference process of VLMs, or exhibit substantially weaker accuracy retention than TwigVLM and its extension TwigVLM++.

#### 3 PILOT STUDIES

Before introducing our TwigVLM, we conduct two pilot studies on the accuracy retention and generation speed, respectively, which reveal some crucial but easily overlooked observations in previous VLM acceleration works.

#### 3.1 Preliminaries

To better understand the pilot studies, we first provide a concise overview of the architecture and two-stage inference process of modern VLMs. After that, we briefly introduce FastV [17], a typical VLM acceleration method via adaptive visual token pruning at inference.

**VLM architecture and inference.** Modern VLMs are usually comprised of a vision encoder and an LLM, where the LLM further consists of a stack of L transformer layers.

During inference, the learned VLM first encodes an input image as a sequence of M visual tokens  $\mathbf{X_v} = [\mathbf{v}_1,...,\mathbf{v}_M] \in \mathbb{R}^{M \times d}$  and represents a language prompt as a sequence of N textual tokens  $\mathbf{X_q} = [\mathbf{q}_1,...,\mathbf{q}_N] \in \mathbb{R}^{N \times d}$ , where d is the common dimensionality of the multimodal tokens. These two groups of tokens are concatenated into  $\mathbf{X} = [\mathbf{X_v}, \mathbf{X_q}]$ ,

which is passed through the LLM to generate a sequence of S response tokens  $\mathbf{y} = (y_1, ..., y_S)$  in an autoregressive manner. At each generation step j, we have:

$$y_j = \operatorname*{argmax}_{\hat{y}_j} p_{\mathrm{LLM}}(\hat{y}_j | \mathbf{X}, \mathbf{y}_{< j}), \tag{1}$$

where  $\mathbf{y}_{< j}$  is the response sequence before the current token  $y_j$ . In practice, the iterative generation process is divided into the *prefilling* and *decoding* stages by leveraging the KV-cache technique [59].

**Prefilling stage.** This stage executes the forward pass of the input tokens **X**, which remains unchanged during the response generation. Therefore, their intermediate representations in the LLM, namely the key-value pairs in each self-attention (SA) block of the transformer layer, can be cached and reused in the subsequent decoding stage. Specifically, the SA operation is given by:

$$SA(Q, K, V) = AV = softmax(\mathcal{M}(QK^T/\sqrt{d}))V$$
  
 $Q = XW_q, \quad K = XW_k, \quad V = XW_v,$ 
(2)

where  $\mathcal{M}(\cdot)$  is the causal masking function and  $\mathbf{A}$  is the attention map of query-key pairs.  $\mathbf{Q}, \mathbf{K}, \mathbf{V}$  denote the queries, keys, and values, while  $\mathbf{W_q}, \mathbf{W_k}, \mathbf{W_v}$  are their linear projection matrices, respectively.

As mentioned, the key-value pairs  ${\bf K}$  and  ${\bf V}$  of all transformer layers in the LLM are stored in the KV-cache to expedite the decoding stage.

**Decoding stage.** The generation of the response token  $y_j$  is based on its previous tokens  $(\mathbf{X}, \mathbf{y}_{< j})$ . As the attention computation in Eq.(2) is causal and parallel for each query, the generation for  $y_j$  can be accelerated by feeding only the *single* token into the LLM and computing its corresponding key and value, while reusing the rest key-value pairs from the KV-cache. After each decoding step, the computed key-value pairs for  $y_j$  are appended to the KV-cache for the efficient generation of subsequent tokens.

**Visual token pruning.** The visual tokens in VLMs are much more than the textual tokens and exhibit severe redundancy [18], [19]. To accelerate the inference process of VLMs, some recent works, such as FastV [17] and SparseVLM [18], have been proposed to drop redundant visual tokens based on the attention scores during the prefilling stage. Specifically, FastV works by passing the input tokens  $\mathbf{X}$  through the first K VLM layers to obtain the attention map  $\mathbf{A}^{(K)}$  and token sequence  $\mathbf{X}^{(K)} = [\mathbf{X}_{\mathbf{v}}^{(K)}, \mathbf{X}_{\mathbf{q}}^{(K)}]$  after layer K. Denoting R as the number of visual tokens after pruning, the retained token sequence  $\hat{\mathbf{X}}^{(K)}$  is obtained as follows:

$$\hat{\mathbf{X}}^{(K)} = \mathcal{P}(\mathbf{X}^{(K)}, \mathbf{A}^K, R) 
= \left[ \text{TopR}\left(\mathbf{X}_{\mathbf{v}}^{(K)}, \sum_{i=M+1}^{M+N} \mathbf{A}^{(K)}[i, 1:M] \right), \mathbf{X}_{\mathbf{q}}^{(K)} \right].$$
(3)

 $\operatorname{TopR}(\mathbf{X}_{\mathbf{V}}^{(K)},\mathbf{a})$  selects top-R most significant visual tokens from  $\mathbf{X}_{\mathbf{V}}^{(K)}$ , which is guided by the attention scores  $\mathbf{a}$  from textual tokens to visual tokens. In FastV, both K and R are set to small values, which means the token lengths are short in most VLM layers. Thus, the computational complexity of the prefilling stage is significantly reduced at the expense of a certain amount of performance drop. For a fair comparison

![](_page_3_Figure_2.jpeg)

Fig. 2: Attention quality for token selection. The RelAcc (defined in §6.2) is evaluated on GQA [60] and TextVQA [61], which measures the accuracy gap between LLaVA-1.5-7B and its FastV variants with R=128 and varied attention layer depth D. Taking an image with two prompts as an example, we visualize their attention maps from typical layers D=2 and 18, respectively.

of different token pruning methods, the average number of retained tokens  $\bar{R}$  is used and defined as follows<sup>2</sup>:

$$\bar{R} = [M \times K + R \times (L - K)]/L \tag{4}$$

## 3.2 Study 1: Attention Quality for Token Selection

The performance of visual token pruning methods heavily depends on the quality of the chosen attention map for pruning. Existing approaches often utilize the attention map from an early layer, e.g., K=2, to guide the visual token pruning at the same layer. We ask: Whether an attention map from a late layer contains more informative signals to facilitate token pruning? Notably, directly applying the attention map of a late layer to guide the token pruning at an early layer will inevitably introduce redundant computation, so this pilot study only aims to evaluate the quality of the attention maps in different layers.

The pilot experiment is conducted as follows. We use LLaVA-1.5-7B as the base VLM and FastV as the token pruning strategy. Different from the original FastV that selects and prunes visual tokens at the same layer K, we perform token selection guided by the attention of the *D*-th layer and then perform token pruning at the K-th layer. In this experiment, we set K=2 and let  $D \ge K$ . From the results in Fig. 2, we can see that the accuracy grows as the attention depth Dincreases, showing that the attention maps from later layers provide more precise signals for visual token pruning than those from early layers. We hypothesize that later layers are closer to the prediction head (i.e., the loss function), thus their attention maps understand the relation between multimodal tokens more accurately. Moreover, from the visualized attention maps, we find that the selected tokens by early-layer attention (D=2) are insensitive to different

2. The computed result will be rounded to the nearest integer.

![](_page_3_Figure_10.jpeg)

(a) Decoding time with varied S

(b) P&D time at S=32

Fig. 3: **Prefilling and decoding time costs**. (a) Prefilling time (gray dotted line) and decoding time for LLaVA-1.5-7B [23] with different response lengths. (b) Prefilling (P) and decoding (D) time comparisons of LLaVA-1.5-7B and its FastV-based variant.

prompts, while the selected tokens by late-layer attention (D=18) are more related to the prompt. This observation also verifies our hypothesis above.

## 3.3 Study 2: Prefilling and Decoding Time Costs

As mentioned above, the inference process consists of the prefilling and decoding stages. FastV and other token-pruning methods primarily accelerate the prefilling stage. We ask: Can these methods also facilitate the acceleration of the decoding stage, especially when the generated response is relatively long? This is a practical question since VLMs are often expected to generate long responses (e.g., 30 tokens) in real-world scenarios.

To answer this question, we conduct the pilot experiment as follows. We take LLaVA-1.5-7B [23] as the base VLM and compute its time spent on the prefilling and decoding stages for the response length S ranging from 2 to 128. The results in Fig. 3a show a linear increase in decoding time w.r.t. the response length S, while the prefilling time can be neglected when  $S \ge 32$ .

Next, we compare the prefilling and decoding times of LLaVA-1.5-7B and its FastV-based variant (LLaVA+FastV) when generating responses of the same length S=32. From Fig. 3b, we observe that although FastV effectively reduces the prefilling time, it attains limited speedup in the decoding stage. This frustrating observation can be explained by two facts: (i) the KV-cache mechanism undermines the speedup for SA blocks achieved by FastV, and (ii) FastV does not accelerate the FFN blocks during decoding, but they account for the majority of the computational costs.

To summarize, these observations highlight the significance and necessity of improving the decoding efficiency for long response generation.

## 4 THE PROPOSED TWIGVLM

Inspired by the pilot studies above, we propose TwigVLM, a simple yet effective approach for accelerating VLMs. As depicted in Fig. 4, our TwigVLM is obtained by training a lightweight twig block upon an early layer of a frozen

![](_page_4_Figure_2.jpeg)

Fig. 4: **Training and two-stage inference of TwigVLM**. (a) The twig block is initialized from the base VLM and is coupled with the first *K* layers of base VLM to form a shallow model, which can be trained efficiently. (b) In the prefilling stage, different from previous approaches that perform token selection based on the attention maps from the base VLM, TwigVLM introduces a twig-guided token pruning (TTP) strategy to obtain more precise signals to guide the pruning of visual tokens. (c) In the decoding stage, the shallow and deep sub-networks act as the draft and target models, respectively, enabling it to perform the self-speculative decoding (SSD) [22] to accelerate the generation of long responses.

VLM. Subsequently, the learned twig block is used in both the prefilling and decoding stages to simultaneously achieve more precise visual token pruning and faster generation speed for long responses.

#### 4.1 Architecture and Training of TwigVLM

Denoting a L-layer base VLM as  $\mathcal{M}_b = \{\mathcal{T}_l\}_{l=1}^L$ , we attach a twig block  $\{\mathcal{G}_t\}_{t=1}^T$ , i.e., an extra lightweight module with T transformer layer, upon an early layer of  $\mathcal{M}_b$ . Specifically, the twig block is appended after the K-th layer  $\mathcal{T}_K$ , resembling a twig growing from a tree trunk. When  $K + T \ll L$ , we obtain a very shallow VLM by combining the twig block and its based K trunk layers  $\mathcal{M}_s = \{\mathcal{T}_k\}_{k=1}^K \cup \{\mathcal{G}_t\}_{t=1}^T$ .

As shown in Fig. 4a, the training strategy of TwigVLM is simple yet efficient. All the model weights in the twig block are initialized from the layers  $\{\mathcal{T}_i\}_{i=K+1}^{K+T}$  and the prediction head of  $\mathcal{M}_b$ , respectively. After that, the shallow VLM  $\mathcal{M}_s$  is finetuned using the same training data and the conventional autoregressive (AR) loss as those for  $\mathcal{M}_b$ . Notably, only the lightweight twig block is updated during the TwigVLM training, consuming only about 10% of the training time compared to its base VLM.

#### 4.2 Efficient inference with TwigVLM

The obtained TwigVLM can be seen as a composition of a deep VLM  $\mathcal{M}_b$  and a shallow VLM  $\mathcal{M}_s$ , where their first K layers are shared with each other. We can leverage this ingenious architecture to accelerate both the prefilling and decoding stages during inference.

3. For notational simplicity, we omit the visual encoder and the prediction head in the definition of  $\mathcal{M}_b$  and  $\mathcal{M}_s$ .

Prefilling acceleration via token pruning. The prefilling stage can be accelerated by applying the visual token pruning strategy. Previous methods rely on attention maps from early layers to select key visual tokens and then prune the rest. As discussed in §3.2, we have verified that the attention map of a late layer, which is closer to the prediction head, can provide more precise signals than that of an early layer. Inspired by this observation, we introduce a twig-guided token pruning (TTP) strategy that utilizes the attention map from the last twig layer to select key tokens.

As shown in Fig. 4b, given the input tokens  $\mathbf{X}$ , the TTP strategy first feeds them through the first K layers of  $\mathcal{M}_b$  to obtain the latent representations  $\mathbf{X}_{\mathcal{M}_b}^{(K)}$ , and then feeds them through the twig block to obtain the attention map of the last twig layer  $\mathbf{A}_{\mathcal{M}_s}^{(K+T)}$ . After that, we use the token pruning function in Eq.(3) to obtain the representations of the retained visual tokens  $\hat{\mathbf{X}}_{\mathcal{M}_b}^{(K)}$  as follows:

$$\hat{\mathbf{X}}_{\mathcal{M}_b}^{(K)} = \mathcal{P}(\mathbf{X}_{\mathcal{M}_b}^{(K)}, \mathbf{A}_{\mathcal{M}_s}^{(K+T)}, R), \tag{5}$$

where this condensed token sequence  $\hat{\mathbf{X}}_{\mathcal{M}_b}^{(K)}$  is further fed through the remaining layers of  $\mathcal{M}_b$ .

To achieve aggressive token pruning with small  $\bar{R}$ , both K and R need to be small values according to Eq.(4), leading to considerable performance drops. To mitigate this issue, several works adopt *progressive* token pruning strategies [18], [43] to retain more tokens at middle layers. In addition, previous studies have shown that the visual tokens in late layers (*e.g.*, after the 20th layer) barely contribute to the prediction [37], [44]. Motivated by these works, we additionally introduce a simple *FinalWipe* strategy by removing *all* the visual tokens after the  $K_{\rm f}$ -th layer (a late

layer) in  $\mathcal{M}_b$ . Thus, Eq.(4) can be reformulated as follows:

$$\bar{R} = [M \times K + R \times (K_f - K)]/L. \tag{6}$$

When  $\bar{R}$  and K are fixed, this strategy enables a larger R, which facilitates the performance of our TwigVLM.

**Decoding acceleration via self-speculation.** The acceleration for the decoding stage remains underexplored, especially for the long response scenarios. TwigVLM's architecture, which contains two sub-networks, namely the deep one  $\mathcal{M}_b$  and shallow one  $\mathcal{M}_s$ , enables the self-speculative decoding (SSD) strategy [22], [36] to address this challenge. As illustrated in Fig. 4c, we take the shallow model  $\mathcal{M}_s$  as a *draft* model to efficiently generate multiple subsequent tokens (*i.e.*, draft tokens), and then take the deep model  $\mathcal{M}_b$  as a *target* model to verify these tokens in parallel.

Specifically, SSD operates through multiple draft-thenverify iterations to generate the response. In each iteration, the draft model generates multiple tokens in an autoregressive manner. These tokens are then verified by feeding them through the target model in a single forward pass to determine their acceptance. Although the total computational costs are not reduced, the SSD strategy accelerates the decoding stage by processing multiple draft tokens in parallel, which maximally utilizes GPUs' parallel computing capabilities. Moreover, the draft and target models in TwigVLM share the computation and KV-cache of the first K layers, thus achieving further efficiency gains.

It is worth noting that the SSD strategy produces the same response as the target model, which means the accuracy of TwigVLM is only affected by the TTP strategy.

## 5 EXTENDING TWIGVLM TO TWIGVLM++

The pruning capability of the original TwigVLM emerges as a by-product of autoregressive training and is never directly optimized, which limits its effectiveness on stronger or larger base VLMs. To address this, we extend TwigVLM to an improved TwigVLM++ variant by introducing a novel *multi-head* twig architecture with two decoupled heads. Accordingly, we introduce a two-stage training paradigm: (i) twig training via distillation learning (stage-1) and (ii) pruning optimization via reinforcement learning (stage-2). Detailed model architecture and training paradigm are shown in Fig. ??. Furthermore, we leverage a tree-based self-speculative decoding (SSD) to replace the original sequence-based SSD, yielding higher inference throughput.

#### 5.1 Multi-head Twig Architecture

In TwigVLM, the attention map of the last twig layer  $\mathcal{G}_T$  serves both autoregressive token prediction and visual token pruning. TwigVLM++ decouples these two functions by extending the twig block with a *multi-head* architecture (see Fig. 1b), which consists of a **decoding head** (D-Head) and a **pruning head** (P-Head). The D-Head retains the standard next-token prediction function of the original twig block. The P-Head is a lightweight auxiliary module attached to the self-attention (SA) layer of  $\mathcal{G}_T$ , dedicated to computing visual token importance scores.

**P-Head design.** Let  $\mathbf{X}^{(K+T)}$  denote the input to the SA layer of  $\mathcal{G}_T$ . We extract the hidden state at the last textual token

![](_page_5_Figure_13.jpeg)

- (a) Twig training via distillation learning (§5.2)
- (b) Pruning optimization via reinforcement learning (§5.3)

Fig. 5: The multi-head twig Architecture and two-stage training paradigm of TwigVLM++.

position as  $\mathbf{X_q} \in \mathbb{R}^d$  and the hidden states at visual token positions as  $\mathbf{X_k} \in \mathbb{R}^{M \times d}$ . From the projected query and key representations  $\mathbf{Q} = \mathbf{X}^{(K+T)}\mathbf{W_q}$  and  $\mathbf{K} = \mathbf{X}^{(K+T)}\mathbf{W_k}$ , we extract the query vector  $\tilde{\mathbf{q}} \in \mathbb{R}^{H \times d_h}$  and key matrix  $\tilde{\mathbf{K}} \in \mathbb{R}^{H \times M \times d_h}$ , where H is the number of attention heads and  $d_h$  is the head dimension. The P-Head applies two learnable gating projections  $\mathbf{G}_q$  and  $\mathbf{G}_k$  (each a linear layer followed by a nonlinear activation) to modulate these representations. The importance score  $\mathbf{s} \in \mathbb{R}^M$  is then computed as:

$$\mathbf{s} = \frac{1}{H} \sum_{h=1}^{H} \sigma \left( \frac{\left( \mathbf{G}_{q}(\mathbf{x}_{\mathbf{q}})^{(h)} \odot \tilde{\mathbf{q}}^{(h)} \right) \left( \mathbf{G}_{k}(\mathbf{X}_{\mathbf{k}})^{(h)} \odot \tilde{\mathbf{K}}^{(h)} \right)^{T}}{\sqrt{d_{h}}} \right),$$

where  $\sigma(\cdot)$  is the softmax function,  $(\cdot)^{(h)}$  denotes the h-th head component and  $\odot$  is the element-wise product. The resulting s is a normalized score over visual tokens. During inference, the pruning function in Eq.(5) is replaced by:

$$\hat{\mathbf{X}}_{\mathcal{M}_b}^{(K)} = \mathcal{P}(\mathbf{X}_{\mathcal{M}_b}^{(K)}, \mathbf{s}, R). \tag{8}$$

By separating the pruning and prediction pathways, the multi-head architecture provides parameter decoupling between the two tasks, allowing each head to be optimized toward its own objective. More importantly, it creates the design space for targeted pruning optimization, as described in the following training stages.

## 5.2 Twig Training via Distillation Learning

As illustrated in Fig. 5(a), in the first training stage, we train the twig block (including the P-Head) using the standard autoregressive next-token prediction (NTP) loss, augmented with two distillation losses that leverage the frozen base VLM  $\mathcal{M}_b$  as the teacher.

**PredKL.** To improve the alignment between the twig (draft) model and the base (target) model, we introduce a prediction-level KL divergence loss:

$$\mathcal{L}_{\text{PredKL}} = \text{KL}(p_{\mathcal{M}_b} \parallel p_{\mathcal{M}_s}), \tag{9}$$

where  $p_{\mathcal{M}_s}$  and  $p_{\mathcal{M}_b}$  are the next-token prediction distributions of  $\mathcal{M}_s$  and  $\mathcal{M}_b$ , respectively. This strong-to-weak

distillation provides the twig block with richer supervisory signals, enhancing its understanding of visual tokens and thereby improving the quality of importance score estimation for pruning.

**AttnKL.** To directly supervise the pruning signal, we introduce an attention-level KL divergence loss. Let  $\mathbf{a}_b \in \mathbb{R}^M$  denote the attention distribution from textual tokens to visual tokens at a designated layer of  $\mathcal{M}_b$ , averaged over all attention heads. The AttnKL loss is defined as:

$$\mathcal{L}_{\text{AttnKL}} = \text{KL}(\mathbf{a}_b \parallel \mathbf{s}), \qquad (10)$$

where s is the P-Head score from Eq.(7). This loss guides the P-Head to produce importance scores that are consistent with the attention patterns of the deep model, which has been shown to provide more precise signals for token selection (see §3.2).

The overall loss for the first training stage is:

$$\mathcal{L}_{\text{stage-1}} = \mathcal{L}_{\text{NTP}} + \alpha \cdot \mathcal{L}_{\text{PredKL}} + \gamma \cdot \mathcal{L}_{\text{AttnKL}}, \tag{11}$$

where  $\alpha$  and  $\gamma$  are two balancing hyper-parameters.

#### 5.3 Pruning Optimization via Reinforcement Learning

After the first stage, the P-Head has acquired a reasonable pruning ability. As illustrated in Fig. 5(b), in the second stage, we further optimize the P-Head's parameters via reinforcement learning (RL) to directly maximize the post-pruning model performance, while the rest of the model weights are frozen. This training stage re-uses the SFT dataset from the first stage but only needs  $\sim\!\!10\%$  of the training samples. Therefore, this training stage is highly efficient in both computation and data.

Action space. We formulate the visual token pruning as a sequential decision process. Given the importance score distribution  $\pi(\cdot)=\mathbf{s}$  produced by the P-Head, a pruning action a consists of selecting R visual token positions to retain. To make this combinatorial selection amenable to policy gradient optimization, we model it as a sequential sampling process without replacement: at each step, a token position is sampled according to the current distribution, then its probability is set to zero, and the distribution is renormalized before the next sampling step. The probability of a complete action  $\mathbf{a}=(a_1,\ldots,a_R)$  is:

$$\pi_{\theta}(\mathbf{a}) = \prod_{j=1}^{R} \frac{\pi(a_j)}{\sum_{m \notin \{a_1, \dots, a_{j-1}\}} \pi(m)}.$$
 (12)

**Reward function.** Instead of relying on active roll-out and explicit answer verification, we adopt a reference-based reward for RL training [62], [63]. The reward is derived from the model's own log-likelihood on the reference answer—the annotated response readily available in standard SFT data—under the pruned input. Specifically, given a training sample with input  $\mathbf{X}$  and ground-truth response  $\mathbf{y}^* = (y_1^*, \dots, y_S^*)$ , we apply the pruning action  $\mathbf{a}$  to obtain the pruned input  $\hat{\mathbf{X}}$ , then feed it through the base model  $\mathcal{M}_b$  to compute the per-token log-probability of generating  $\mathbf{y}^*$ . The reward of action  $\mathbf{a}$  is defined as:

$$r(\mathbf{a}) = \frac{1}{S} \sum_{i=1}^{S} \log p_{\mathcal{M}_b}(y_j^* \mid \hat{\mathbf{X}}, \mathbf{y}_{< j}^*),$$
 (13)

which is the mean log-probability (i.e., the negative lengthnormalized cross-entropy loss) of the reference answer. A higher reward indicates that the pruning better preserves the task-relevant information needed for generating the correct response.

**Training procedure.** We adopt a GRPO-style [64] optimization. For each training sample, we sample G pruning actions from the current policy  $\pi_{\theta}$  and compute their rewards. The advantage is estimated via group-level normalization:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})},$$
(14)

where  $r_i=r(\mathbf{a}_i)$  are the rewards of the i-th action, mean $(\cdot)$  and  $\mathrm{std}(\cdot)$  are the mean and standard deviation functions. Since we perform strict on-policy updates (one policy sampling followed by one gradient step), the importance ratio  $\pi_{\theta}/\pi_{\theta_{\mathrm{old}}}$  equals 1 and the clipping mechanism in GRPO does not activate. Thus, the loss simplifies to:

$$\mathcal{L}_{\text{stage-2}} = \frac{1}{G} \sum_{i=1}^{G} \hat{A}_i \cdot \log \pi_{\theta}(\mathbf{a}_i). \tag{15}$$

**Dynamic pruning-ratio schedule.** To enable a single trained model to support different pruning ratios at test time without retraining, we randomize  $\bar{R}$  during RL training. At each training step,  $\bar{R}$  is sampled from a predefined candidate set  $\mathcal{R} = \{\bar{R}_1, \dots, \bar{R}_n\}$  (sorted in ascending order), so that the P-Head learns to produce effective importance scores across a range of pruning ratios.

Since smaller R values (more aggressive pruning) are inherently harder to optimize, we adopt a curriculum-based schedule whose sampling distribution gradually shifts toward smaller values as training progresses:

$$P(\bar{R} = \bar{R}_i) = \frac{\exp(-\beta(t) \cdot i)}{\sum_{j=1}^n \exp(-\beta(t) \cdot j)}, \quad i = 1, 2, \dots, n, (16)$$

where  $\beta(t) = \beta_{\max} \cdot (t/T)^p$  is an annealing parameter, t and T are the current and total training steps, and p controls the annealing speed. The distribution is uniform at t=0 and concentrates on  $\bar{R}_1$  as  $t \to T$ , progressively directing the optimization toward the most aggressive pruning ratio.

#### 5.4 Inference Acceleration via Tree-based SSD

As described in §4.2, TwigVLM accelerates the decoding stage via self-speculative decoding (SSD), where  $\mathcal{M}_s$  drafts a single token sequence and  $\mathcal{M}_b$  verifies it. The efficiency of this approach is limited by the number of accepted tokens per verification step. To improve this, TwigVLM++ adopts a tree-based SSD strategy [65].

Instead of generating a single draft sequence,  $\mathcal{M}_s$  constructs a *token tree* governed by an expansion width  $\mathcal{E}$ , a selection width  $\mathcal{K}$ , and a tree depth  $\mathcal{D}$ . At each level, the top- $\mathcal{K}$  nodes are selected for expansion, and each selected node branches into  $\mathcal{E}$  children corresponding to its top- $\mathcal{E}$  predicted tokens from  $\mathcal{M}_s$ . Each root-to-leaf path in the resulting tree forms a distinct candidate sequence, increasing the coverage of the target model's prediction. For parallel verification,  $\mathcal{M}_b$  processes the entire tree in a single forward pass using *tree attention* [65], which replaces the standard causal mask with a *topology-aware causal mask* so that each

node attends only to its ancestors. The verified tokens are determined by traversing the tree from the root until no child matches the target model's output.

Although the drafting and verification of the token tree introduce more computational overheads, tree-based SSD yields more accepted tokens per verification step due to the expanded path coverage, translating to higher decoding throughput in practice.

## **6 EXPERIMENTS**

We evaluate the performance of TwigVLM and TwigVLM++ on three popular VLMs, namely LLaVA-1.5-7B [23], LLaVA-NeXT-7B [66] and Qwen2.5-VL-7B [5], to compare with the state-of-the-art VLM acceleration methods. After that, we conduct comprehensive ablation studies to analyze the effectiveness of the key elements of TwigVLM and TwigVLM++.

#### **6.1 Implementation Details**

Unless otherwise noted, we implement TwigVLM with the following hyper-parameters: the number of twig layers T=3, the token pruning position K=2, and the FinalWipe position Kf=24. Based on the default setting above, we adjust R using Eq.(6) to fairly compare TwigVLM with other methods at the same pruning ratio 1 − R/M¯ .

For TwigVLM++, the stage-1 training follows the same protocol as TwigVLM but additionally trains the multiple heads with the distillation losses (α=0.1, γ=1.0 in Eq.(11)). In stage-2, we train the P-Head using G=32 groups per sample on 50K samples from the same SFT dataset as stage-1. The candidate set for the dynamic pruning-ratio schedule is R={64, 85, 107, 128, 149, 171, 192}. The annealing parameters are set to βmax=8.0 and p=2.0. For tree-based SSD, we use the expansion configuration of E=10, K=10 and D=4.

For training data, we use the LLaVA-665K dataset [23] to train TwigVLM/TwigVLM++ for the LLaVA-1.5-7B and LLaVA-NeXT-7B models, and a dataset of 5M single-image samples from the MAmmoTH-VL-10M dataset [67] for the Qwen2.5-VL-7B model. All experiments are conducted on a server with 8×A100 GPUs. More implementation details are provided in the supplementary.

#### **6.2 Evaluation Metrics**

To evaluate the performance of TwigVLM, we consider two key aspects: *accuracy preservation* and *inference speedup*, which are respectively measured by the two metrics as follows. **Relative Accuracy (RelAcc)** is a widely used metric in previous works [17], [20], which calculates the proportion of the accuracy of the pruned model to the base model. When multiple benchmarks are provided, their relative accuracies are separately calculated and then averaged as the final RelAcc. **Relative Speed (RelSpd)** quantifies the relative generation speed of a pruned model compared to its base model. The generation speed is computed by dividing the number of generated tokens by the generation time—which encompasses both the prefilling and decoding stages—and then averaged across samples from a specified benchmark.

| Method          |      |      |      |                                      |      | GQA MMB MME VQAT SQAI VQAV2 RelAcc |       |
|-----------------|------|------|------|--------------------------------------|------|------------------------------------|-------|
|                 |      |      |      | Upper Bound, 576 Tokens (100%)       |      |                                    |       |
| LLaVA-1.5-7B    | 61.9 | 64.7 | 1862 | 58.2                                 | 69.5 | 78.5                               | 100%  |
|                 |      |      |      | Retain Averaged 192 Tokens (↓ 66.7%) |      |                                    |       |
| FastV [17]      | 56.5 | 63.7 | 1786 | 57.3                                 | 69.5 | 74.6                               | 96.5% |
| SparseVLM [18]  | 57.6 | 62.5 | 1721 | 56.1                                 | 69.1 | 75.6                               | 95.7% |
| PDrop [43]      | 57.3 | 63.3 | 1797 | 56.5                                 | 69.2 | 75.1                               | 96.5% |
| MustDrop [19]   | 58.2 | 62.3 | 1787 | 56.5                                 | 69.2 | 76.0                               | 96.6% |
| VisionZip [20]  | 59.3 | 63.0 | 1783 | 57.3                                 | 68.9 | 76.8                               | 97.4% |
| VisionZip‡ [20] | 60.1 | 63.4 | 1834 | 57.8                                 | 68.2 | 77.4                               | 98.3% |
| TwigVLM         | 61.2 | 64.0 | 1848 | 58.0                                 | 68.8 | 78.1                               | 99.2% |
| TwigVLM++       | 61.2 | 64.3 | 1868 | 58.0                                 | 69.2 | 78.2                               | 99.6% |
|                 |      |      |      | Retain Averaged 128 Tokens (↓ 77.8%) |      |                                    |       |
| FastV           | 53.0 | 61.4 | 1646 | 56.0                                 | 69.5 | 69.2                               | 92.2% |
| SparseVLM       | 56.0 | 60.0 | 1696 | 54.9                                 | 67.1 | 73.8                               | 93.2% |
| PDrop           | 57.1 | 61.6 | 1761 | 56.6                                 | 68.4 | 72.9                               | 95.1% |
| MustDrop        | 56.9 | 61.1 | 1745 | 56.3                                 | 68.5 | 74.6                               | 95.1% |
| VisionZip       | 57.6 | 62.0 | 1762 | 56.8                                 | 68.9 | 75.6                               | 96.1% |
| VisionZip‡      | 58.9 | 62.6 | 1823 | 57.0                                 | 68.3 | 76.6                               | 97.3% |
| TwigVLM         | 60.6 | 63.5 | 1818 | 57.8                                 | 69.5 | 77.9                               | 98.7% |
| TwigVLM++       | 60.8 | 63.7 | 1856 | 58.0                                 | 69.5 | 77.9                               | 99.2% |
|                 |      |      |      | Retain Averaged 64 Tokens (↓ 88.9%)  |      |                                    |       |
| FastV           | 44.1 | 45.9 | 1218 | 50.7                                 | 70.0 | 52.0                               | 77.0% |
| SparseVLM       | 52.7 | 56.2 | 1505 | 51.8                                 | 62.2 | 68.2                               | 89.9% |
| PDrop           | 47.5 | 58.8 | 1561 | 50.6                                 | 69.0 | 69.2                               | 87.6% |
| FasterVLM [21]  | 51.5 | 58.5 | 1573 | 53.1                                 | 69.6 | 66.8                               | 89.1% |
| MustDrop        | 53.1 | 60.0 | 1612 | 54.2                                 | 63.4 | 69.3                               | 89.6% |
| VisionZip       | 55.1 | 60.1 | 1690 | 55.5                                 | 69.0 | 72.4                               | 93.3% |
| VisionZip‡      | 57.0 | 61.5 | 1756 | 56.0                                 | 68.8 | 74.2                               | 95.2% |
| TwigVLM         | 58.8 | 60.4 | 1760 | 55.8                                 | 70.0 | 75.6                               | 96.0% |
| TwigVLM++       | 59.7 | 63.2 | 1801 | 56.7                                 | 69.5 | 76.8                               | 97.7% |

TABLE 1: Performance comparisons with three pruning ratios on six VLM benchmarks [60], [61], [68], [69], [70], [71]. All methods are applied on the same base model **LLaVA-1.5- 7B**. The best result for each benchmark and pruning ratio is **bolded** and the second best result is underlined.

#### **6.3 Main Results**

**Results on LLaVA-1.5 and LLaVA-NeXT.** As shown in TABLE 1 and 2, we take LLaVA-1.5-7B and LLaVA-NeXT-7B as the base VLMs and compare TwigVLM with the existing VLM acceleration methods on six commonly-used VLM benchmarks. We can see that TwigVLM consistently and significantly outperforms all its counterparts under different pruning ratios. Notably, TwigVLM achieves nearperfect performance preservation (more than 99.0%) when pruning 66.7% visual tokens on LLaVA-1.5-7B and 77.8% tokens on LLaVA-NeXT-7B, showing the superiority of our TTP strategy over existing pruning strategies. Furthermore, TwigVLM++, equipped with the RL-optimized pruning, maintains the already high accuracy at moderate pruning ratios (e.g., 99.6% on LLaVA-1.5-7B with 66.7% pruning), while yielding more substantial improvements under aggressive pruning—boosting RelAcc from 96.0% to 97.7% on LLaVA-1.5-7B and from 96.2% to 97.1% on LLaVA-NeXT-7B at the highest pruning ratio (88.9%).

To better understand the effectiveness of our approach, we compare TwigVLM and TwigVLM++ with two representative methods [17], [20] by visualizing the attention map

| Method        | GQA                                  | MMB         | MME         | $\boldsymbol{VQ}\boldsymbol{A}^T$ | $\textbf{SQA}^{\text{I}}$ | $\boldsymbol{VQA^{V2}}$ | RelAcc |  |
|---------------|--------------------------------------|-------------|-------------|-----------------------------------|---------------------------|-------------------------|--------|--|
|               | Upper                                | Bound,      | 2880 T      | okens (1                          | 100%)                     |                         |        |  |
| LLaVA-NeXT-7B | 64.2                                 | 67.9        | 1851        | 61.3                              | 70.2                      | 81.8                    | 100%   |  |
| Re            | Retain Averaged 640 Tokens (↓ 77.8%) |             |             |                                   |                           |                         |        |  |
| FastV         | 62.0                                 | 65.8        | 1807        | 60.0                              | 69.1                      | 79.5                    | 97.6%  |  |
| SparseVLM     | 60.3                                 | 65.7        | 1772        | 57.8                              | 67.7                      | 77.1                    | 95.4%  |  |
| VisionZip     | 61.3                                 | 66.3        | 1787        | <u>60.2</u>                       | 68.1                      | 79.1                    | 97.1%  |  |
| VisionZip‡    | 62.4                                 | 65.9        | 1778        | 60.8                              | 67.9                      | 79.9                    | 97.4%  |  |
| TwigVLM       | <u>63.4</u>                          | <u>67.4</u> | 1864        | 58.6                              | <u>69.9</u>               | 81.2                    | 99.0%  |  |
| TwigVLM++     | 63.4                                 | 67.6        | <u>1858</u> | 58.6                              | 70.2                      | 81.2                    | 99.0%  |  |
| Re            | etain A                              | veraged     | 320 Tol     | kens (↓                           | 88.9%                     | )                       |        |  |
| FastV         | 54.9                                 | 60.0        | 1539        | 54.8                              | 68.2                      | 69.6                    | 88.2%  |  |
| SparseVLM     | 57.7                                 | 64.3        | 1694        | 55.9                              | 67.3                      | 73.4                    | 92.3%  |  |
| MustDrop      | 57.3                                 | 62.8        | 1641        | 59.9                              | 68.0                      | 73.7                    | 92.6%  |  |
| VisionZip     | 59.3                                 | 63.1        | 1702        | 58.9                              | 67.3                      | 76.2                    | 93.8%  |  |
| VisionZip‡    | 61.0                                 | 64.4        | <u>1770</u> | <u>59.3</u>                       | 67.5                      | 78.4                    | 95.8%  |  |
| TwigVLM       | <u>62.2</u>                          | <u>65.0</u> | 1758        | 57.4                              | <u>68.7</u>               | <u>79.7</u>             | 96.2%  |  |
| TwigVLM++     | 62.2                                 | 66.6        | 1773        | 57.8                              | 69.0                      | 80.4                    | 97.1%  |  |

TABLE 2: Performance comparisons on **LLaVA-NeXT-7B**. This table follows the same layout and notes as TABLE 1.

used for token selection. The examples in Fig. 6 suggest that TwigVLM/TwigVLM++ can better understand the *fine-grained* semantics in both the prompt and image, thus identifying more informative visual tokens for token pruning. Compared to TwigVLM, TwigVLM++ produces more accurate focal points while maintaining the necessary dispersed attention pattern.

Results on Qwen2.5-VL for image and video understanding. To evaluate the generalization of TwigVLM and TwigVLM++ on stronger base VLMs, we further conduct experiments on Qwen2.5-VL-7B across both image and video benchmarks. As shown in TABLE 3, TwigVLM consistently surpasses FastV and achieves comparable RelAcc with VisionZip across all pruning ratios, while TwigVLM++ substantially outperforms VisionZip by a large margin (e.g., 94.4% vs. 88.4% at 88.9% pruning). On video benchmarks, TwigVLM performs comparably to VisionZip, and TwigVLM++ further surpasses it in most cases, demonstrating the generalization of our approach to the video domain. Notably, the accuracy gains brought by TwigVLM++ are more pronounced on Qwen2.5-VL than on LLaVA-1.5 (e.g., +8.4% vs. +1.7% in RelAcc at 88.9% pruning), suggesting that the proposed multi-head twig architecture and twostage training paradigm yield greater benefits on stronger base VLMs.

Generation speed comparisons. As mentioned above, TwigVLM can effectively accelerate the generation. To validate this, we conduct intensive experiments on two typical benchmarks TextVQA [61] and MM-Vet [72] to compare the speedup of different VLM acceleration methods based on LLaVA-1.5-7B. The results in Fig. 7 show that: (i) TwigVLM achieves superior or competitive speedup in all configurations, suggesting dual advantages in speed and accuracy. (ii) All the methods attain a similar level of RelSpd (120%~130%) on TextVQA with short responses. In contrast, TwigVLM delivers significantly higher speedup (~150%) than FastV (~104%) and VisionZip (~106%) on MM-Vet with long responses. This reveals the superiority of our SSD strategy in long response generation. (iii) A higher

![](_page_8_Figure_7.jpeg)

Fig. 6: **Visualized attention map comparisons** of TwigVLM, TwigVLM++ and two typical token pruning methods. The visualized attention maps show that our methods identify accurate visual tokens to the prompt and predict the right answer, while both counterparts fail to do that. More examples are provided in the supplementary.

pruning ratio (i.e., a smaller  $\overline{R}$ ) leads to more speedup on TextVQA, but has little effect on MM-Vet, which has been observed and explained in §3.3. (iv) TwigVLM++ further boosts speed across all configurations by replacing sequence-based SSD with tree-based SSD. The improvement is particularly pronounced on MM-Vet with long responses, where TwigVLM++ reaches ∼197% RelSpd compared to TwigVLM's  $\sim$ 154%, an over 40% additional speedup. On TextVQA with short responses, TwigVLM++ also achieves consistent gains ( $137\% \sim 139\%$  vs.  $123\% \sim 128\%$ ), confirming the effectiveness of tree-based SSD for both short and long response scenarios. Besides, the last column of TABLE 3 reports the RelSpd of different methods on Qwen2.5-VL-7B, where TwigVLM and TwigVLM++ consistently achieve substantially higher speedup than all counterparts, demonstrating the generalization of our acceleration strategies across different base VLMs.

## 6.4 Ablation Studies for TwigVLM

To validate the effectiveness of TwigVLM's key components, we conduct ablation experiments using the default setting (T=3, K=2, K<sub>f</sub>=24,  $\bar{R}$ =64). Taking LLaVA-1.5-7B as the reference model, the relative accuracy (RelAcc) is evaluated on the six benchmarks mentioned in TABLE 1 and the relative speed (RelSpd) is evaluated on MM-Vet. Results in TABLE 4 are discussed in detail below.

| Method    |             |             | Image Benchmarks |                             |                                               |                     |             | Video Benchmarks |                    |             | PolAcc      | RelSpd      |              |               |
|-----------|-------------|-------------|------------------|-----------------------------|-----------------------------------------------|---------------------|-------------|------------------|--------------------|-------------|-------------|-------------|--------------|---------------|
| Method    | GQA         | MME         | MMB              | $\mathbf{SQA}^{\mathrm{I}}$ | $\mathbf{V}\mathbf{Q}\mathbf{A}^{\mathrm{T}}$ | $\mathbf{VQA}^{V2}$ | MMStar      | OCRBench         | Blink              | VideoMME    | EgoSchema   | MVB ench    | KeiAcc       | Keispu        |
|           |             |             |                  |                             |                                               | Ирр                 | er Bound,   | 100% Tokens      | (100%)             |             |             |             |              |               |
| Q2.5VL-7B | 60.7        | 2347        | 82.7             | 75.3                        | 83.2                                          | 77.9                | 63.3        | 827              | 56.4               | 63.4        | 58.8        | 69.6        | 100.0%       | 100.0%        |
|           |             |             |                  |                             |                                               | Retain .            | Averaged 3  | 3.3% Tokens      | (↓ 66.7            | 7%)         |             |             |              |               |
| FastV     | 57.2        | 2299        | 80.9             | 75.5                        | 81.5                                          | 74.2                | 58.9        | 682              | 53.2               | 61.7        | 57.1        | 66.2        | 95.2%        | 101.7%        |
| VisionZip | 60.0        | 2387        | 83.4             | <u>77.1</u>                 | 76.3                                          | 77.9                | <u>61.1</u> | 706              | <u>53.9</u>        | 61.5        | <u>58.3</u> | <u>68.1</u> | 97.2%        | 103.4%        |
| TwigVLM   | 60.4        | 2338        | 79.4             | 77.9                        | <u>82.6</u>                                   | <u>78.4</u>         | 60.0        | <u>744</u>       | 53.3               | <u>62.4</u> | 57.0        | 68.0        | <u>97.5%</u> | <u>147.7%</u> |
| TwigVLM++ | <u>60.1</u> | <u>2373</u> | <u>82.7</u>      | 77.0                        | 83.2                                          | 78.9                | 62.2        | 825              | 54.7               | 63.6        | 58.3        | 69.1        | 99.7%        | 187.1%        |
|           |             |             |                  |                             |                                               | Retain .            | Averaged 2  | 2.2% Tokens      | (↓ 77.8            | 3%)         |             |             |              |               |
| FastV     | 53.5        | 2246        | 78.6             | 75.3                        | 79.2                                          | 70.6                | 55.1        | 589              | 51.8               | 60.0        | 56.2        | 61.3        | 91.1%        | 103.1%        |
| VisionZip | <u>59.2</u> | <u>2318</u> | 82.6             | 77.0                        | 71.6                                          | 76.2                | <u>58.9</u> | 628              | <u>53.2</u>        | 61.2        | 58.2        | <u>66.8</u> | 94.7%        | 105.8%        |
| TwigVLM   | 59.9        | 2238        | 77.4             | <b>78.1</b>                 | <u>81.4</u>                                   | <u>78.4</u>         | 57.2        | <u>685</u>       | 51.5               | <u>61.4</u> | 55.1        | 66.4        | 95.0%        | <u>151.2%</u> |
| TwigVLM++ | 59.8        | 2346        | <u>82.1</u>      | <u>77.1</u>                 | 82.5                                          | 78.4                | 60.5        | 813              | 54.0               | 62.4        | <u>58.0</u> | 68.6        | 98.7%        | 192.4%        |
|           |             |             |                  |                             |                                               | Retain .            | Averaged 1  | 1.1% Tokens      | $(\downarrow 88.9$ | %)          |             |             |              |               |
| FastV     | 45.1        | 1859        | 61.5             | 72.8                        | 62.9                                          | 58.2                | 41.9        | 309              | 44.3               | 55.3        | 52.4        | 59.3        | 76.7%        | 104.3%        |
| VisionZip | 56.6        | <u>2153</u> | 79.2             | <u>76.1</u>                 | 58.7                                          | 70.5                | <u>54.9</u> | 486              | 49.6               | <u>59.5</u> | 57.2        | 65.6        | 88.4%        | 107.1%        |
| TwigVLM   | <u>57.6</u> | 2020        | 67.0             | 74.9                        | <u>73.0</u>                                   | <b>75.6</b>         | 48.2        | <u>518</u>       | 48.6               | 56.6        | 49.3        | 61.0        | 86.0%        | <u>152.3%</u> |
| TwigVLM++ | 58.3        | 2240        | <u>78.7</u>      | 76.1                        | 79.5                                          | <u>75.2</u>         | 56.6        | 772              | <u>49.5</u>        | 60.1        | <u>55.0</u> | <u>65.5</u> | 94.4%        | 193.2%        |

TABLE 3: Performance comparisons on **Qwen2.5-VL-7B** across image and video benchmarks. The vertical line separates image benchmarks (left) from video benchmarks (right). The RelAcc is evaluated following the same protocol as in TABLE 1, and the RelSpd is evaluated on MM-Vet as in Fig. 7.

![](_page_9_Figure_4.jpeg)

Fig. 7: **Generation speed comparisons** on two benchmarks: (a) TextVQA with short responses and (b) MM-Vet with long responses.  $\bar{S}$  denotes the average number of generated tokens on the whole benchmark. The *RelSpd* of each bar is highlighted in red.

**Visual token selection.** The performance of visual token pruning relies on the chosen attention map for token selection. TABLE 4a compares three TwigVLM variants with token pruning guided by different attention sources: (a) the K-th layer of the VLM backbone, (b) the (K+T)-th layer of the VLM backbone, and (c) the last twig layer which has the same depth as (b). From the results, we can see that using the attention map of the K-th backbone layer yields the lowest accuracy (82.3%), which verifies the conclusion in §3.3 that the semantics in early-layer attention is ambiguous. With the same attention depth D=(K+T), selecting tokens by the attention from the backbone (86.2%) is inferior to that from the twig layer (96.0%), indicating that the quality of the attention map is also impacted by the "distance" between the layer it is on and the prediction head.

**Acceleration strategies.** The inference acceleration of TwigVLM is achieved by the synergy between the twig-

guided token pruning (TTP) and self-speculative decoding (SSD) strategies during prefilling and decoding, respectively. In TABLE 4b, we compare the three acceleration strategies as follows: (a) visual token pruning by FastV [17], (b) SSD only without token pruning, and (c) the standard TwigVLM with SSD and TTP. While FastV achieves modest speedup (104.3%), SDD delivers a remarkable improvement (146.7%) as the decoding time is longer than prefilling. Finally, the TTP and SSD strategies are complementary and their synergy achieves the highest speedup (153.6%).

**Twig block initialization.** The initialization strategy for twig layers has a prominent impact on TwigVLM's performance. From TABLE 4c, we can see that: random initialization in (a) performs poorly due to the lack of knowledge inherited from the base VLM. Moreover, the strategy in (b) that uses the last T VLM layers to initialize the twig layers results in distinct improvements. Finally, initializing from layers K to K+T achieves the best results in both accuracy and speed. The speed improvements are contributed by the higher token acceptance rate<sup>4</sup>. More detailed analyses about token acceptance rate are provided in the supplementary.

Number of twig layers. TABLE 4d varies the number of twig layers T from 1 to 4 to investigate its effect on TwigVLM. We observe that a single twig layer (T=1) leads to the highest speed but the lowest accuracy. Increasing T to 2 or 3 improves accuracy with negligible speed degradation. However, further increasing T to 4 leads to a distinct performance drop in terms of both accuracy and speed. These observations can be explained by two factors: i) improving retained token selection (via more twig layers) obtains diminishing accuracy returns, which finally saturates at T=3, and ii) deeper twig block involves linearly increasing computation while the token acceptance ratio in SSD achieves saturation, leading to a decrease in generation speed. Thus, T=3 offers an optimal choice for the number of twig layers.

4. The token acceptance rate measures how often verification accepts each of the draft tokens [22].

| attention source, depth ${\cal D}$ | RelAcc (%) |
|------------------------------------|------------|
| (a) VLM backbone, K                | 82.3       |
| (b) VLM backbone, $K+T$            | 86.2       |
| (c) twig layer, $K+T$              | 96.0       |

(a) **Visual token selection.** The attention map from the last twig layer results in the highest accuracy as it is close to the prediction head.

| T | RelAcc (%) | RelSpd (%) |
|---|------------|------------|
| 1 | 93.9       | 154.1      |
| 2 | 95.2       | 152.6      |
| 3 | 96.0       | 153.6      |
| 4 | 95.8       | 145 4      |

(d) **Number of twig layers.** T=3 results in the highest accuracy with negligible speed degradation compared to T=1.

| acceleration strategy          | RelSpd (%) |
|--------------------------------|------------|
| (a) token purning (FastV [17]) | 104.3      |
| (b) SSD                        | 146.7      |
| (c) TTP & SSD                  | 153.6      |

(b) Acceleration strategies. The TTP and SSD strategies are complementary to each other, and their synergy achieves the highest speedup.

| K | R  | RelAcc (%) | RelSpd (%) |
|---|----|------------|------------|
| 0 | 85 | 93.6       | 137.5      |
| 1 | 64 | 95.5       | 136.5      |
| 2 | 41 | 96.0       | 153.6      |
| 3 | 15 | 85.5       | 145.2      |

(e) **Pruning position.** When  $\bar{R}$  is fixed to 64, K=2 achieves the optimal balance between accuracy and speed.

| initialization strategy           | RelAcc (%) | RelSpd (%) |
|-----------------------------------|------------|------------|
| (a) random init.                  | 87.2       | 120.4      |
| (b) VLM layers[ $L$ - $T$ : $L$ ] | 90.4       | 131.4      |
| (c) VLM layers $[K:K+T]$          | 96.0       | 153.6      |

(c) **Twig block initialization.** The twig layers initialized from the K-th to (K+T)-th layers in the base VLM achieve the best accuracy and speed.

| FinalWipe | $K_{\mathrm{f}}$ | R  | RelAcc (%) | RelSpd (%) |
|-----------|------------------|----|------------|------------|
| ×         | 32               | 30 | 93.1       | 154.6      |
| ✓         | 20               | 50 | 95.8       | 151.3      |
| ✓         | 24               | 41 | 96.0       | 153.6      |
| ✓         | 28               | 37 | 95.1       | 154.1      |

(f) **FinalWipe.** The FinalWipe strategy facilitates model accuracy and  $K_f$ =24 is the optimal choice to balance accuracy and speed.

TABLE 4: **Ablation experiments for TwigVLM**. Taking LLaVA-1.5-7B as the base VLM, the relative accuracy (RelAcc) is evaluated on the six benchmarks mentioned in TABLE 1 and the relative speed (RelSpd) is evaluated on MM-Vet. In each table, the best result is **bolded** and the default setting (T=3, K=2, K<sub>f</sub>=24,  $\bar{R}$ =64) is marked in green .

| Head(s)         | Loss                                                                                   | Stage-1 | Stage-2 |
|-----------------|----------------------------------------------------------------------------------------|---------|---------|
| D-Head          | $\mathcal{L}_{\text{NTP}}$                                                             | 96.0%   | -       |
| D-Head + P-Head | $\mathcal{L}_{NTP} + \mathcal{L}_{AttnKL}$                                             | 95.0%   | 97.2%   |
| D-Head + P-Head | $\mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{AttnKL}} + \mathcal{L}_{\text{PredKL}}$ | 96.4%   | 97.7%   |

(a) Different training settings in stage-1. Scores refer to the RelAcc evaluated with  $\bar{R}$ =64.

| RL setting            | Rel/ | Acc(%) | $@\bar{R}$ | # training | RelAcc(%) @ $\bar{R}$ |      |      |  |
|-----------------------|------|--------|------------|------------|-----------------------|------|------|--|
| for $\bar{R}$         | 192  | 128    | 64         | samples    | 192                   | 128  | 64   |  |
| static $\bar{R}$ =192 | 99.4 | 99.3   | 97.1       | 10k        | 99.1                  | 98.8 | 97.2 |  |
| static $\bar{R}$ =128 | 99.5 | 99.2   | 97.4       | 20k        | 99.4                  | 99.1 | 97.6 |  |
| static $\bar{R}$ =64  | 99.1 | 99.0   | 98.0       | 50k        | 99.6                  | 99.2 | 97.7 |  |
| dynamic               | 99.6 | 99.2   | 97.7       | 100k       | 99.4                  | 99.0 | 97.8 |  |

(b) Different pruning-ratio (c) Different sizes of the trainstrategy in stage-2.

TABLE 5: **Ablation experiments for TwigVLM++**. The best result is **bolded** and the default setting is marked in dark green .

**Pruning position.** To achieve an optimal number of retained tokens when  $\bar{R}$  is fixed to 64, we use Eq.(6) to explore different combinations of K and R. Increasing K leads to a smaller R, which means a trade-off between the number of visual tokens in early layers and the number of visual tokens in subsequent layers. As shown in TABLE 4e, both the accuracy and speed achieve the best results at K=2, which is chosen in our default setting.

**FinalWipe.** The FinalWipe works by removing all visual tokens after the  $K_{\rm f}$ -th layer. TABLE 4f ablates different TwigVLM variants with/without this strategy. When Final-Wipe is not introduced, the accuracy is relatively low due to the insufficiency of retained tokens. Introducing FinalWipe facilitates accuracy with limited speed drop, which can be explained by the fact that the visual tokens become less important in late layers.  $K_{\rm f}$ =24 achieves a good balance between accuracy and speed, thus is used as the default.

#### 6.5 Ablation Studies for TwigVLM++

We ablate the key design choices in TwigVLM++ training on LLaVA-1.5-7B. Results are reported in TABLE 5.

Architecture and loss in stage-1. TABLE 5a compares different head configurations and loss combinations. With the single D-Head baseline achieving 96.0% RelAcc after stage-1, introducing the P-Head with only  $\mathcal{L}_{\text{AtmKL}}$  actually degrades stage-1 performance to 95.0%, suggesting that the multi-head architecture splits the training capacity and leads to insufficient optimization of each head. Adding  $\mathcal{L}_{\text{PredKL}}$  effectively compensates for this issue, raising stage-1 accuracy to 96.4% and yielding the best stage-2 result of 97.7%. This confirms that strong-to-weak distillation provides complementary supervision that benefits both the D-Head and P-Head training.

Pruning-ratio strategy in stage-2. TABLE 5b compares static and dynamic pruning-ratio strategies for stage-2 RL. Training with a static  $\bar{R}$ =64 achieves the highest accuracy at  $\bar{R}$ =64 (98.0%) but degrades performance at larger retention settings ( $\bar{R}$ =192: 99.1%,  $\bar{R}$ =128: 99.0%), as the policy overfits to aggressive pruning. The reverse trend holds for static  $\bar{R}$ =192. In contrast, the dynamic strategy achieves the best result at  $\bar{R}$ =192 (99.6%) while maintaining competitive accuracy across other ratios, delivering the most balanced performance overall.

**Training data scale in stage-2.** TABLE 5c varies the number of RL training samples drawn from the LLaVA-665K SFT dataset. Performance improves notably from 10k to 20k samples, after which the gains largely saturate—the gap between 20k and 50k is marginal (e.g., 97.6% vs. 97.7% at  $\bar{R}$ =64), and scaling to 100k yields no further improvement. This indicates that only 50k samples suffice for the RL training to converge, demonstrating the data efficiency of the proposed stage-2 optimization.

#### 7 CONCLUSION

In this paper, we present TwigVLM—a conceptually simple yet effective approach that accelerates VLM inference by growing a lightweight twig block upon an early layer of the base VLM, enabling both twig-guided token pruning (TTP) for prefilling acceleration and self-speculative decoding (SSD) for decoding acceleration. To further improve the pruning quality, we extend TwigVLM to TwigVLM++

by introducing a multi-head twig architecture that decouples pruning from next-token prediction, and a two-stage training paradigm that combines distillation learning with reinforcement learning to directly optimize the pruning decisions. Furthermore, the tree-based SSD is adopted to achieve higher decoding throughput. Extensive ablations, comparative experiments, and comprehensive analyses on a wide range of image and video benchmarks demonstrate the superiority of TwigVLM and TwigVLM++ over existing state-of-the-art methods. We hope our work may serve as a solid baseline to inspire future research on visual token reduction and VLM acceleration.

