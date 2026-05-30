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

# **APPENDIX A MORE IMPLEMENTATION DETAILS**

**TwigVLM/TwigVLM++ training.** As described in the main text, the twig block is trained by finetuning the shallow VLM Ms. Specifically, M<sup>s</sup> is initialized with the weights of the first K+T layers and the prediction head of the corresponding base VLM Mb. During finetuning, only the last T layers and the prediction head—collectively termed the twig block—are updated, while the first K layers remain frozen. This process follows the same training manner to train the base VLM Mb. Theoretically, any suitable multimodal instruction tuning dataset can be employed to finetune Ms.

The base VLMs evaluated in the main text experiments all leverage the open-source datasets to train their twig blocks. Specifically, we use the LLaVA-665K dataset [23] to train TwigVLM/TwigVLM++ for LLaVA-1.5-7B and LLaVA-NeXT-7B models, and a dataset of 5M single-image samples from the MAmmoTH-VL-10M dataset [67] for Qwen2.5-VL-7B. The optimization hyper-parameters used for training TwigVLM/TwigVLM++ are detailed in TABLE 6. All training is performed on a server equipped with 8 NVIDIA A100 GPUs. Under these conditions, the training of TwigVLM is highly efficient, requiring only approximately 10% of the time needed to train the corresponding base VLM, *e.g.*, training the twig block for the LLaVA-1.5-7B model takes about 10 GPU hours, while the training of the original LLaVA-1.5- 7B takes about 100 GPU hours. TwigVLM++ training also needs only 20% of that time.

**Twig-guided token pruning (TTP).** During inference, TwigVLM leverages the TTP strategy to perform token pruning over the base VLM: (i) at the K-th layer, selecting R key visual tokens (output by the K-th layer) and discarding the rest tokens guided by the attention map from the last twig layer, and (ii) applying the FinalWipe strategy to further remove all the visual tokens after the Kf-th layer. Therefore, we adjust the value of R to satisfy different pruning ratios calculated by the average number of retained visual tokens R¯. TABLE 7 shows the default pruning settings for TwigVLM under different pruning ratios.

**Self-speculative decoding (SSD).** For efficient generation of long responses, TwigVLM applies the SSD strategy by using the M<sup>s</sup> as the *draft* model and M<sup>b</sup> as the *target* model. Specifically, in each SSD iteration, the draft model efficiently predicts δ = 5 subsequent draft tokens in an autoregressive manner. To further improve efficiency, this draft generation process is equipped with an early-exit mechanism that allows the draft model to stop generation if the probability

| config                     | setting                              |  |  |
|----------------------------|--------------------------------------|--|--|
|                            | TwigVLM & TwigVLM++ stage-1 (shared) |  |  |
| optimizer                  | AdamW                                |  |  |
| weight decay               | 0.                                   |  |  |
| optimizer momentum         | β1, β2=0.9, 0.98                     |  |  |
| batch size                 | 128                                  |  |  |
| learning rate schedule     | cosine decay                         |  |  |
| peak learning rate         | 1e-4                                 |  |  |
| warm-up strategy           | linearly warm-up                     |  |  |
| warm-up ratio              | 0.03                                 |  |  |
| training samples           | 665K                                 |  |  |
| training epochs            | 1                                    |  |  |
|                            | TwigVLM++ stage-1 (additional)       |  |  |
| PredKL coefficient α       | 0.1                                  |  |  |
| AttnKL coefficient γ       | 1.0                                  |  |  |
|                            | TwigVLM++ stage-2                    |  |  |
| optimizer                  | AdamW                                |  |  |
| peak learning rate         | 2e-5                                 |  |  |
| batch size                 | 128                                  |  |  |
| group size G               | 32                                   |  |  |
| candidate set R            | {64,85,107,128,149,171,192}          |  |  |
| annealing params (βmax, p) | (8.0, 2.0)                           |  |  |
| training samples           | 50K                                  |  |  |
| training epochs            | 1                                    |  |  |

TABLE 6: **Training settings.** The first section lists hyperparameters shared by TwigVLM and the training stage-1 of TwigVLM++. The second and third sections list additional hyper-parameters specific to TwigVLM++.

| R¯  | pruning ratio | K | R   | Kf |
|-----|---------------|---|-----|----|
| 192 | 66.7%         | 2 | 227 | 24 |
| 128 | 77.8%         | 2 | 134 | 24 |
| 64  | 88.9%         | 2 | 41  | 24 |

TABLE 7: **Pruning settings.** These hyper-parameters correspond to the default TTP settings of different pruning ratios.

of the current predicted token falls below a predefined threshold θ = 0.6. The target model then verifies these generated draft tokens in parallel, accepts those matching the target model's predictions, and then predicts a next token by itself. The iteration repeats until the <EOS> token is generated. Note that the TTP and SSD strategies can be seamlessly integrated, as detailed in Algorithm 1.

**Tree-based self-speculative decoding.** TwigVLM++ replaces the sequential SSD with a tree-based variant to increase the number of accepted tokens per verification step. The token tree T is rooted at the last accepted token and constructed level by level, governed by three hyperparameters: an expansion width E, a selection width K, and a tree depth D. Let T<sup>l</sup> denote the set of nodes at level l. At the first level, M<sup>s</sup> computes the draft distribution conditioned on the current prefix and takes the top-E tokens as children of the root, giving |T1|=E. For each subsequent level l > 1, the top-K nodes from Tl−<sup>1</sup> (ranked by prediction probability) are selected for expansion; each selected node u is fed into M<sup>s</sup> together with its root-to-u prefix to produce E children, yielding |T<sup>l</sup> |=K·E. To bound the verification cost, the completed tree is pruned to retain at most Nmax candidate

## **Algorithm 1** Pseudocode of TwigVLM's inference process

```
# bVLM: the base VLM model, i.e., M_b
# twig: the twig block
# K: Number of shared low layers
# K_f: The position to apply FinalWipe
# R: Number of retained visual tokens when pruning
# delta: Maximum draft token length
# theta: Confidence threshold to stop draft
def sVLM_forward(tokens):
   X_k = bVLM.forward_low_layers(tokens, k=K)
   prob, Attn_last = twig.forward(X_k)
   a_i = argmax(prob)
   return X_k, prob, Attn_last, a_i
def TwigVLM_inference(img, ques):
   draft_toks = [] # temporary buffer for draft tokens
   final_resp = [] # buffer for final response
   # Prefilling stage of sVLM
   X_k, _, Attn, a_i = sVLM_forward((img, ques))
   draft_toks.append(a_i)
   # Prune visual tokens in X_k using Eq. (5)
   # X_k_b means shared token latents for bVLM
   X_k_b = pruning(X_k, Attn, r=R)
   # The loop of self speculative decoding
   while EOS_TOKEN not in final_resp:
       X_k, prob, _, a_i = sVLM_forward(a_i)
       draft_toks.append(a_i)
       X_k_b = concat(X_k_b, X_k, axis=1)
       # the condition to stop draft and verify
       if len(draft_toks) >= delta or prob < theta:
           # removing all visual tokens after layer K_f
           tgt_probs = bVLM.forward_high_layers(
              X_k_b, k=K, fianl_wipe=K_f)
           # verification
           right_toks = [a for a, p in zip(draft_toks, tgt_probs[:-1])
                 if argmax(p) == a]
           right_toks.append(argmax(tgt_probs[-1]))
           final_resp.extend(right_tokens)
           # reset temporary variables
           draft_toks = []
           X_k_b = None
           a_i = final_resp[-1]
   return final_resp
```

nodes by preferentially keeping the highest-confidence leaf nodes and their ancestors. In our default setting, we use E=10, K=10, D=4, and Nmax=60. For verification, M<sup>b</sup> processes the pruned tree in a single forward pass via *tree attention* [65], using a *topology-aware causal mask* so that each node attends only to its ancestors. The target model traverses the tree from the root, accepting a child at each level whose token matches the target model's prediction, and stops at the first level where no child matches; a bonus token predicted by M<sup>b</sup> is then appended. The full procedure is detailed in Algorithm 2.

## **APPENDIX B MORE EXPERIMENTAL RESULTS**

#### **B.1 More performance comparisons**

**Comparisons on more benchmarks.** Taking LLaVA-1.5- 7B as the base VLM, TABLE 9 compares the accuracies among TwigVLM, TwigVLM++ and other visual token pruning methods on *nine* VLM benchmarks under three different pruning ratios. TwigVLM and TwigVLM++ consistently outperform or match their counterparts on all benchmarks and pruning ratios, achieving the best overall RelAcc. In particular, TwigVLM and TwigVLM++ even surpass the upper bound given by the base VLM in RelAcc (100.3%&100.4%) with a 66.7% pruning ratio, demonstrating

## **Algorithm 2** Pseudocode of TwigVLM++'s inference process

```
# bVLM: the base VLM model, i.e., M_b
# twig: the twig block
# K: Number of shared low layers
# K_f: The position to apply FinalWipe
# R: Number of retained visual tokens when pruning
# delta: Number of tree draft iterations
# top_k: Branch factor for tree-based drafting
# max_tokens: Maximum number of candidate tokens in tree
def sVLM_forward(tokens, tree_mask):
   X_k = bVLM.forward_low_layers(tokens, k=K)
   prob, X_t, qk = twig.forward(X_k, tree_mask)
   return X_k, X_t, prob, qk
def TwigVLM++_inference(img, ques):
   final_resp = [] # buffer for final response
   # Prefilling stage of sVLM
   X_k, X_t, _, qk = sVLM_forward((img, ques), None)
   # Prune visual tokens using P-Head
   Attn_phead = P_Head(X_t, qk, img_tags)
   X_k_b = pruning(X_k, Attn_phead, r=R)
   # The loop of tree-based speculative decoding
   while EOS_TOKEN not in final_resp:
       # Tree-based draft generation
       draft_toks = []
       for i in range(delta):
           X_k, prob, _, _ = sVLM_forward(a_i, tree_mask)
           # Top-k sampling for tree expansion
           topk_toks = topk(prob, k=top_k)
           draft_toks.append(topk_toks)
           a_i = topk_toks
           X_k_b = concat(X_k_b, X_k, axis=1)
       # Build tree structure from draft tokens
       tree_cands, tree_pos_ids, tree_mask = \
           build_tree_candidates(draft_toks, top_k, max_tokens)
       # Verify with tree-structured forward
       tgt_probs = bVLM.forward_high_layers(
           X_k_b, tree_cands, tree_pos_ids, tree_mask,
           k=K, final_wipe=K_f)
       # Select best candidate path
       accept_len = verify_tree(tree_cands, tgt_probs)
       best_path = select_best_path(tree_cands, accept_len)
       # Accept tokens and add bonus token
       accept_toks = best_path[:accept_len]
       accept_toks.append(argmax(tgt_probs[accept_len]))
       final_resp.extend(accept_toks)
       # Reset for next iteration
       a_i = final_resp[-1]
       X_k_b = None
   return final_resp
```

their spectacular effectiveness and robustness in accelerating VLMs to deal with various tasks.

**Comparisons on a larger base VLM.** To further demonstrate the generalization ability and superiority of our TwigVLM and TwigVLM++, we present additional experimental results on a larger VLM, LLaVA-1.5-13B, as shown in TABLE 8. TwigVLM and TwigVLM++ consistently achieve the best overall RelAcc compared to all the counterparts, with their superiority being more significant as the increase of pruning ratios. These results verify the scalability and generalization ability of our TwigVLM and TwigVLM++ in accelerating large VLMs.

## **B.2 More ablation studies on TwigVLM**

**Token acceptance rate in SSD.** In the context of speculative decoding methods [22], [34], [36], the token acceptance rate (*abbr.* TokAR) serves as a critical metric for assessing the efficacy of these approaches. TokAR is defined as the proportion of the draft tokens generated by the draft model

| Method                               |      |      |      |      |      | GQA MMB MME VQAT SQAI VQAV2 RelAcc |       |  |
|--------------------------------------|------|------|------|------|------|------------------------------------|-------|--|
| Upper Bound, 576 Tokens (100%)       |      |      |      |      |      |                                    |       |  |
| LLaVA-1.5-13B                        | 63.2 | 67.7 | 1818 | 61.3 | 72.8 | 80.0                               | 100%  |  |
| Retain Averaged 192 Tokens (↓ 66.7%) |      |      |      |      |      |                                    |       |  |
| FastV                                | 60.3 | 67.4 | 1807 | 60.4 | 74.0 | 77.7                               | 98.6% |  |
| VisionZip                            | 59.1 | 66.9 | 1754 | 59.5 | 73.5 | 78.1                               | 97.4% |  |
| VisionZip‡                           | 61.6 | 67.1 | 1790 | 59.9 | 72.7 | 78.6                               | 98.5% |  |
| TwigVLM                              | 62.5 | 68.6 | 1840 | 60.4 | 73.1 | 79.4                               | 99.9% |  |
| TwigVLM++                            | 62.6 | 68.2 | 1829 | 60.4 | 73.2 | 79.4                               | 99.8% |  |
| Retain Averaged 128 Tokens (↓ 77.8%) |      |      |      |      |      |                                    |       |  |
| FastV                                | 57.5 | 65.9 | 1758 | 58   | 73.8 | 74.3                               | 95.7% |  |
| VisionZip                            | 57.9 | 66.7 | 1743 | 58.7 | 74.0 | 76.8                               | 96.6% |  |
| VisionZip‡                           | 60.1 | 67.6 | 1736 | 59.2 | 73.0 | 77.6                               | 97.4% |  |
| TwigVLM                              | 61.2 | 66.9 | 1811 | 60.2 | 73.4 | 79.1                               | 98.9% |  |
| TwigVLM++                            | 62.3 | 67.7 | 1849 | 60.3 | 72.9 | 79.1                               | 99.6% |  |
| Retain Averaged 64 Tokens (↓ 88.9%)  |      |      |      |      |      |                                    |       |  |
| FastV                                | 50.1 | 55.9 | 1408 | 52.2 | 73.2 | 61.1                               | 83.6% |  |
| VisionZip                            | 56.2 | 64.9 | 1676 | 57.4 | 74.4 | 73.7                               | 94.2% |  |
| VisionZip‡                           | 58.1 | 65.6 | 1671 | 58.5 | 72.3 | 75.2                               | 94.9% |  |
| TwigVLM                              | 60.0 | 67.4 | 1765 | 58.4 | 72.4 | 77.0                               | 97.1% |  |
| TwigVLM++                            | 61   | 65.7 | 1793 | 58.5 | 72.7 | 78.0                               | 97.5% |  |

TABLE 8: Performance comparisons of our TwigVLM and TwigVLM++ with other token pruning methods on the **LLaVA-1.5-13B** model.

that are subsequently accepted by the target model. In TwigVLM, TokAR plays a key role, which is influenced by the effectiveness of the twig block and has a significant impact on model's generation speed.

To analyze how TokAR is influenced by the design choices in TwigVLM, we evaluate this metric on several representative variants from the ablation studies presented in the main text. From the results shown in TABLE 10, we have the following findings: (i) A more effective draft model can be trained by only modifying the initialization strategy without altering the architecture. The variant (c) achieves the highest TokAR (57.4%) and thus the highest generation speedup. (ii) Increasing the number of twig layers T introduces more computational costs while improving TokAR at the same time. As a result, the RelSpd exhibits only a modest decline when T increases from 1 to 3. However, it drops distinctly at T=4, which indicates that TokAR begins to saturate. These findings suggest that TwigVLM achieves higher speedup by striking an optimal balance between TokAR and computation costs of the draft model.

**Data efficiency.** To demonstrate the data efficiency of TwigVLM, we train multiple models using different proportions (i.e., 25%, 50%, 75%, and 100%) of each model's respective training dataset for LLaVA-1.5-7B and Qwen2.5- VL-7B. As shown in Fig. 8, both models exhibit a general upward trend in accuracy and speed as the amount of training data increases. Remarkably, however, even when trained on only 50% of their respective datasets, TwigVLM models already achieve competitive, and in some cases comparable, performance to models trained on the full dataset. Moreover, TwigVLM requires only 10% of the training cost of the corresponding base VLM (see A). Recall the results in TABLE 5c of the main text, we can sum up that it is highly efficient and feasible to apply our TwigVLM and

![](_page_13_Figure_7.jpeg)

Fig. 8: Performance comparisons of TwigVLM models trained with **different proportions of the training dataset**. Specifically, we use LLaVA-665K to train TwigVLM models for LLaVA-1.5-7B and use MAmmoTH-VL-10M to train TwigVLM models for Qwen2.5-VL-7B. Even with only 50% of the respective training data, TwigVLM is able to maintain competitive accuracy and speed.

TwigVLM++ in industrial scenarios.

**Memory footprint analysis.** We measure the inference VRAM usage of the LLaVA-1.5-7B and LLaVA-Next-7B models in TABLE 11. The introduction of the twig block brings 8% extra VRAM cost for loading model weights. Compared to the base VLM, the overall inference VRAM cost of TwigVLM is comparable or slightly reduced due to the substantial reduction of visual tokens.

## **APPENDIX C MORE VISUALIZED RESULTS**

In this section, we provide more visualized results to validate the effectiveness of TwigVLM's two key components: the twig-guided visual token pruning (TTP) and selfspeculative decoding (SSD). We use LLaVA-1.5-7B as the base VLM in the following experiments.

**Visual token pruning.** To better understand the effectiveness of the proposed TTP strategy, we compare TwigVLM with two representative token pruning methods, namely FastV [17] and VisionZip [20], by visualizing their attention map for token selection and providing the corresponding answer predictions. We provide 16 examples from the GQA and TextVQA benchmarks. As illustrated in Fig. 9, TwigVLM demonstrates superior ability to comprehend the semantics in both the textual prompt and image, and accurately identify task-specific image patches (*i.e.*, visual tokens), thereby activating more informative visual tokens for token pruning. In contrast, FastV and VisionZip often fail to capture the fine-grained visual details, leading to

| Method          | GQA                            | MMB  | MME  | VQAT | SQAI | VQAV2                                | POPE | MMMU | MM-Vet | RelAcc |
|-----------------|--------------------------------|------|------|------|------|--------------------------------------|------|------|--------|--------|
|                 | Upper Bound, 576 Tokens (100%) |      |      |      |      |                                      |      |      |        |        |
| LLaVA-1.5-7B    | 61.9                           | 64.7 | 1862 | 58.2 | 69.5 | 78.5                                 | 85.9 | 36.3 | 31.1   | 100%   |
|                 |                                |      |      |      |      | Retain Averaged 192 Tokens (↓ 66.7%) |      |      |        |        |
| FastV [17]      | 56.5                           | 63.7 | 1786 | 57.3 | 69.5 | 74.6                                 | 79.2 | 35.7 | 28.1   | 95.6%  |
| SparseVLM [18]  | 57.6                           | 62.5 | 1721 | 56.1 | 69.1 | 75.6                                 | 83.6 | 33.8 | 31.5   | 96.2%  |
| PDrop [43]      | 57.3                           | 63.3 | 1797 | 56.5 | 69.2 | 75.1                                 | 82.3 | -    | -      | 96.4%  |
| MustDrop [19]   | 58.2                           | 62.3 | 1787 | 56.5 | 69.2 | 76.0                                 | 82.6 | -    | -      | 96.6%  |
| VisionZip [20]  | 59.3                           | 63.0 | 1783 | 57.3 | 68.9 | 76.8                                 | 85.3 | 36.6 | 31.7   | 98.5%  |
| VisionZip‡ [20] | 60.1                           | 63.4 | 1834 | 57.8 | 68.2 | 77.4                                 | 84.9 | 36.2 | 32.6   | 99.2%  |
| TwigVLM         | 61.2                           | 64.0 | 1848 | 58.0 | 68.8 | 78.1                                 | 87.2 | 36.6 | 32.8   | 100.3% |
| TwigVLM++       | 61.2                           | 64.3 | 1868 | 58.0 | 69.2 | 78.2                                 | 86.9 | 36.4 | 32.6   | 100.4% |
|                 |                                |      |      |      |      | Retain Averaged 128 Tokens (↓ 77.8%) |      |      |        |        |
| FastV           | 53.0                           | 61.4 | 1646 | 56.0 | 69.5 | 69.2                                 | 73.2 | 36.3 | 28.0   | 92.1%  |
| SparseVLM       | 56.0                           | 60.0 | 1696 | 54.9 | 67.1 | 73.8                                 | 80.5 | 33.8 | 30.0   | 93.6%  |
| PDrop           | 57.1                           | 61.6 | 1761 | 56.6 | 68.4 | 72.9                                 | 82.3 | -    | -      | 95.2%  |
| MustDrop        | 56.9                           | 61.1 | 1745 | 56.3 | 68.5 | 74.6                                 | 78.7 | -    | -      | 94.6%  |
| VisionZip       | 57.6                           | 62.0 | 1762 | 56.8 | 68.9 | 75.6                                 | 83.2 | 37.9 | 32.6   | 98.1%  |
| VisionZip‡      | 58.9                           | 62.6 | 1823 | 57.0 | 68.3 | 76.6                                 | 83.7 | 37.3 | 32.9   | 98.8%  |
| TwigVLM         | 60.6                           | 63.5 | 1818 | 57.8 | 69.5 | 77.9                                 | 86.6 | 36.6 | 30.8   | 99.2%  |
| TwigVLM++       | 60.8                           | 63.7 | 1856 | 58.0 | 69.5 | 77.9                                 | 87.0 | 36.4 | 31.4   | 99.8%  |
|                 |                                |      |      |      |      | Retain Averaged 64 Tokens (↓ 88.9%)  |      |      |        |        |
| FastV           | 44.1                           | 45.9 | 1218 | 50.7 | 70.0 | 52.0                                 | 55.6 | 34.0 | 17.8   | 75.3%  |
| SparseVLM       | 52.7                           | 56.2 | 1505 | 51.8 | 62.2 | 68.2                                 | 75.1 | 32.7 | 23.3   | 85.6%  |
| PDrop           | 47.5                           | 58.8 | 1561 | 50.6 | 69.0 | 69.2                                 | 55.9 | -    | -      | 84.4%  |
| FasterVLM [21]  | 51.5                           | 58.5 | 1573 | 53.1 | 69.6 | 66.8                                 | 67.2 | -    | 27.5   | 87.6%  |
| MustDrop        | 53.1                           | 60.0 | 1612 | 54.2 | 63.4 | 69.3                                 | 68.0 | -    | -      | 88.1%  |
| VisionZip       | 55.1                           | 60.1 | 1690 | 55.5 | 69.0 | 72.4                                 | 77.0 | 36.2 | 31.7   | 94.5%  |
| VisionZip‡      | 57.0                           | 61.5 | 1756 | 56.0 | 68.8 | 74.2                                 | 80.9 | 35.6 | 30.2   | 95.6%  |
| TwigVLM         | 58.8                           | 60.4 | 1760 | 55.8 | 70.0 | 75.6                                 | 82.7 | 35.9 | 29.9   | 96.3%  |
| TwigVLM++       | 59.7                           | 63.2 | 1801 | 56.7 | 69.5 | 76.8                                 | 86.6 | 35.7 | 29.0   | 97.6%  |

TABLE 9: Performance of TwigVLM/TwigVLM++ on **LLaVA-1.5-7B** compared to existing methods under three different pruning ratios. The best result for each benchmark and pruning ratio is **bolded**.

| ablation variant                                  | TokAR (%) | RelSpd (%) |  |  |  |  |
|---------------------------------------------------|-----------|------------|--|--|--|--|
| Twig block initialization (Table 4c in main text) |           |            |  |  |  |  |
| (a) random init.                                  | 37.7      | 120.4      |  |  |  |  |
| (b) VLM layers[L-T:L]                             | 44.1      | 131.4      |  |  |  |  |
| (c) VLM layers[K:K+T]                             | 57.4      | 153.6      |  |  |  |  |
| Number of twig layers (Table 4d in main text)     |           |            |  |  |  |  |
| (d) T = 1                                         | 48.7      | 154.1      |  |  |  |  |
| (e) T = 2                                         | 53.4      | 152.6      |  |  |  |  |
| (f) T = 3                                         | 57.4      | 153.6      |  |  |  |  |
| (g) T = 4                                         | 58.1      | 145.4      |  |  |  |  |

TABLE 10: **Token acceptance rate in SSD**. We evaluate the token acceptance rate (TokAR) of the variants in the ablation experiments of the main text.

suboptimal token selection and incorrect predictions. Notably, even though TwigVLM predicts an incorrect answer, its activated visual tokens according to the attention map are reasonable. This suggests that TwigVLM's occasional failures may not be caused by the visual token pruning, but due to the limitations of the base VLM. These findings verify and explain the effectiveness of the TTP strategy.

| model         | avg. visual<br>tokens (R¯) | model weights<br>VRAM (GB) | inference<br>VRAM (GB) |
|---------------|----------------------------|----------------------------|------------------------|
| LLaVA-1.5-7B  | 576                        | 14.3                       | 15.8                   |
| + TwigVLM     | 64                         | 15.5                       | 16.5                   |
| LLaVA-Next-7B | 2,880                      | 14.3                       | 17.9                   |
| + TwigVLM     | 320                        | 15.5                       | 16.8                   |

TABLE 11: Memory footprint comparisons during inference.

**Self-speculative decoding.** To better understand the decoding behavior of the SSD strategy in TwigVLM, we show 8 examples of generated long responses on MM-Vet. From the results in Fig. 10, we have two key observations: (i) In general, the proportion of accepted tokens (in green) surpasses that of the corrected tokens (in black) by the target model, indicating that TwigVLM achieves significant speedup through its high token acceptance rate. (ii) The majority of *easy* tokens, such as those associated with grammar and punctuation, are readily accepted. In contrast, the *hard* tokens, which often demand complex reasoning, have a high probability of being corrected by the target model. In practice, the proportion of easy tokens is usually larger than the hard ones, which confirms the effectiveness of

![](_page_15_Figure_2.jpeg)

Fig. 9: Visualization of attention maps and predictions for FastV [17], VisionZip [20], and our TwigVLM on the examples chosen from the GQA [60] and TextVQA [61]. For the examples in the last row, we additionally provide the predictions from the LLaVA-1.5-7B.

our SSD strategy in accelerating the decoding stage while maintaining the generation quality.

# APPENDIX D EVALUATION BENCHMARKS

In this section, we provide a brief overview of the benchmarks used in our experiments.

![](_page_16_Picture_2.jpeg)

**Q**: Is that wine in the background?

**A**: No, there is no wine in the background. The image features a pizza and a soda in a box, along with a laptop and a potted plant.

![](_page_16_Picture_5.jpeg)

**Q**: what is the green logo on the car?

**A**: The green logo on the car is the Monster Energy logo.

![](_page_16_Picture_8.jpeg)

**Q**: Introduce the player on the right.

**A**: The player on the right is named Kak·, and he is wearing a jersey with the number 8.

![](_page_16_Picture_11.jpeg)

**Q**: Is there any reflection of zebra in water?

**A**: Yes, there is a reflection of the zebra in the water.

![](_page_16_Picture_14.jpeg)

**Q**: Can you give a short introduction to this painting?

**A**: The painting is a famous artwork by Vincent Van Gogh, featuring a night scene with a sky filled with stars. The scene is set in a village, with a church

 steeple visible in the distance. The painting captures the beauty of the night sky and the tranquility of the village, creating a serene atmosphere. The combination of the stars, the church steeple, and the village buildings make this painting a captivating representation of the artist's vision.

![](_page_16_Picture_18.jpeg)

**Q**: Can you give a short introduction to this person?

**A**: The image features a statue of an old man with a long beard, sitting and holding a staff in his hands. The statue appears to be a

representation of a wise or respected figure, possibly a religious leader or a historical figure. The statue is situated in front of a building, possibly a temple or a museum, and it serves as a symbol of wisdom, knowledge, or spiritual guidance.

![](_page_16_Picture_22.jpeg)

**Q**: Provide a detailed description of the given image.

**A**: The image features a dining table with a white plate filled with a variety of breakfast foods. The plate contains eggs, toast, and a selection of fruits,

including strawberries and blueberries. There are also sausages on the plate, adding to the delicious assortment of breakfast items. A cup of coffee is placed next to the plate, completing the breakfast setup. A fork is positioned on the table, ready for use. The table setting is inviting and perfect for a morning meal.

![](_page_16_Picture_26.jpeg)

**Q**: Can you give a short introduction to this movie?

**A**: The movie "The Dark Knight" is a 2008 superhero film directed by Christopher Nolan. It is the second installment in the Batman film series and stars

Heath Ledger as the Joker, along with Christian Bale as Batman/Bruce Wayne, Michael Caine as Alfred, and Maggie Gyllenhall as Rachel Dawes. The film follows Batman as he battles the Joker, a psychopathic criminal mastermind who seeks to take over Gotham City. The movie is known for its intense action sequences, complex plot, and the iconic performance of Heath Ledger as the Joker.

Fig. 10: Examples of the generated responses using the self-speculative decoding (SSD) on MM-Vet [72], with accepted tokens by the target model being highlighted in green.

**GQA** [60] is a benchmark that focuses on visual scene understanding and reasoning, leveraging scene graphs, questions, and images. It incorporates spatial relationships and object properties, posing challenges for models to perform accurate visual reasoning under complex visual environments.

**MMBench** [68] adopts a hierarchical evaluation approach

with three levels: Level-1 (perception and reasoning), Level-2 (six sub-abilities), and Level-3 (20 specific dimensions). This structured framework allows for a comprehensive evaluation of model performance, making it an effective tool for assessing a wide range of visual understanding capabilities. We denote it as "MMB" in the main text.

**MME** [69] assesses models across 14 subtasks that probe

both perceptual and cognitive skills. Carefully crafted instruction-answer pairs guarantee a fair and comprehensive evaluation of a model's multimodal performance. The final score reported on this benchmark is the summation of both the perception and cognition scores.

**ScienceQA** [70] spans multiple scientific fields, including natural, language, and social sciences, with questions organized into 26 topics, 127 categories, and 379 skills. It evaluates a model's multimodal comprehension, multi-step reasoning, and interpretability, providing a rich testbed for assessing scientific knowledge application in visual contexts. In our experiments, we only evaluate the performance on the samples with images, denoted as "SQA<sup>I</sup> " in the experimental tables.

**VQA-v2** [71] is a large-scale benchmark featuring 265K images of real-world scenes and objects, with each image paired with open-ended questions and 10 human-provided ground truth answers.

**TextVQA** [61] tests a model's ability to process and reason about text embedded within images. By requiring the integration of visual and textual information, it serves as a critical benchmark for evaluating text-based reasoning in visual contexts. To save space, we denote it as "VQA<sup>T</sup> " in the experimental tables.

**POPE** [73] targets object hallucination evaluation by posing binary questions about object presence in images. It employs metrics such as Accuracy, Recall, Precision, and F1 score across three sampling methods. The reported score is calculated by the mean accuracy over the three indicators: adversarial, random, and popular.

**MMMU** [74] challenges models with tasks requiring college-level expertise and reasoning skills. It comprises 11.5K questions drawn from exams, quizzes, and textbooks, spanning six key disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. Featuring 30 subjects and 183 subfields, MMMU involves diverse image types, *e.g.*, charts, diagrams, and chemical structures, demanding advanced perceptual and domain-specific reasoning abilities akin to those of human experts.

**MM-Vet** [72] evaluates six fundamental vision-language capabilities: recognition, OCR, knowledge, language generation, spatial awareness, and mathematical reasoning. It examines 16 specific combinations of these skills through quantitative metrics, offering a nuanced perspective on a model's proficiency in tackling intricate multimodal tasks.

**MMStar** [75] is an elite vision-indispensable multi-modal benchmark designed to rigorously evaluate the genuine multi-modal capabilities of large vision-language models. It comprises 1,500 carefully selected samples spanning six core capability dimensions, including coarse and fine-grained perception, instance reasoning, logical reasoning, science & technology, and mathematics. Each sample is manually verified to ensure that visual content is essential for arriving at the correct answer, thereby filtering out instances answerable through text-only reasoning or dataset bias.

**OCRBench** [76] is a comprehensive evaluation benchmark designed to assess the optical character recognition capabilities of large vision-language models across diverse text-related visual understanding tasks. It encompasses 29 sub-tasks spanning five major categories—text recognition, scene text-centric VQA, document-oriented VQA, key information extraction, and handwritten mathematical expression recognition—comprising 1,000 human-verified question-answer pairs.

**BLINK** [77] is a multimodal benchmark that evaluates visual perception capabilities, which are straightforward for humans yet challenging for current multimodal large language models. It reformats 14 classic computer vision tasks into 3,807 multiple-choice questions, covering relative depth estimation, jigsaw puzzle solving, visual correspondence, forensics detection, spatial reasoning, and more. The benchmark reveals that even state-of-the-art models significantly lag behind human performance on these perceptionoriented tasks.

**Video-MME** [78] is a comprehensive evaluation benchmark for assessing video understanding capabilities of multimodal large language models. It comprises 900 videos totaling 254 hours with 2,700 human-annotated question-answer pairs, spanning short (<2 minutes), medium (4–15 minutes), and long (30–60 minutes) durations across 30 broad categories. The benchmark evaluates models in both withsubtitle and without-subtitle settings, thereby disentangling visual and textual comprehension abilities.

**EgoSchema** [79] is a diagnostic benchmark for very longform video language understanding, featuring over 5,000 human-curated multiple-choice question-answer pairs derived from Ego4D videos. Each question requires temporal reasoning over three-minute video clips, making it substantially more demanding than prior video QA benchmarks that typically involve short clips of only a few seconds. The benchmark specifically targets the assessment of models' capacity for extended temporal comprehension and egocentric activity understanding.

**MVBench** [80] is a comprehensive benchmark designed to evaluate the temporal understanding capabilities of multimodal video large language models. It defines 20 challenging video understanding tasks—such as action sequence recognition, scene transition detection, and attribute change identification—that specifically require dynamic, temporal reasoning rather than reliance on single-frame cues. Evaluation is conducted through a multiple-choice questionanswering format, enabling scalable and reproducible assessment across diverse temporal reasoning dimensions.

## **REFERENCES**

- [1] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, "Language models are few-shot learners," in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 1877–1901.
- [2] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi *et al.*, "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning," *arXiv preprint arXiv:2501.12948*, 2025.
- [3] OpenAI, "Gpt-4v(ision) system card," OpenAI, Tech. Rep., 2023.
- [4] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual instruction tuning," in *Advances in Neural Information Processing Systems*, vol. 36, 2023, pp. 34 892–34 916.

- [5] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, "Qwen-vl: A frontier large vision-language model with versatile abilities," *arXiv preprint arXiv:2308.12966*, 2023.
- [6] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma *et al.*, "How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites," *Science China Information Sciences*, vol. 67, no. 12, p. 220101, 2024.
- [7] Z. Shao, Z. Yu, J. Yu, X. Ouyang, L. Zheng, Z. Gai, M. Wang, Z. Kuang, and J. Ding, "Imp: Highly capable large multimodal models for mobile devices," *IEEE Transactions on Multimedia*, 2025.
- [8] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, Q. Ye, and F. Wei, "Grounding multimodal large language models to the world," in *The Twelfth International Conference on Learning Representations*, 2024.
- [9] C. Ma, Y. Jiang, J. Wu, Z. Yuan, and X. Qi, "Groma: Localized visual tokenization for grounding multimodal large language models," in *European Conference on Computer Vision*. Springer, 2024, pp. 417–435.
- [10] J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, Y. Dan, C. Zhao, G. Xu, C. Li, J. Tian *et al.*, "mplug-docowl: Modularized multimodal large language model for document understanding," *arXiv preprint arXiv:2307.02499*, 2023.
- [11] C. Luo, Y. Shen, Z. Zhu, Q. Zheng, Z. Yu, and C. Yao, "Layoutllm: Layout instruction tuning with large language models for document understanding," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2024, pp. 15 630–15 640.
- [12] J. Wang, H. Xu, J. Ye, M. Yan, W. Shen, J. Zhang, F. Huang, and J. Sang, "Mobile-agent: Autonomous multi-modal mobile device agent with visual perception," *arXiv preprint arXiv:2401.16158*, 2024.
- [13] C. Zhang, Z. Yang, J. Liu, Y. Han, X. Chen, Z. Huang, B. Fu, and G. Yu, "Appagent: Multimodal agents as smartphone users," *arXiv preprint arXiv:2312.13771*, 2023.
- [14] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn *et al.*, "Rt-2: Visionlanguage-action models transfer web knowledge to robotic control," *arXiv preprint arXiv:2307.15818*, 2023.
- [15] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi *et al.*, "Openvla: An open-source vision-language-action model," *arXiv preprint arXiv:2406.09246*, 2024.
- [16] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.
- [17] L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang, "An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models," in *European Conference on Computer Vision*. Springer, 2024, pp. 19–35.
- [18] Y. Zhang, C.-K. Fan, J. Ma, W. Zheng, T. Huang, K. Cheng, D. Gudovskiy, T. Okuno, Y. Nakata, K. Keutzer *et al.*, "Sparsevlm: Visual token sparsification for efficient vision-language model inference," *arXiv preprint arXiv:2410.04417*, 2024.
- [19] T. Liu, L. Shi, R. Hong, Y. Hu, Q. Yin, and L. Zhang, "Multistage vision token dropping: Towards efficient multimodal large language model," *arXiv preprint arXiv:2411.10803*, 2024.
- [20] S. Yang, Y. Chen, Z. Tian, C. Wang, J. Li, B. Yu, and J. Jia, "Visionzip: Longer is better but not necessary in vision language models," *arXiv preprint arXiv:2412.04467*, 2024.
- [21] Q. Zhang, A. Cheng, M. Lu, Z. Zhuo, M. Wang, J. Cao, S. Guo, Q. She, and S. Zhang, "[cls] attention is all you need for trainingfree visual token pruning: Make vlm inference faster," *arXiv preprint arXiv:2412.01818*, 2024.
- [22] M. Elhoushi, A. Shrivastava, D. Liskovich, B. Hosmer, B. Wasti, L. Lai, A. Mahmoud, B. Acun, S. Agarwal, A. Roman *et al.*, "Layerskip: Enabling early exit inference and self-speculative decoding," *arXiv preprint arXiv:2404.16710*, 2024.
- [23] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Improved baselines with visual instruction tuning," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 26 296–26 306.
- [24] Z. Shao, M. Wang, Z. Yu, W. Pan, Y. Yang, T. Wei, H. Zhang, N. Mao, W. Chen, and J. Yu, "Growing a twig to accelerate large vision-language models," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2025, pp. 20 064–20 074.
- [25] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," in *First Conference on Language Modeling*, 2024.

- [26] Y. Sun, L. Dong, Y. Zhu, S. Huang, W. Wang, S. Ma, Q. Zhang, J. Wang, and F. Wei, "You only cache once: Decoder-decoder architectures for language models," *Advances in Neural Information Processing Systems*, vol. 37, pp. 7339–7361, 2025.
- [27] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan *et al.*, "Deepseek-v3 technical report," *arXiv preprint arXiv:2412.19437*, 2024.
- [28] H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang, and L. Qiu, "Llmlingua: Compressing prompts for accelerated inference of large language models," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023, pp. 13 358–13 376.
- [29] J. Mu, X. Li, and N. Goodman, "Learning to compress prompts with gist tokens," *Advances in Neural Information Processing Systems*, vol. 36, pp. 19 327–19 352, 2023.
- [30] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, "FlashAttention: ´ Fast and memory-efficient exact attention with IO-awareness," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.
- [31] S. Dai, H. Genc, R. Venkatesan, and B. Khailany, "Efficient transformer inference with statically structured sparse attention," in *2023 60th ACM/IEEE Design Automation Conference (DAC)*. IEEE, 2023, pp. 1–6.
- [32] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, "Efficient streaming language models with attention sinks," in *The Twelfth International Conference on Learning Representations*, 2024.
- [33] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Re, C. Barrett ´ *et al.*, "H2o: Heavy-hitter oracle for efficient generative inference of large language models," *Advances in Neural Information Processing Systems*, vol. 36, pp. 34 661–34 710, 2023.
- [34] Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *International Conference on Machine Learning*. PMLR, 2023, pp. 19 274–19 286.
- [35] T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao, "Medusa: Simple llm inference acceleration framework with multiple decoding heads," *arXiv preprint arXiv:2401.10774*, 2024.
- [36] F. Liu, Y. Tang, Z. Liu, Y. Ni, D. Tang, K. Han, and Y. Wang, "Kangaroo: Lossless self-speculative decoding for accelerating llms via double early exiting," *Advances in Neural Information Processing Systems*, vol. 37, pp. 11 946–11 965, 2025.
- [37] Z. Zhang, S. Yadav, F. Han, and E. Shutova, "Cross-modal information flow in multimodal large language models," *arXiv preprint arXiv:2411.18620*, 2024.
- [38] W. Ye, Q. Wu, W. Lin, and Y. Zhou, "Fit and prune: Fast and training-free visual token pruning for multi-modal large language models," *arXiv preprint arXiv:2409.10197*, 2024.
- [39] X. Ye, Y. Gan, Y. Ge, X.-P. Zhang, and Y. Tang, "Atp-llava: Adaptive token pruning for large vision language models," *arXiv preprint arXiv:2412.00447*, 2024.
- [40] Q. Wu, W. Lin, W. Ye, Y. Zhou, X. Sun, and R. Ji, "Accelerating multimodal large language models via dynamic visual-token exit and the empirical findings," *arXiv preprint arXiv:2411.19628*, 2024.
- [41] Y. Han, X. Liu, P. Ding, D. Wang, H. Chen, Q. Yan, and S. Huang, "Rethinking token reduction in mllms: Towards a unified paradigm for training-free acceleration," *arXiv preprint arXiv:2411.17686*, 2024.
- [42] J. Chen, L. Ye, J. He, Z.-Y. Wang, D. Khashabi, and A. Yuille, "Llavolta: Efficient multi-modal models via stage-wise visual context compression," *arXiv preprint arXiv:2406.20092*, 2024.
- [43] L. Xing, Q. Huang, X. Dong, J. Lu, P. Zhang, Y. Zang, Y. Cao, C. He, J. Wang, F. Wu *et al.*, "Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction," *arXiv preprint arXiv:2410.17247*, 2024.
- [44] Z. Zhang, P. Pham, W. Zhao, K. Wan, Y.-J. Li, J. Zhou, D. Miranda, A. Kale, and C. Xu, "Treat visual tokens as text? but your mllm only needs fewer efforts to see," *arXiv preprint arXiv:2410.06169*, 2024.
- [45] W. Chai, E. Song, Y. Du, C. Meng, V. Madhavan, O. Bar-Tal, J.- N. Hwang, S. Xie, and C. D. Manning, "Auroracap: Efficient, performant video detailed captioning and a new benchmark," *arXiv preprint arXiv:2410.03051*, 2024.
- [46] D. Bolya, C.-Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman, "Token merging: Your vit but faster," in *The Eleventh International Conference on Learning Representations*, 2023.
- [47] Y. Shang, M. Cai, B. Xu, Y. J. Lee, and Y. Yan, "Llava-prumerge: Adaptive token reduction for efficient large multimodal models," *arXiv preprint arXiv:2403.15388*, 2024.

- [48] P. K. A. Vasu, F. Faghri, C.-L. Li, C. Koc, N. True, A. Antony, G. Santhanam, J. Gabriel, P. Grasch, O. Tuzel *et al.*, "Fastvlm: Efficient vision encoding for vision language models," *arXiv preprint arXiv:2412.13303*, 2024.
- [49] Q. Zhang, A. Cheng, M. Lu, R. Zhang, Z. Zhuo, J. Cao, S. Guo, Q. She, and S. Zhang, "Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms," in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2025, pp. 20 857–20 867.
- [50] Y. Jiang, Q. Wu, W. Lin, W. Yu, and Y. Zhou, "What kind of visual tokens do we need? training-free visual token pruning for multimodal large language models from the perspective of graph," in *Proceedings of the AAAI Conference on Artificial Intelligence*, 2025, aAAI 2025.
- [51] M. Endo, X. Wang, and S. Yeung-Levy, "Feather the throttle: Revisiting visual token pruning for vision-language model acceleration," in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2025, pp. 22 826–22 835.
- [52] Z. Wen, Y. Gao, W. Li, C. He, and L. Zhang, "Token pruning in multimodal large language models: Are we solving the right problem?" in *Findings of the Association for Computational Linguistics: ACL 2025*, 2025, pp. 15 537–15 549.
- [53] M. Gagrani, R. Goel, W. Jeon, J. Park, M. Lee, and C. Lott, "On speculative decoding for multimodal large language models," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 8285–8289.
- [54] W. Zhao, Y. Han, J. Tang, Z. Li, Y. Song, K. Wang, Z. Wang, and Y. You, "A stitch in time saves nine: Small vlm is a precise guidance for accelerating large vlms," *arXiv preprint arXiv:2412.03324*, 2024.
- [55] M. Huo, J. Zhang, H. Wang, J. Xu, Z. Chen, H. Tai, and Y. Chen, "Spec-llava: Accelerating vision-language models with dynamic tree-based speculative decoding," 2025. [Online]. Available: https://arxiv.org/abs/2509.11961
- [56] H. Huang, F. Yang, Z. Liu, X. Yin, D. Li, P. Ren, and E. Barsoum, "Specvlm: Fast speculative decoding in visionlanguage models," *CoRR*, vol. abs/2509.11815, 2025. [Online]. Available: https://arxiv.org/abs/2509.11815
- [57] Y. Ji, J. Zhang, H. Xia, J. Chen, L. Shou, G. Chen, and H. Li, "SpecVLM: Enhancing speculative decoding of video LLMs via verifier-guided token pruning," in *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, Eds. Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 7205–7219. [Online]. Available: https://aclanthology.org/2025.emnlp-main.366/
- [58] J. Kang, H. Shu, W. Li, Y. Zhai, and X. Chen, "Vispec: Accelerating vision-language models with vision-aware speculative decoding," in *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*, 2025. [Online]. Available: https://openreview.net/forum? id=x2BsIdJJJW
- [59] Z. Zhou, X. Ning, K. Hong, T. Fu, J. Xu, S. Li, Y. Lou, L. Wang, Z. Yuan, X. Li *et al.*, "A survey on efficient inference for large language models," *arXiv preprint arXiv:2404.14294*, 2024.
- [60] D. A. Hudson and C. D. Manning, "Gqa: A new dataset for realworld visual reasoning and compositional question answering," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019.
- [61] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, "Towards vqa models that can read," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019, pp. 8317–8326.
- [62] Y. Tang, S. Wang, L. Madaan, and R. Munos, "Beyond verifiable rewards: Scaling reinforcement learning in language models to unverifiable data," in *Advances in Neural Information Processing Systems*, 2025. [Online]. Available: https://openreview. net/forum?id=pc6M9h3T9m
- [63] X. Zhou, Z. Liu, A. Sims, H. Wang, T. Pang, C. Li, L. Wang, M. Lin, and C. Du, "Reinforcing general reasoning without verifiers," in *International Conference on Learning Representations*, 2026. [Online]. Available: https://openreview.net/forum?id=nnwvwge40d
- [64] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "Deepseekmath: Pushing the limits of mathematical reasoning in open language models," *arXiv preprint arXiv:2402.03300*, 2024.
- [65] X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi, C. Shi, Z. Chen, D. Arfeen, R. Cen, and Z. Jia, "Specinfer: Accelerating large language model serving

- with tree-based speculative inference and verification," in *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)*, 2024.
- [66] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, "Llava-next: Improved reasoning, ocr, and world knowledge," January 2024. [Online]. Available: https://llava-vl.github.io/ blog/2024-01-30-llava-next/
- [67] J. Guo, T. Zheng, Y. Bai, B. Li, Y. Wang, K. Zhu, Y. Li, G. Neubig, W. Chen, and X. Yue, "Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale," 2024. [Online]. Available: https://arxiv.org/abs/2412.05237
- [68] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu *et al.*, "Mmbench: Is your multi-modal model an all-around player?" in *European Conference on Computer Vision*. Springer, 2025, pp. 216–233.
- [69] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, Z. Qiu, W. Lin, J. Yang, X. Zheng *et al.*, "Mme: A comprehensive evaluation benchmark for multimodal large language models," *arXiv preprint arXiv:2306.13394*, 2023.
- [70] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan, "Learn to explain: Multimodal reasoning via thought chains for science question answering," *Advances in Neural Information Processing Systems*, vol. 35, pp. 2507–2521, 2022.
- [71] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, "Making the v in vqa matter: Elevating the role of image understanding in visual question answering," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2017, pp. 6904–6913.
- [72] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, "Mm-vet: Evaluating large multimodal models for integrated capabilities," *arXiv preprint arXiv:2308.02490*, 2023.
- [73] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, "Evaluating object hallucination in large vision-language models," *arXiv:2305.10355*, 2023.
- [74] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun *et al.*, "Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 9556–9567.
- [75] L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, and F. Zhao, "Are we on the right way for evaluating large vision-language models?" in *Advances in Neural Information Processing Systems*, 2024.
- [76] Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X.-C. Yin, C.-L. Liu, L. Jin, and X. Bai, "Ocrbench: On the hidden mystery of ocr in large multimodal models," *Science China Information Sciences*, vol. 67, no. 12, 2024.
- [77] X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A. Smith, W.-C. Ma, and R. Krishna, "Blink: Multimodal large language models can see but not perceive," in *Proceedings of the European Conference on Computer Vision*, 2024.
- [78] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang *et al.*, "Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025.
- [79] K. Mangalam, R. Akshulakov, and J. Malik, "Egoschema: A diagnostic benchmark for very long-form video language understanding," in *Advances in Neural Information Processing Systems*, 2023.
- [80] K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Lou, L. Wang, and Y. Qiao, "Mvbench: A comprehensive multimodal video understanding benchmark," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 22 195–22 206.