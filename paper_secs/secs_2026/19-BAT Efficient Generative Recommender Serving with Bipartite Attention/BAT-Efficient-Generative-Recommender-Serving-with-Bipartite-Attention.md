# **BAT: Efficient Generative Recommender Serving with Bipartite Attention**

Jie Sun\*
jiesun@zju.edu.cn
Zhejiang University
Hangzhou, China
Taobao & Tmall Group of
Alibaba
Beijing, China

Shaohang Wang\*
shhwang@connect.hku.hk
Taobao & Tmall Group of
Alibaba
Beijing, China
The University of Hong
Kong
Hong Kong, China

Zimo Zhang\* zzmyouyou@zju.edu.cn Zhejiang University Hangzhou, China Zhengyu Liu liuzhengyu.lzy@taobao.com Taobao & Tmall Group of Alibaba Beijing, China

Yunlong Xu yunlong.xyl@alibabainc.com Taobao & Tmall Group of Alibaba Beijing, China Peng Sun tengming.sp@taobao.com Taobao & Tmall Group of Alibaba Beijing, China Bo Zhao bo.zhao@aalto.fi Aalto University Espoo, Finland Bingsheng He dcsheb@nus.edu.sg National University of Singapore Singapore, Singapore

Fei Wu wufei@zju.edu.cn Zhejiang University Hangzhou, China Zeke Wang wangzeke@zju.edu.cn Zhejiang University Hangzhou, China

#### **Abstract**

Generative Recommenders (GRs) have recently emerged as promising alternatives to traditional Deep Learning Recommendation Models (DLRMs). Despite their potential, GRs remain computationally expensive in inference, exhibiting compute-bound characteristics similar to the prefill stage of Large Language Model (LLM) inference. Prefix caching can reduce redundant computation by reusing previously constructed KV caches. However, the unique properties of GRs, i.e., highly personalized user profiles and real-time item retrieval, make cache reuse across queries challenging, resulting in limited computational savings.

To address these challenges, we present **BAT**, an efficient serving system for GRs. The key observation is that the semantics between user and item tokens are permutation-invariant. Building on this, we propose *Bipartite Attention*, a novel attention mechanism that enables adaptive selection of either the user or the item as the prompt prefix without compromising accuracy, thereby unlocking new opportunities

\*Equal Contribution.

![](_page_0_Picture_15.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

ASPLOS '26, Pittsburgh, PA, USA.
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2359-9/2026/03
https://doi.org/10.1145/3779212.3790131

for KV cache reuse. We further co-design a disaggregated KV cache pool to proactively manage user-prefix and itemprefix caches as separate components. Since introducing item caches incurs additional memory overhead, we develop a hot-replicated cold-sharded item cache placement strategy that minimizes memory usage and maintains low communication overheads. Finally, we introduce a hotness-aware prompt scheduling strategy to optimize prefix selection under memory constraints. Extensive experiments on multiple recommendation datasets demonstrate that BAT improves serving throughput by up to 1.6× over the conventional user-as-prefix approach, while reducing total computation by up to 58%.

*CCS Concepts:* • Computer systems organization  $\rightarrow$  *Cloud computing;* • Information systems  $\rightarrow$  Online advertising; • Computing methodologies  $\rightarrow$  Machine learning.

**Keywords:** Generative Recommendation; Large Language Models: AI Infrastructure

#### **ACM Reference Format:**

Jie Sun, Shaohang Wang, Zimo Zhang, Zhengyu Liu, Yunlong Xu, Peng Sun, Bo Zhao, Bingsheng He, Fei Wu, and Zeke Wang. 2026. BAT: Efficient Generative Recommender Serving with Bipartite Attention. In *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26), March 21–26, 2026, Pittsburgh, PA, USA.* ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3779212.3790131

