# **VideoRoPE: What Makes for Good Video Rotary Position Embedding?**

Xilin Wei \* 12 Xiaoran liu \* 123 Yuhang Zang 2 Xiaoyi Dong 24 Pan Zhang 2 Yuhang Cao 2 Jian Tong 2 Haodong Duan 2 Qipeng Guo 2 Jiaqi Wang 2 Xipeng Qiu 123 Dahua Lin 245

#### **Abstract**

While Rotary Position Embedding (RoPE) and its variants are widely adopted for their long-context capabilities, the extension of the 1D RoPE to video, with its complex spatio-temporal structure, remains an open challenge. This work first introduces a comprehensive analysis that identifies four key characteristics essential for the effective adaptation of RoPE to video, which have not been fully considered in prior work. As part of our analysis, we introduce a challenging V-NIAH-D (Visual Needle-In-A-Haystack with Distractors) task, which adds periodic distractors into V-NIAH. The V-NIAH-D task demonstrates that previous RoPE variants, lacking appropriate temporal dimension allocation, are easily misled by distractors. Based on our analysis, we introduce VideoRoPE, with a 3D structure designed to preserve spatio-temporal relationships. VideoRoPE features low-frequency temporal allocation to mitigate periodic oscillations, a diagonal layout to maintain spatial symmetry, and adjustable temporal spacing to decouple temporal and spatial indexing. VideoRoPE consistently surpasses previous RoPE variants, across diverse downstream tasks such as long video retrieval, video understanding, and video hallucination. Our code is available at https://github.com/Wiselnn570/VideoRoPE.

### 1. Introduction

Rotary Position Embedding (RoPE) (Su et al., 2024) helps Transformer models understand word order by assigning each token a unique positional 'marker' calculated using a mathematical rotation matrix. RoPE has advantages in long-

Proceedings of the  $42^{nd}$  International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

<span id="page-0-0"></span>*Table 1.* Comparison between different RoPE variants for Video Large Language Models (Video LLMs).

|                                | 2D/3D<br>Structure | Frequency<br>Allocation | Spatial<br>Symmetry | Temporal<br>Index Scaling |
|--------------------------------|--------------------|-------------------------|---------------------|---------------------------|
| Vanilla RoPE (Su et al., 2024) | Х                  | Х                       | Х                   | Х                         |
| TAD-RoPE (Gao et al., 2024)    | Х                  | X                       | X                   | 1                         |
| RoPE-Tie (Su, 2024a)           | ✓                  | X                       | 1                   | X                         |
| M-RoPE (Wang et al., 2024a)    | ✓                  | Х                       | Х                   | X                         |
| VideoRoPE (Ours)               | ✓                  | 1                       | 1                   | 1                         |
| MLVU<br>65,56                  | 61.33              |                         | LongVide            | eoBench                   |

Vanilla RoPE ■ TAD-RoPE ■ M-RoPE ■ VideoRoPE(Ours)

Figure 1. VideoRoPE outperforms RoPE variants on benchmarks.

46 20

(Temporal)

context understanding (Ding et al., 2024b), and continues to be a default choice in leading Large Language Models (LLMs) like the LLaMA (Touvron et al., 2023a;b; Dubey et al., 2024) and QWen (Yang et al., 2024a;b) series.

The original RoPE implementation (Vanilla RoPE) (Su et al., 2024) is designed for sequential 1D data like text. However, recent Video Large Language Models (Video LLMs) (Li et al., 2023; Lin et al., 2023a; Chen et al., 2024a; Maaz et al., 2024b; Zhang et al., 2024d; Wang et al., 2024c; Chen et al., 2024b; Zhang et al., 2024b) process video, which has a more complex spatio and temporal structure. As shown in Tab. 1, although several RoPE-based approaches (Gao et al., 2024; Wang et al., 2024a) have been proposed to support video inputs, these variants exhibit limitations and do not fully satisfy the following key characteristics:

(1) 2D/3D Structure. Some existing Video LLMs direct flatten the video frame into 1D embeddings and apply the 1D structure RoPE (Su et al., 2024; Gao et al., 2024). These solutions fail to capture video data's inherent 2D or 3D (temporal (t), horizontal (x), and vertical (y)) structure, thus hindering explicit spatial and temporal representation.

<sup>\*</sup>Equal contribution <sup>1</sup>Fudan University, Shanghai, China <sup>2</sup>Shanghai AI Laboratory, Shanghai, China <sup>3</sup>Shanghai Innovation Institute, Shanghai, China <sup>4</sup>The Chinese University of Hong Kong <sup>5</sup>CPII under InnoHK. Correspondence to: Yuhang Zang <zangyuhang@pjlab.org.cn>, Qipeng Guo <guoqipeng@pjlab.org.cn>, Jiaqi Wang <wangjiaqi@pjlab.org.cn>.

<span id="page-1-1"></span>![](_page_1_Figure_1.jpeg)

Figure 2. Left: To demonstrate the importance of frequential allocation, based on VIAH (a) we present a more challenging V-NIAH-D task (b) that similar images are inserted as distractors. Right: Compared to M-RoPE, our VideoRoPE is more robust in retrieval and is less affected by distractors. See Fig. [7](#page-6-0) in the Experiments section for details on the horizontal and vertical axes.

(2) Frequency Allocation. Previous approaches such as M-RoPE used in QWen2-VL [\(Wang et al.,](#page-11-0) [2024a\)](#page-11-0) employ 3D structure, dividing feature dimensions into distinct subsets for (t, x, y) encoding, respectively. How to determine the optimal allocation of these dimension subsets, and their associated frequencies [1](#page-1-0) are not well studied. Some previous work allocates the lower dimensions corresponding to the high frequency to represent the t. However, the temporal dimension t is significantly tortured by periodic oscillation, and distant positions may have the same embeddings.

We present a simple setting to verify this point. Based on the previous long-video retrieval task V-NIAH (Visual Needle-In-A-Haystack) [\(Zhang et al.,](#page-12-0) [2024d\)](#page-12-0), we insert several similar images that do not affect the question's answer before and after the needle image as distractor [\(Hsieh et al.,](#page-9-3) [2024;](#page-9-3) [Yuan et al.,](#page-11-4) [2024\)](#page-11-4), forming a new task, V-NIAH-D (Visual Needle-In-A-Haystack with Distractors). As shown in Fig. [2,](#page-1-1) we find that previous M-RoPE is misled by distractors, showing a significant performance decline from V-NIAH to V-NIAH-D. Our observation demonstrates that the periodic oscillation reduces Video LLMs' robustness.

(3) Spatial Symmetry. The distance between the end of the precedent textual input and the start of visual input equals the distance between the end of visual input and the start of subsequent textual input [\(Su,](#page-10-5) [2024b\)](#page-10-5). Such a symmetry ensures that the visual input receives equal contextual influence from both the preceding and subsequent textual information.

