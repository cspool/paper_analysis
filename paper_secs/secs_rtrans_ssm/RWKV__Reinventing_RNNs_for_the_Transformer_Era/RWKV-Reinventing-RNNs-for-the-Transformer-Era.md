# **RWKV: Reinventing RNNs for the Transformer Era**

Bo Peng<sup>1,2\*</sup> Eric Alcaide<sup>2,3,4\*</sup> Quentin Anthony<sup>2,5\*</sup>
Alon Albalak<sup>2,6</sup> Samuel Arcadinho<sup>2,7</sup> Stella Biderman<sup>2,8</sup> Huanqi Cao<sup>9</sup> Xin Cheng<sup>10</sup>
Michael Chung<sup>11</sup> Xingjian Du<sup>1</sup> Matteo Grella<sup>12</sup> Kranthi Kiran GV<sup>2,13</sup> Xuzheng He<sup>2</sup>
Haowen Hou<sup>14</sup> Jiaju Lin<sup>1</sup> Przemysław Kazienko<sup>15</sup> Jan Kocoń<sup>15</sup> Jiaming Kong<sup>16</sup>
Bartłomiej Koptyra<sup>15</sup> Hayden Lau<sup>2</sup> Krishna Sri Ipsit Mantri<sup>17</sup> Ferdinand Mom<sup>18,19</sup>
Atsushi Saito<sup>2,20</sup> Guangyu Song<sup>21</sup> Xiangru Tang<sup>22</sup> Bolun Wang<sup>23</sup> Johan S. Wind<sup>24</sup>
Stanisław Woźniak<sup>15</sup> Ruichong Zhang<sup>9</sup> Zhenyuan Zhang<sup>2</sup> Qihang Zhao<sup>25,26</sup>
Peng Zhou<sup>23</sup> Qinghua Zhou<sup>5</sup> Jian Zhu<sup>27</sup> Rui-Jie Zhu<sup>28,29</sup>

<sup>1</sup>Generative AI Commons <sup>2</sup>EleutherAI <sup>3</sup>U. of Barcelona <sup>4</sup>Charm Therapeutics <sup>5</sup>Ohio State U. <sup>6</sup>U. of C., Santa Barbara <sup>7</sup>Zendesk <sup>8</sup>Booz Allen Hamilton <sup>9</sup>Tsinghua University <sup>10</sup>Peking University <sup>11</sup>Storyteller.io <sup>12</sup>Crisis24 <sup>13</sup>New York U.
 <sup>14</sup>National U. of Singapore <sup>15</sup>Wroclaw U. of Science and Technology <sup>16</sup>Databaker Technology <sup>17</sup>Purdue U. <sup>18</sup>Criteo AI Lab <sup>19</sup>Epita <sup>20</sup>Nextremer <sup>21</sup>Moves <sup>22</sup>Yale U. <sup>23</sup>RuoxinTech <sup>24</sup>U. of Oslo <sup>25</sup>U. of Science and Technology of China
 <sup>26</sup>Kuaishou Technology <sup>27</sup>U. of British Columbia <sup>28</sup>U. of C., Santa Cruz <sup>29</sup>U. of Electronic Science and Technology of China

#### **Abstract**

Transformers have revolutionized almost all natural language processing (NLP) tasks but suffer from memory and computational complexity that scales quadratically with sequence length. In contrast, recurrent neural networks (RNNs) exhibit linear scaling in memory and computational requirements but struggle to match the same performance as Transformers due to limitations in parallelization and scalability. We propose a novel model architecture, Receptance Weighted Key Value (RWKV), that combines the efficient parallelizable training of transformers with the efficient inference of RNNs.

Our approach leverages a linear attention mechanism and allows us to formulate the model as either a Transformer or an RNN, thus parallelizing computations during training and maintains constant computational and memory complexity during inference. We scale our models as large as 14 billion parameters, by far the largest dense RNN ever trained, and find RWKV performs on par with similarly sized Transformers, suggesting future work can leverage this architecture to create more efficient models. This work presents a significant step towards reconciling trade-offs between computational efficiency and model performance in sequence processing tasks. <sup>1</sup>

#### <span id="page-0-2"></span>1 Introduction

Deep learning has greatly advanced artificial intelligence, impacting a range of scientific and industrial uses. These often involve complex sequential data

processing tasks such as natural language understanding, conversational AI, time-series analysis, and indirectly sequential formats like images and graphs (Brown et al., 2020; Ismail Fawaz et al., 2019; Wu et al., 2020; Albalak et al., 2022). Predominant among these techniques include RNNs and Transformers (Vaswani et al., 2017), each with specific benefits and drawbacks. RNNs require less memory, particularly for handling long sequences. However, they suffer from the vanishing gradient problem and non-parallelizability in the time dimension during training, limiting their scalability (Hochreiter, 1998; Le and Zuidema, 2016).

<span id="page-0-1"></span>![](_page_0_Figure_11.jpeg)

Figure 1: Average performance of RWKV models compared to transformers across twelve NLP tasks. For further details, see section 5.

Transformers emerged as a powerful alternative, adept at managing local and long-range dependencies and supporting parallelized training (Tay et al., 2022). Models such as GPT-3 (Brown et al., 2020), ChatGPT (OpenAI, 2022; Kocoń et al., 2023),

<span id="page-0-0"></span><sup>\*</sup> Equal first authorship. Others listed alphabetically. 
<sup>1</sup>Code at: https://github.com/BlinkDL/RWKV-LM

<span id="page-1-0"></span>

| Model               | Time           | Space                                 |
|---------------------|----------------|---------------------------------------|
| Transformer         | 2<br>O(T<br>d) | 2 +<br>O(T<br>T d)                    |
| Reformer            | O(T log T d)   | O(T log T + T d)                      |
| Performer           | O(T d2         | 2<br>log d) O(T d log d + d<br>log d) |
| Linear Transformers | O(T d2<br>)    | 2<br>O(T d + d<br>)                   |
| AFT-full            | 2<br>O(T<br>d) | O(T d)                                |
| AFT-local           | O(T sd)        | O(T d)                                |
| MEGA                | O(cT d)        | O(cd)                                 |
| RWKV (ours)         | O(Td)          | O(d)                                  |

Table 1: Inference complexity comparison with different Transformers. Here T denotes the sequence length, d the feature dimension, c is MEGA's chunk size of quadratic attention, and s is the size of a local window for AFT.

LLaMA [\(Touvron et al.,](#page-12-3) [2023\)](#page-12-3), and Chinchilla [\(Hoffmann et al.,](#page-10-4) [2022\)](#page-10-4) showcase the potential of Transformers in NLP. However, the self-attention mechanism's quadratic complexity makes it computationally and memory intensive for tasks involving long sequences and constrained resources. This has stimulated research to enhance Transformers' scalability, sometimes sacrificing some of their effectiveness [\(Wang et al.,](#page-12-4) [2020;](#page-12-4) [Zaheer et al.,](#page-12-5) [2020;](#page-12-5) [Dao et al.,](#page-9-1) [2022a\)](#page-9-1).

To tackle these challenges, we introduce the Receptance Weighted Key Value (RWKV) model, combining the strengths of RNNs and Transformers while circumventing key drawbacks. RWKV alleviates memory bottleneck and quadratic scaling associated with Transformers [\(Katharopoulos et al.,](#page-10-5) [2020\)](#page-10-5) with efficient linear scaling, while maintaining the expressive properties of the Transformer, such as parallelized training and robust scalability. RWKV reformulates the attention mechanism with a variant of linear attention, replacing traditional dot-product token interaction with more effective channel-directed attention. This implementation, *without approximation*, offers the lowest computational and memory complexity; see Table [1.](#page-1-0)

The motivation behind RWKV is to balance computational efficiency with expressive capacity in neural networks. It offers a solution for handling large-scale models with billions of parameters, exhibiting competitive performance at a reduced computational cost. Experiments suggest RWKV addresses scaling and deployment challenges in AI, especially for sequential data processing, pointing towards more sustainable and efficient AI models.

Our contributions in this paper are as follows:

• The introduction of RWKV, a novel architec-

- ture combining RNNs and Transformer advantages while mitigating their limitations.
- Detailed experiments demonstrating RWKV's performance and efficiency on benchmark datasets for large-scale models.
- The release of pretrained models, from 169 million to 14 billion parameters, trained on the Pile [\(Gao et al.,](#page-9-2) [2020;](#page-9-2) [Biderman et al.,](#page-8-1) [2022\)](#page-8-1).[2](#page-1-1)

