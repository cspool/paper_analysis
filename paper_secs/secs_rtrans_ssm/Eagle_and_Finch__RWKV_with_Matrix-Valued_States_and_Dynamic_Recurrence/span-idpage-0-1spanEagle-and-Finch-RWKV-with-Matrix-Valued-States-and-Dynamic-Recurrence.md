# <span id="page-0-1"></span>Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence

 $Bo\ Peng^{1,2,*}Daniel\ Goldstein^{2,3,*},\ Quentin\ Anthony^{2,4,23,*},\ Alon\ Albalak^{2,5,6},\ Eric\ Alcaide^{2,7,8},\ Stella\ Biderman^2,\ Eugene\ Cheah^{1,2,3},\ Xingjian\ Du^1,\ Teddy\ Ferdinan^9,\ Haowen\ Hou^{10},\ Przemysław\ Kazienko^9,\ Kranthi\ Kiran\ GV^{2,11},\ Jan\ Kocoń^9,\ Bartłomiej\ Koptyra^9,\ Satyapriya\ Krishna^{12},\ Ronald\ McClelland\ Jr.^{2,13},\ Jiaju\ Lin^{24},\ Niklas\ Muennighoff^{14},\ Fares\ Obeid^2,\ Atsushi\ Saito^{2,15},\ Guangyu\ Song^{2,25},\ Haoqin\ Tu^{16,17},\ Cahya\ Wirawan^{18},\ Stanisław\ Woźniak^9,\ Ruichong\ Zhang^{19},\ Bingchen\ Zhao^{20},\ Qihang\ Zhao^{21},\ Peng\ Zhou^{21},\ Jian\ Zhu^{22},\ and\ Rui-Jie\ Zhu^{17}$ 

<sup>1</sup>RWKV Project (under Linux Foundation AI & Data), <sup>2</sup>EleutherAI, <sup>3</sup>Recursal AI, <sup>4</sup>Ohio State University, <sup>5</sup>University of California, Santa Barbara, <sup>6</sup>SynthLabs, <sup>7</sup>Charm Therapeutics, <sup>8</sup>Dalle Molle Institute for Artificial Intelligence Research, <sup>9</sup>Wroclaw Tech, <sup>10</sup>Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), <sup>11</sup>New York University, <sup>12</sup>Harvard University, <sup>13</sup>Ronsor Labs, <sup>14</sup>Contextual AI, <sup>15</sup>Nextremer Co. Ltd., <sup>16</sup>University of Chinese Academy of Sciences, <sup>17</sup>University of California, Santa Cruz, <sup>18</sup>AI-Research.id, <sup>19</sup>Tsinghua University, <sup>20</sup>University of Edinburgh, <sup>21</sup>LuxiTech Co. Ltd., <sup>22</sup>University of British Columbia, <sup>23</sup>Zyphra, <sup>24</sup>Pennsylvania State University, <sup>25</sup>Tano Labs

#### **Abstract**

We present Eagle (RWKV-5) and Finch (RWKV-6), sequence models improving upon the RWKV (RWKV-4) (Peng et al., 2023) architecture. Our architectural design advancements include multiheaded matrix-valued states and a dynamic recurrence mechanism that improve expressivity while maintaining the inference efficiency characteristics of RNNs. We introduce a new multilingual corpus with 1.12 trillion tokens and a fast tokenizer based on greedy matching for enhanced multilinguality. We trained four Eagle models, ranging from 0.46 to 7.5 billion parameters, and two Finch models with 1.6 and 3.1 billion parameters and find that they achieve competitive performance across a wide variety of benchmarks. We release all our models on HuggingFace under the Apache 2.0 license. <sup>1</sup>

Training code at: https://github.com/RWKV/RWKV-LM Inference code at: https://github.com/RWKV/ChatRWKV

Time-parallel training code at: https://github.com/RWKV/RWKV-infctx-trainer

<sup>\*</sup>Equal first authorship. Others listed alphabetically.

<span id="page-0-0"></span><sup>&</sup>lt;sup>1</sup>Models at: https://huggingface.co/RWKV

