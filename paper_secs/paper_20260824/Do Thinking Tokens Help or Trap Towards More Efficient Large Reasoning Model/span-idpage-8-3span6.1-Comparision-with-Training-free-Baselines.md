# <span id="page-8-3"></span>6.1 Comparision with Training-free Baselines

We first establish the performance ceiling of training-free approaches. While the promptbased NoThink method exhibits the expected performance-efficiency trade-off (26.3% token reduction with a 2.1-point accuracy penalty), ThinkTokenPenalty emerges as the superior training-free method, achieving 30.6% token reduction while preserving accuracy on DeepSeek-R1-Distill-Qwen-1.5B. This finding reveals that thinking tokens in the base model may contribute minimally to reasoning quality, primarily inflating response length. Such result aligns with and extend the key insights from Section 3.1 across a broader range of datasets, providing additional empirical validation that thinking tokens are largely dispensable for reasoning.

However, training-based optimization unlocks substantially greater potential. Compared to ThinkTokenPenalty, DuP-PO outperforms by 3.9 points while maintaining a significant token efficiency, demonstrating superior performance-efficiency trade-offs that training-free methods cannot achieve.

#### Takeaway

Training-based optimization is essential: DuP-PO substantially outperforms all training-free approaches on performance.

#### <span id="page-8-4"></span>6.2 Comparision with Base Models

Our empirical evaluation on DeepSeek-R1-Distill-Qwen-1.5B reveals that DuP-PO

achieves substantial improvements with minimal training overhead. Specifically, the method requires only 80 RL steps to deliver 4*.*0 points average performance gain alongside 15*.*4% token reduction.

DuP-PO achieves consistent performance improvements across benchmark complexities with substantial efficiency gains. Simpler benchmarks like MATH500 show pronounced benefits with 3*.*5-point improvements and 24*.*7% token savings. On more challenging benchmarks such as AIME24 and AIME25, performance gains remain consistent while token reduction becomes more conservative. AIME24 improves from 21*.*8 to 25*.*0 with 8*.*3% token reduction, and AIME25 from 20*.*4 to 21*.*7 with 10*.*7% fewer tokens. These results indicate DuP-PO's adaptive optimization across problem complexities.

#### **Takeaway**

DuP-PO consistently **improves both performance and token efficiency** over the base model with **minimal training cost**.

#### <span id="page-9-8"></span>**6.3 Comparision with GRPO**

We conduct a controlled comparison against GRPO to isolate the contributions of our proposed innovations. Standard GRPO training (90 steps) yields 2*.*7 points average improvement with 6*.*2% token reduction relative to the base model.

DuP-PO demonstrates clear algorithmic superiority. With fewer training iterations (80 vs 90 steps), our method achieves 1*.*3 points higher accuracy than GRPO while consuming fewer reasoning tokens (5*,*162 vs 5*,*724). This demonstrates superior efficiency with reduced training iterations and lower token consumption during inference.

#### **Takeaway**

DuP-PO outperforms GRPO in **performance**, **token efficiency**, and **training speed** simultaneously.

## <span id="page-9-9"></span>**7 Conclusion**

In this paper, we present DuP-PO, a novel reinforcement learning algorithm designed to

address the overthinking problem in the 1*.*5B LRM. Our experimental results demonstrate that frequent use of thinking tokens is not necessarily essential for model performance improvement. Through fine-grained control of thinking tokens, our method achieves superior balance between performance enhancement and token efficiency compared to baseline approaches, requiring only lightweight training on the base LRM across multiple mathematical benchmarks. For future work, we plan to validate the reliability and robustness of our approach by extending experiments to larger model architectures and diverse domain benchmarks.

## **References**

<span id="page-9-4"></span>Pranjal Aggarwal and Sean Welleck. 2025. L1: Controlling how long a reasoning model thinks with reinforcement learning. *arXiv preprint arXiv:2503.04697*.

<span id="page-9-7"></span>Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, and Donghyun Kwak. 2025. [Online difficulty filtering for rea](https://arxiv.org/abs/2504.03380)[soning oriented reinforcement learning.](https://arxiv.org/abs/2504.03380) *Preprint*, arXiv:2504.03380.

<span id="page-9-2"></span>Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2025. [Do](https://arxiv.org/abs/2412.21187) [not think that much for 2+3=? on the overthink](https://arxiv.org/abs/2412.21187)[ing of o1-like llms.](https://arxiv.org/abs/2412.21187) *Preprint*, arXiv:2412.21187.

<span id="page-9-6"></span>Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan Han, Chen Luo, Jing Huang, Zhen Li, and 1 others. 2025. Stepwise perplexity-guided refinement for efficient chain-of-thought reasoning in large language models. *arXiv preprint arXiv:2502.13260*.

<span id="page-9-0"></span>DeepSeek-AI. 2025. [Deepseek-r1: Incentivizing rea](https://doi.org/10.48550/ARXIV.2501.12948)[soning capability in llms via reinforcement learn](https://doi.org/10.48550/ARXIV.2501.12948)[ing.](https://doi.org/10.48550/ARXIV.2501.12948) *CoRR*, abs/2501.12948.

<span id="page-9-5"></span>Gongfan Fang, Xinyin Ma, and Xinchao Wang. 2025. [Thinkless: Llm learns when to think.](https://arxiv.org/abs/2505.13379) *Preprint*, arXiv:2505.13379.

<span id="page-9-1"></span>Sicheng Feng, Gongfan Fang, Xinyin Ma, and Xinchao Wang. 2025. [Efficient reasoning models: A](https://arxiv.org/abs/2504.10903) [survey.](https://arxiv.org/abs/2504.10903) *Preprint*, arXiv:2504.10903.

<span id="page-9-3"></span>Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Aurick Qiao, and Hao Zhang. 2024. Efficiently serving llm reasoning programs with certaindex. *arXiv preprint arXiv:2412.20993*.

- <span id="page-10-20"></span>Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. [Olympiadbench: A challenging bench](https://arxiv.org/abs/2402.14008)[mark for promoting agi with olympiad-level bilin](https://arxiv.org/abs/2402.14008)[gual multimodal scientific problems.](https://arxiv.org/abs/2402.14008) *Preprint*, arXiv:2402.14008.
- <span id="page-10-16"></span>Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. 2025. Thinkprune: Pruning long chain-ofthought of llms via reinforcement learning. *arXiv preprint arXiv:2504.01296*.
- <span id="page-10-18"></span>Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. 2024. [Numinamath.](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)
- <span id="page-10-5"></span>Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. 2025. C3ot: Generating shorter chain-ofthought without compromising effectiveness. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pages 24312–24320.
- <span id="page-10-19"></span>Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. [Solving quantitative rea](https://arxiv.org/abs/2206.14858)[soning problems with language models.](https://arxiv.org/abs/2206.14858) *Preprint*, arXiv:2206.14858.
- <span id="page-10-21"></span>Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. [Let's verify step by step.](https://openreview.net/forum?id=v8L0pN6EOi) In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net.
- <span id="page-10-10"></span>Zichen Liu, Changyu Chen, Wenjun Li, Tianyu Pang, Chao Du, and Min Lin. 2025. There may not be aha moment in r1-zero-like training — a pilot study. [https://oatllm.notion.](https://oatllm.notion.site/oat-zero) [site/oat-zero](https://oatllm.notion.site/oat-zero). Notion Blog.
- <span id="page-10-4"></span>Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. 2025a. [O1-pruner: Length](https://arxiv.org/abs/2501.12570)[harmonizing fine-tuning for o1-like reasoning](https://arxiv.org/abs/2501.12570) [pruning.](https://arxiv.org/abs/2501.12570) *Preprint*, arXiv:2501.12570.
- <span id="page-10-13"></span>Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. 2025b. O1-pruner: Lengthharmonizing fine-tuning for o1-like reasoning pruning. *arXiv preprint arXiv:2501.12570*.
- <span id="page-10-12"></span>Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. 2025a. Reasoning models can be effective without thinking. *arXiv preprint arXiv:2504.09858*.

- <span id="page-10-6"></span>Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. 2025b. Cotvalve: Length-compressible chain-of-thought tuning. *arXiv preprint arXiv:2502.09601*.
- <span id="page-10-3"></span>Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel J. Candès, and Tatsunori Hashimoto. 2025a. [s1:](https://doi.org/10.48550/ARXIV.2501.19393) [Simple test-time scaling.](https://doi.org/10.48550/ARXIV.2501.19393) *CoRR*, abs/2501.19393.
- <span id="page-10-8"></span>Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025b. [s1: Simple](https://arxiv.org/abs/2501.19393) [test-time scaling.](https://arxiv.org/abs/2501.19393) *Preprint*, arXiv:2501.19393.
- <span id="page-10-7"></span>Tergel Munkhbat, Namgyu Ho, Seo Hyun Kim, Yongjin Yang, Yujin Kim, and Se-Young Yun. 2025. Self-training elicits concise reasoning in large language models. *arXiv preprint arXiv:2502.20122*.
- <span id="page-10-9"></span>Open-R1-Team. 2025. [Mini r1 countdown game.](https://huggingface.co/blog/open-r1/mini-r1-contdown-game) Blog post.
- <span id="page-10-0"></span>OpenAI. 2024. Learning to reason with llms. [https://openai.com/index/](https://openai.com/index/learning-to-reason-with-llms/) [learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/). Accessed: 2025-05-07.
- <span id="page-10-11"></span>Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, and Jing Shao. 2025. [Demystifying](https://arxiv.org/abs/2506.02867) [reasoning dynamics with mutual information:](https://arxiv.org/abs/2506.02867) [Thinking tokens are information peaks in llm](https://arxiv.org/abs/2506.02867) [reasoning.](https://arxiv.org/abs/2506.02867) *Preprint*, arXiv:2506.02867.
- <span id="page-10-15"></span>Yuxiao Qu, Matthew YR Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, and Aviral Kumar. 2025. Optimizing test-time compute via meta reinforcement fine-tuning. *arXiv preprint arXiv:2503.07572*.
- <span id="page-10-1"></span>Qwen Team. 2025. Qwq-32b-preview. [https:](https://qwenlm.github.io/blog/qwq-32b-preview/) [//qwenlm.github.io/blog/qwq-32b-preview/](https://qwenlm.github.io/blog/qwq-32b-preview/). Accessed: 15 March 2025.
- <span id="page-10-17"></span>John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. [Prox](https://api.semanticscholar.org/CorpusID:28695052)[imal policy optimization algorithms.](https://api.semanticscholar.org/CorpusID:28695052) *ArXiv*, abs/1707.06347.
- <span id="page-10-2"></span>ByteDance Seed, :, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, Yufeng Yuan, Yu Yue, Lin Yan, Qiying Yu, Xiaochen Zuo, Chi Zhang, Ruofei Zhu, Zhecheng An, and 255 others. 2025. [Seed1.5-thinking: Ad](https://arxiv.org/abs/2504.13914)[vancing superb reasoning models with reinforce](https://arxiv.org/abs/2504.13914)[ment learning.](https://arxiv.org/abs/2504.13914) *Preprint*, arXiv:2504.13914.
- <span id="page-10-14"></span>Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, Zhaoxiang Liu, and Shiguo Lian. 2025. Dast: Difficulty-adaptive slow-thinking

- for large reasoning models. *arXiv preprint arXiv:2503.04472*.
- <span id="page-11-20"></span>Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv: 2409.19256*.
- <span id="page-11-4"></span>Jinyan Su and Claire Cardie. 2025. [Thinking](https://arxiv.org/abs/2505.18298) [fast and right: Balancing accuracy and rea](https://arxiv.org/abs/2505.18298)[soning length with adaptive rewards.](https://arxiv.org/abs/2505.18298) *Preprint*, arXiv:2505.18298.
- <span id="page-11-1"></span>Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, and Xia Ben Hu. 2025. [Stop overthinking: A survey](https://doi.org/10.48550/ARXIV.2503.16419) [on efficient reasoning for large language models.](https://doi.org/10.48550/ARXIV.2503.16419) *CoRR*, abs/2503.16419.
- <span id="page-11-11"></span>Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, and 1 others. 2025. Kimi k1.5: Scaling reinforcement learning with llms. *arXiv preprint arXiv:2501.12599*.
- <span id="page-11-14"></span>Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, and Dongbin Zhao. 2025. Learning when to think: Shaping adaptive reasoning in r1-style models via multistage rl. *arXiv preprint arXiv:2505.10832*.
- <span id="page-11-7"></span>Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. 2025a. [Beyond the 80/20 rule: High-entropy minority](https://arxiv.org/abs/2506.01939) [tokens drive effective reinforcement learning for](https://arxiv.org/abs/2506.01939) [llm reasoning.](https://arxiv.org/abs/2506.01939) *Preprint*, arXiv:2506.01939.
- <span id="page-11-19"></span>Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2025b. [Thoughts are all over the place: On](https://arxiv.org/abs/2501.18585) [the underthinking of o1-like llms.](https://arxiv.org/abs/2501.18585) *Preprint*, arXiv:2501.18585.
- <span id="page-11-9"></span>Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, and 1 others. 2025c. Thoughts are all over the place: On the underthinking of o1-like llms. *arXiv preprint arXiv:2501.18585*.
- <span id="page-11-2"></span>Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, and Wenjie Li. 2025. Tokenskip: Controllable chain-of-thought compression in llms. *arXiv preprint arXiv:2502.12067*.
- <span id="page-11-15"></span>Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. 2025. Learning to reason under off-policy guidance. *arXiv preprint arXiv:2504.14945*.

- <span id="page-11-16"></span>An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. [Qwen3 technical report.](https://arxiv.org/abs/2505.09388) *Preprint*, arXiv:2505.09388.
- <span id="page-11-8"></span>Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Zheng Lin, Li Cao, and Weiping Wang. 2025b. Dynamic early exit in reasoning models. *arXiv preprint arXiv:2504.15895*.
- <span id="page-11-12"></span>Junjie Yang, Ke Lin, and Xing Yu. 2025c. Think when you need: Self-adaptive chain-of-thought learning. *arXiv preprint arXiv:2504.03234*.
- <span id="page-11-6"></span>Shu Yang, Junchao Wu, Xin Chen, Yunze Xiao, Xinyi Yang, Derek F. Wong, and Di Wang. 2025d. [Understanding aha moments: from external ob](https://arxiv.org/abs/2504.02956)[servations to internal mechanisms.](https://arxiv.org/abs/2504.02956) *Preprint*, arXiv:2504.02956.
- <span id="page-11-10"></span>Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. 2025. Demystifying long chain-of-thought reasoning in llms. *arXiv preprint arXiv:2502.03373*.
- <span id="page-11-3"></span>Ping Yu, Jing Xu, Jason Weston, and Ilia Kulikov. 2024. Distilling system 2 into system 1. *arXiv preprint arXiv:2407.06023*.
- <span id="page-11-17"></span>Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025. [Dapo: An open](https://arxiv.org/abs/2503.14476)[source llm reinforcement learning system at scale.](https://arxiv.org/abs/2503.14476) *Preprint*, arXiv:2503.14476.
- <span id="page-11-18"></span>Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, Xiangyu Yu, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Chi Zhang, and 8 others. 2025. [Vapo: Efficient](https://arxiv.org/abs/2504.05118) [and reliable reinforcement learning for advanced](https://arxiv.org/abs/2504.05118) [reasoning tasks.](https://arxiv.org/abs/2504.05118) *Preprint*, arXiv:2504.05118.
- <span id="page-11-0"></span>Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. 2025. [Simplerl-zoo: Investigating and taming zero re](https://arxiv.org/abs/2503.18892)[inforcement learning for open base models in the](https://arxiv.org/abs/2503.18892) [wild.](https://arxiv.org/abs/2503.18892) *Preprint*, arXiv:2503.18892.
- <span id="page-11-13"></span>Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. 2025. Adaptthink: Reasoning models can learn when to think. *arXiv preprint arXiv:2505.13417*.
- <span id="page-11-5"></span>Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. 2025. [R1-zero's "aha moment" in visual reasoning on a](https://arxiv.org/abs/2503.05132) [2b non-sft model.](https://arxiv.org/abs/2503.05132) *Preprint*, arXiv:2503.05132.

## **Contents**

| 1 | Introduction                                                                                                                                                                                                                                                 |                                      |  |  |  |  |  |  |  |  |  |  |  |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|--|--|--|--|--|--|--|--|--|--|--|
| 2 | Related work<br>Analysis on Thinking Tokens<br><br>Mitigating Overthinking<br>                                                                                                                                                                               |                                      |  |  |  |  |  |  |  |  |  |  |  |
| 3 | Rethinking the Role of Thinking Tokens<br>3.1<br>When Fewer Thinking Tokens Maintain a Good Reasoning<br><br>3.2<br>Mechanisms Underlying Thinking Traps<br>                                                                                                 | 3<br>3<br>4                          |  |  |  |  |  |  |  |  |  |  |  |
| 4 | Methodology<br>4.1<br>Preliminary<br><br>4.2<br>GRPO<br><br>4.3<br>Token-Level Policy Gradient<br><br>DuP-PO<br>4.4<br><br>4.4.1<br>Dual-Policy Sampling<br><br>4.4.2<br>Token-Level Advantage Scaling<br><br>4.4.3<br>Policy Shaping on Thinking Tokens<br> | 5<br>5<br>5<br>5<br>6<br>6<br>6<br>7 |  |  |  |  |  |  |  |  |  |  |  |
| 5 | Experiments<br>5.1<br>Setup<br><br>5.2<br>Baselines<br><br>5.3<br>RL Practice<br><br>5.3.1<br>Implementation Details<br><br>5.3.2<br>Reward Design<br>                                                                                                       | 8<br>8<br>8<br>8<br>8<br>9           |  |  |  |  |  |  |  |  |  |  |  |
| 6 | Results and Discussion<br>6.1<br>Comparision with Training-free Baselines<br><br>6.2<br>Comparision with Base Models<br><br>6.3<br>Comparision with GRPO<br>                                                                                                 | 9<br>9<br>9<br>10                    |  |  |  |  |  |  |  |  |  |  |  |
| 7 | Conclusion<br>10                                                                                                                                                                                                                                             |                                      |  |  |  |  |  |  |  |  |  |  |  |