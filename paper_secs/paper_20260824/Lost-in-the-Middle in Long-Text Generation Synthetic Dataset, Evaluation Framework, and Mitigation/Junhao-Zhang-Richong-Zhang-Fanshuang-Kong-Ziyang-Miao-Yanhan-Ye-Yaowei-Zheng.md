# Junhao Zhang, Richong Zhang\*, Fanshuang Kong, Ziyang Miao, Yanhan Ye, Yaowei Zheng

School of Computer Science and Engineering, Beihang University {zhangjunhao, kongfs, miaozy, yeyanhan, hiyouga}@buaa.edu.cn, zhangrc@act.buaa.edu.cn

### Abstract

Existing long-text generation methods primarily concentrate on producing lengthy texts from short inputs, neglecting the long-input and longoutput tasks. Such tasks have numerous practical applications while lacking available benchmarks. Moreover, as the input grows in length, existing methods inevitably encounter the "lostin-the-middle" phenomenon. In this paper, we first introduce a Long Input and Output Benchmark (LONGINOUTBENCH), including a synthetic dataset and a comprehensive evaluation framework, addressing the challenge of the missing benchmark. We then develop the Retrieval-Augmented Long-Text Writer (RAL-WRITER), which retrieves and restates important yet overlooked content, mitigating the "lost-in-the-middle" issue by constructing explicit prompts. We finally employ the proposed LONGINOUTBENCH to evaluate our RAL-WRITER against comparable baselines, and the results demonstrate the effectiveness of our approach. Our code has been released at <https://github.com/OnlyAR/RAL-Writer>

