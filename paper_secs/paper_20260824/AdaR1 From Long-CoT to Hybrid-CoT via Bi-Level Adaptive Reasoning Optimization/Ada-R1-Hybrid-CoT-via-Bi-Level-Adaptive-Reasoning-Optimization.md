# Ada-R1: Hybrid CoT via Bi-Level Adaptive Reasoning Optimization

Haotian Luo<sup>1</sup>\*, Haiying He<sup>2</sup>\*, Yibo Wang<sup>3</sup>, Jinluan Yang<sup>4</sup>, Rui Liu<sup>5</sup> Naiqiang Tan<sup>5</sup>, Xiaochun Cao<sup>1</sup>, Dacheng Tao<sup>6</sup>, Li Shen<sup>1†</sup>

Sun Yat-sen University;
China Agricultural University;
Tsinghua University;
Zhejiang University;
Didichuxing Co. Ltd;
Nanyang Technological University

## **Abstract**

Recently, long-thought reasoning models achieve strong performance on complex reasoning tasks, but often incur substantial inference overhead, making efficiency a critical concern. Our empirical analysis reveals that the benefit of using Long-CoT varies across problems: while some problems require elaborate reasoning, others show no improvement—or even degraded accuracy. This motivates adaptive reasoning strategies that tailor reasoning depth to the input. However, prior work primarily reduces redundancy within long reasoning paths, limiting exploration of more efficient strategies beyond the Long-CoT paradigm. To address this, we propose a novel two-stage framework for adaptive and efficient reasoning. First, we construct a hybrid reasoning model by merging long and short CoT models to enable diverse reasoning styles. Second, we apply bi-level preference training to guide the model to select suitable reasoning styles (group-level), and prefer concise and correct reasoning within each style group (instance-level). Experiments demonstrate that our method significantly reduces inference costs compared to other baseline approaches, while maintaining performance. Notably, on five mathematical datasets, the average length of reasoning is reduced by more than 50%, highlighting the potential of adaptive strategies to optimize reasoning efficiency in large language models. Our code is coming soon at https://github.com/StarDewXXX/AdaR1

