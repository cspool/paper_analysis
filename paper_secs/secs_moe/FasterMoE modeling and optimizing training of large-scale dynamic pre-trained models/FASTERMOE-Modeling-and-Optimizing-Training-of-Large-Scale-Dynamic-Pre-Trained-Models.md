# FASTERMOE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models

Jiaao He Tsinghua University hja20@mails.tsinghua.edu.cn

Haojie Wang Tsinghua University wanghaojie@tsinghua.edu.cn Jidong Zhai\*
Tsinghua University
zhaijidong@tsinghua.edu.cn

Fuwen Luo Tsinghua University Ifw19@mails.tsinghua.edu.cn

Qin Li Tsinghua University liqin20@mails.tsinghua.edu.cn Tiago Antunes
Tsinghua University
vazama10@mails.tsinghua.edu.cn

Shangfeng Shi Tsinghua University ssf20@mails.tsinghua.edu.cn

#### **Abstract**

The current trend in deep learning is to scale models to extremely large sizes with the objective of increasing their accuracy. Mixture-of-Expert (MoE) is the most popular pretrained model that makes feasible the training of models with parameters beyond trillion-scale. Thanks to the dynamic activation of experts, i.e., shallow layers specialized in certain domains, it allows for sparse training of bigger models, removing the linearity between model size and computation. However, different from traditional deep learning models, it draws huge challenges to the efficiency of these training systems, including dynamic load imbalance, inefficient synchronous execution mode, and congested all-to-all communication.

To address these challenges, we first propose a performance model that can both accurately predict the latency of different operations of a specific training task, and intuitively analyze its end-to-end performance via a novel roofline-like model. Then, guided by this model, we invent a dynamic shadowing approach to cope with load imbalance, and a smart fine-grained schedule that splits different operations and executes them concurrently. We design a congestion-avoiding expert selection strategy that relieves network congestion for the lower latency of iterations, when modification of expert selection is allowed. We implement

\*Corresponding author

![](_page_0_Picture_13.jpeg)

This work is licensed under a Creative Commons Attribution-NonCommercial International 4.0 License.

PPoPP '22, April 2–6, 2022, Seoul, Republic of Korea © 2022 Association for Computing Machinery. ACM ISBN 978-1-4503-9204-4/22/04.

https://doi.org/10.1145/3503221.3508418

and integrate the above optimizations as a general system, FASTERMOE, empowering efficient distributed MoE model training. FASTERMOE is evaluated on different cluster systems using up to 64 GPUs. It achieves 1.37× - 17.87× speedup compared with state-of-the-art systems for large models, including ZeRO, GShard, and BASE Layer.

Source code of FASTERMOE is now available at https://github.com/thu-pacman/FasterMoE.

CCS Concepts: • Computing methodologies  $\rightarrow$  Massively parallel algorithms.

*Keywords:* Distributed Deep Learning, Parallelism, Performance Modeling

#### 1 Introduction

A pre-trained model is a model that is already trained with a wide range of samples, containing *knowledge*. Compared to training a new model from scratch, using a pre-trained model can greatly reduce resource consumption. Therefore, pre-trained models have been becoming more popular in many domains, such as image processing [22], reading comprehension [3], and language generation [1, 24, 38]. In recent years, both academia and industry are interested in developing pre-trained models with higher accuracy.

One of the promising ways for improving model accuracy is scaling up model size. Researchers have shown that larger models bring significantly higher accuracy [1, 5, 11]. Among these works, Mixture-of-Expert (MoE) [17] appears promising to scale models to extreme size. As shown in Figure 1, different from directly scaling small models to a large dense model, an MoE model consists of many small models, namely *experts*. Training samples are fed into different experts, dynamically selected by a light-weight trainable gate network. In MoE, as experts are sparsely activated and much extra computation is saved, it can significantly increase the number of samples trained in the same period of time and improve the model accuracy compared with classic

dense pre-trained models. Therefore, such dynamic models are becoming increasingly important for training a giant model, such as Google's GShard [11] and Facebook's BASE Layer [12].

<span id="page-1-0"></span>![](_page_1_Figure_3.jpeg)

**Figure 1.** MoE structure for training a large model.

Although flexible MoE structure makes it more feasible to train a giant model beyond trillion-scale, it is still extremely costly. A 600 billion MoE model in GShard [11] takes 2,048 TPUs 4 days to train. To reduce training time, expert parallelism is introduced to train MoE models distributedly, where experts are partitioned onto different workers and each worker processes a different batch of training samples (detailed in Section 2.3). In MoE layers, each input is sent to its desired expert by the system through the network.

The inefficiency of existing MoE training methods mainly comes from dynamic expert selection and flexible MoE structure. We summarize three main challenges when training an MoE model as follows.

**Dynamic expert selection.** With increasing model size, experts are commonly distributed across different workers. A popular expert receives more tokens than others, which causes its resident worker to be heavily loaded while other workers may idle. Even worse, this pattern changes among different iterations dynamically. This behavior significantly affects hardware utilization and training efficiency.

Inefficient synchronous operations. All experts need to obtain their input from many other workers, being one of the most time-consuming operations when training MoE models. It is commonly implemented as synchronous all-to-all operations with variable message sizes. Considering that non-uniform expert selection leads to severe imbalance in both computation and communication, this method of launching synchronous operations can lead to much more overhead.

Mismatch of model design and network topology. Expert selection in MoE model training can significantly affect the training efficiency since it determines both load balance and communication traffic. Existing works like GShard [11] and BASE Layer [12] use different expert selection strategies to balance computation load but ignore communication, despite that the network topology is critical to communication performance. Network contention is frequently caused in current widely-used network topology by the complex communication in MoE.

To address these challenges, we propose FASTERMOE, a highly efficient distributed system for training large dynamic pre-trained models. To capture dynamic behavior introduced by MoE, we build a precise performance model for training tasks. Given an MoE model and system configuration, our performance model can first estimate the latency of operations, and then visualize the task with a roofline-like model for better understanding its performance. Guided by our performance model, we further propose three key optimization strategies for the training process. Dynamic shadowing is enabled to reduce the idling caused by imbalanced expert selection. A fine-grained smart scheduling strategy is introduced to perform computation and communication operations asynchronously, which can fully exploit inter-operation parallelism. Finally, a congestion-avoiding expert selection strategy is designed to lower the latency of iterations, with promising convergence results.

FASTERMOE is tested on 2 different clusters with up to 64 GPUs. Evaluation results show that FASTERMOE achieves up to 17.87× speedup compared with ZeRO Optimizer [25], with mathematical equivalence. When modification of expert selection is allowed, FASTERMOE is 1.37× faster in convergence time over GShard, and 2.19× over BASE Layer.

In summary, we make the following contributions:

- We design a performance model, which can accurately estimate the performance for a given MoE model with a specific parallel strategy.
- We present a roofline-like model to analyze the performance and theoretical limit of different parallelisms, and the improvements of our optimizations below.
- Guided by our performance model, we invent a dynamic shadowing approach to reduce the impact of skew expert popularity.
- We create a smart fine-grained schedule of communication and computation to reduce their latency jointly.
- We design an adjusted expert selection strategy at runtime for faster communication with less congestion, whereas the loss is decreasing in promising slope.
- We implement the above techniques into an end-toend MoE training system, FASTERMOE, and achieve up to 17.87× speedup over state-of-the-art systems.

The rest of this paper is organized as follows. Section 2 introduces the background and main challenges of distributed training. Section 3 presents our performance model and Section 4 introduces optimization strategies guided by our performance model. Section 5 evaluates FasterMoE. More related works are described in Section 6, and Section 7 concludes this paper.

