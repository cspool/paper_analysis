# ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling

Shaohuai Shi<sup>1</sup>, Xinglin Pan<sup>2</sup>, Qiang Wang<sup>1</sup>, Chengjian Liu<sup>3</sup>, Xiaozhe Ren<sup>4</sup>, Zhongzhe Hu<sup>4</sup>, Yu Yang<sup>4</sup>, Bo Li<sup>5</sup>, Xiaowen Chu<sup>2,5</sup>

<sup>1</sup>Harbin Institute of Technology, Shenzhen; <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>3</sup>Shenzhen Technology University; <sup>4</sup>Huawei Central Research Institute, Huawei Technologies

<sup>5</sup>The Hong Kong University of Science and Technology

shaohuais@hit.edu.cn, xpan413@connect.hkust-gz.edu.cn, qiang.wang@hit.edu.cn, liuchengjian@sztu.edu.cn {renxiaozhe, huzhongzhe, yangyu}@huawei.com, bli@cse.ust.hk, xwchu@ust.hk

#### **Abstract**

In recent years, large-scale models can be easily scaled to trillions of parameters with sparsely activated mixture-ofexperts (MoE), which significantly improves the model quality while only requiring a sub-linear increase in computational costs. However, MoE layers require the input data to be dynamically routed to a particular GPU for computing during distributed training. The highly dynamic property of data routing and high communication costs in MoE make the training system low scaling efficiency on GPU clusters. In this work, we propose an extensible and efficient MoE training system, ScheMoE, which is equipped with several features. 1) ScheMoE provides a generic scheduling framework that allows the communication and computation tasks in training MoE models to be scheduled in an optimal way. 2) ScheMoE integrates our proposed novel all-to-all collective which better utilizes intra- and inter-connect bandwidths. 3) ScheMoE supports easy extensions of customized all-to-all collectives and data compression approaches while enjoying our scheduling algorithm. Extensive experiments are conducted on a 32-GPU cluster and the results show that ScheMoE outperforms existing state-of-the-art MoE systems, Tutel and Faster-MoE, by 9%-30%.

