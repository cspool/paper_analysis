# Efficient Data Passing for Serverless Inference Workflows: A GPU-Centric Approach

Hao Wu $^{1,2*}$ , Yaochen Liu $^1$ , Minchen Yu $^3$ , Qizhen Weng $^4$ , Junxiao Deng $^1$ , Yue Yu $^1$ , Hao Fan $^1$ , Song Wu $^1$ , Wei Wang $^2$ , and Hai Jin $^1$ 

<sup>1</sup> National Engineering Research Center for Big Data Technology and System, Services Computing Technology and System Lab, Cluster and Grid Computing Lab, School of Computer Science and Technology, Huazhong University of Science and Technology, China

<sup>2</sup> Hong Kong University of Science and Technology, Hong Kong, China

<sup>3</sup> The Chinese University of Hong Kong (Shenzhen), Shenzhen, China

<sup>4</sup> Institute of Artificial Intelligence (TeleAI), China Telecom, China

{wuhao5,u202115348}@hust.edu.cn,yuminchen@cuhk.edu.cn,wengqzh@chinatelecom.cn {dengjunxiao,yuyue18,haofan,wusong}@hust.edu.cn,weiwa@cse.ust.hk,hjin@hust.edu.cn

#### Abstract

Serverless computing offers a compelling paradigm for deploying machine learning inference workflows composed of heterogeneous CPU and GPU functions. However, existing data-passing solutions in serverless systems primarily rely on host memory for data exchange (host-centric), leading to substantial data movement and salient I/O overhead. Moreover, modern GPU communication libraries (e.g., NCCL, NVSHMEM, UCX) are ill-suited to serverless environments, suffering from redundant data copies, underutilized transfer bandwidth, and inefficient temporary GPU storage.

In this paper, we present GROUTER, a GPU-centric data plane system designed for serverless inference workflows. GROUTER first introduces a unified data passing framework that abstracts host-to-GPU and GPU-to-GPU communication while leveraging function placement to reduce redundant copies. It then aggregates available bandwidth across PCIe links, NVLinks, and NICs to enable parallel transfers with performance isolation between functions. GROUTER also implements elastic GPU storage that adapts to idle memory availability and varying data transfer demands. Evaluations on real-world inference services show that GROUTER reduces data passing latency by up to 87% and improves throughput by up to 1.74× compared to state-of-the-art GPU communication libraries.

