# MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN

Renze Chen crz@pku.edu.cn Peking University China

Chengrui Zhang zhangchr@stu.pku.edu.cn Peking University China

Zijian Ding<sup>∗</sup> bradyd@cs.ucla.edu University of California, Los Angeles United States

Jingwen Leng leng-jw@cs.sjtu.edu.cn Shanghai Jiao Tong University China

> Yun Liang† ericlyun@pku.edu.cn Peking University China

Size Zheng zhengsz@pku.edu.cn Peking University China

Xuanzhe Liu xzl@pku.edu.cn Peking University China

## Abstract

Recently, memory consumption of Deep Neural Network (DNN) rapidly increases, mainly due to long lifetimes and large shapes of tensors. Graph scheduling has emerged as an effective memory optimization technique, which determines the optimal execution, re-computation, swap-out, and swap-in timings for each operator/tensor. However, it often hurts performance significantly and can only manipulate tensors' lifetimes but not shapes, limiting the optimization space. We find that graph transformation, which can change the tensor shapes and graph structure, creates a new tradeoff space between memory and performance. Nevertheless, graph transformation are applied separately so far, with primary focus on optimizing performance and not memory.

In this paper, we propose MAGIS, a DNN memory optimization framework that coordinates graph transformation with graph scheduling. MAGIS uses a hierarchical tree to represent Fission Transformation (F-Trans), a type of transformation which can effectively reduce tensor shapes in a sub-graph. To keep the complexity low, we build a light-weight search space based on graph structure analysis. MAGIS decomposes graph scheduling into graph transformation and re-ordering and designs an incremental scheduling

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for thirdparty components of this work must be honored. For all other uses, contact the owner/author(s).

ASPLOS '24, April 27-May 1, 2024, La Jolla, CA, USA © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0386-7/24/04. <https://doi.org/10.1145/3620666.3651330>

algorithm to alleviate the scheduling overhead after each graph transformation step to efficiently coordinate them. Experimental results show that compared to state-of-theart works, MAGIS only uses 15%∼85% of their peak memory usage with the same latency1 constraint and obtains a better Pareto boundary in dual-objective optimization of memory and performance. Our code is now available at <https://github.com/pku-liang/MAGIS>.

#### ACM Reference Format:

Renze Chen, Zijian Ding, Size Zheng, Chengrui Zhang, Jingwen Leng, Xuanzhe Liu, and Yun Liang. 2024. MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASPLOS '24), April 27-May 1, 2024, La Jolla, CA, USA. ACM, New York, NY, USA, 15 pages. <https://doi.org/10.1145/3620666.3651330>

