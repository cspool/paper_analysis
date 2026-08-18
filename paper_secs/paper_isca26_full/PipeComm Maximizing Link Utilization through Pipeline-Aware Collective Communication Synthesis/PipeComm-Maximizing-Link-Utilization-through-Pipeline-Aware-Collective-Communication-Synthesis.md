# PipeComm: Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis

Ruifan Xu *Peking University* xuruifan@pku.edu.cn

Size Zheng *Peking University* zheng.size@bytedance.com

Yuze Luo *Peking University* luoyuze@stu.pku.edu.cn

> Meng Li *Peking University* meng.li@pku.edu.cn

Yuhao Meng *Peking University* mengyuhao@stu.pku.edu.cn

> Yun Liang§ *Peking University* ericlyun@pku.edu.cn

*Abstract*—Efficient collective communication is crucial for distributed training. While recent topology-aware synthesis approaches attempt to optimize communication based on the network topology, they struggle with heterogeneous environments where links exhibit distinct bandwidths and asymmetric connections. The main inefficiency arises from single-round communication: toward the end of a communication phase, only a few remaining transfers occupy limited links, leaving most network channels idle. Furthermore, existing methods fail to handle the congestion that naturally occurs under overlapping execution, leading to significant bandwidth underutilization.

To address these challenges, we propose PipeComm, a pipelineaware communication synthesis framework that maximizes bandwidth utilization by intelligently overlapping multiple data chunks. By explicitly modeling pipeline behavior, our method enables congestion-free scheduling across iterations and efficiently utilizes heterogeneous links. We develop an optimal synthesis method for constructing high-quality communication patterns and introduce a complementary incremental strategy that significantly improves scalability for large topologies. Experimental results show that the optimal strategy achieves over a 1.39× speedup compared to the state-of-the-art communication methods. Moreover, PipeComm supports diverse collective operations, demonstrating both efficiency and generality.

