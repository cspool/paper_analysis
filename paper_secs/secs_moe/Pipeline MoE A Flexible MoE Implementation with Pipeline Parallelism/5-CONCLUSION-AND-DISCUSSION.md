# 5 CONCLUSION AND DISCUSSION

In this work, we introduce PPMoE, a novel MoE framework that efficiently solves two critical drawbacks of DPMoE. By replacing communication-intensive all-to-all dispatching and gathering of DPMoE with a simple tensor slicing and an inner-node all-reduce, PPMoE largely eases the communication burden of MoE models. By incorporating with pipeline parallel, PPMoE enhances the scalability of the backbone of MoE models. Experimental results have demonstrated the effectiveness of the proposed framework.

Our work paves a new step towards decoupling expert parallel from data parallel and eliminating the communication-intensive all-to-all operations from expert parallel. We expect that in the future with more powerful computational resources and parallel computing techniques, MoE models with reduced communication burden and increased configuration flexibility plays a more significant role in building large language models and their variants for better training/validation efficiency and representative capability.

