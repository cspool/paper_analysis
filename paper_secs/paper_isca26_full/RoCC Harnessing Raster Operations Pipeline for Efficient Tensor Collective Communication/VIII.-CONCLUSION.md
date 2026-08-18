# VIII. CONCLUSION

We introduce RoCC which repurposes underutilized ROP units to enable fine-grained overlapping of collective communication (CC) and computation in multi-GPU deep learning computing. RoCC offloads CC to ROPs by mapping collective operations to ROP micro-operations and introducing a lightweight inter-ROP messaging method. RoCC delivers an average of 51% and 23% speedups for various LLMs over the baseline using SMs for both collectives and computations, and an oracle kernel fusion approach on 4 and 8 GPUs. In larger systems with 32 to 256 GPUs, RoCC consistently achieves speedups from 13% to 21%.

#### ACKNOWLEDGEMENTS

This work was supported by NSF grants CCF-2452081, CAREER-2341039, and CAREER-2047521. Part of this research was conducted using Pinnacles (NSF MRI, # 2019144) at the Cyber Infrastructure and Research Technologies (CIRT) at the University of California Merced, and Ampere® Altra® processors in servers donated by Ampere Computing.

