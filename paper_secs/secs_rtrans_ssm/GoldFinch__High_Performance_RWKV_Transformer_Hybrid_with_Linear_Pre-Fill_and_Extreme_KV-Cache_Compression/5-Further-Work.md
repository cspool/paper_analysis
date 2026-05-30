# **5 Further Work**

We anticipate updating this pre-print with further studies as results become available, including checkpoint upgrade results and evaluations, longer experiment training runs, and new long context experiments. Please check back for updates.

Most of the experiments done for this pre-print were performed over a short period of time on a single node containing 8 RTX 4090 cards. In the future we hope to demonstrate GoldFinch's performance on larger models with significantly more tokens.

We expect that GoldFinch will work similarly with other linear attention and SSM architectures in place of the Finch-C2 blocks. For example, it should be possible to implement a "GoldMamba" architecture in the same style.

Further work might explore increased memory reduction for the global KV-Cache via quantization, and application of ring attention [Liu et al.](#page-12-3) [\(2023\)](#page-12-3) to lower the memory requirements when extending to very long contexts. As a hybrid architecture model, GoldFinch will likely benefit from any future improvements to the RWKV and transformer architectures.

