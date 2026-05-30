# BTS: Harmonizing Specialized Experts into a Generalist LLM

Qizhen Zhang1,2,<sup>∗</sup> , Prajjwal Bhargava1,<sup>⋄</sup> , Chloe Bi1,<sup>⋄</sup> , Chris X. Cai1,<sup>⋄</sup> , Jakob Foerster1,2,<sup>⋄</sup> , Jeremy Fu1,<sup>⋄</sup> , Punit Singh Koura1,<sup>⋄</sup> , Ruan Silva1,<sup>⋄</sup> , Sheng Shen1,<sup>⋄</sup> , Emily Dinan1,† , Suchin Gururangan1,† , Mike Lewis1,†

We present Branch-Train-Stitch (BTS), an efficient and flexible training algorithm for combining independently trained large language model (LLM) experts into a single, capable generalist model. Following [Li et al.](#page-15-0) [\(2022\)](#page-15-0), we start with a single "seed" language model which is branched into domain-specific (e.g., coding or math) experts with continual pretraining. BTS combines experts into a generalist model using lightweight stitch layers, which are inserted between frozen experts and the seed LLM, and trained on a small datamix of the expert domains. Stitch layers enable the seed LLM to integrate representations from any number of experts during the forward pass, allowing it to generalize to new domains, despite remaining frozen. Because BTS does not alter the constituent LLMs, BTS provides a modular and flexible approach: experts can be easily removed and new experts can be added with only a small amount of training. Compared to alternative model merging approaches, BTS yields the best generalist performance on a variety of downstream tasks, retaining the specialized capabilities of each of the experts.

Date: February 4, 2025

Correspondence: Qizhen Zhang at [qizhen.zhang@eng.ox.ac.uk](mailto:qizhen.zhang@eng.ox.ac.uk)

![](_page_0_Picture_8.jpeg)

