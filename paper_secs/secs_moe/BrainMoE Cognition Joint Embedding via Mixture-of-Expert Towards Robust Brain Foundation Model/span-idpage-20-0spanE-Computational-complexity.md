# <span id="page-20-0"></span>**E** Computational complexity

The limitation of BrainMoE is more computational complexity, as listed in Table 8. BrainMoE with 12 experts spends  $4\times$  more time than baselines. All-in-one with 36 experts nearly doubles the time cost.

<span id="page-20-1"></span><sup>&</sup>lt;sup>2</sup>https://www.ukbiobank.ac.uk/

<span id="page-20-2"></span>https://www.humanconnectome.org/

<span id="page-20-3"></span><sup>4</sup>https://www.humanconnectome.org/study/hcp-young-adult/overview

<span id="page-20-4"></span><sup>5</sup>https://adni.loni.usc.edu/

<span id="page-20-5"></span> $<sup>^6</sup> https://auckland.figshare.com/articles/dataset/NeurIPS\_2022\_Datasets/21397377$ 

<span id="page-20-6"></span> $<sup>^{7} \</sup>mathtt{https://github.com/Chrisa142857/brain\_moe}$ 

<span id="page-21-0"></span>Table 7: The comparison of experimental datasets between previous works.

|                   | BrainLM (2024) [23]  | BrainMass (2024) [34]     | BrainJEPA (2024) [11] | BrainMoE (Ours)              |
|-------------------|----------------------|---------------------------|-----------------------|------------------------------|
| Brain atlas       | AAL424               | C200                      | Schaefer400           | AAL116                       |
| Cognitive state   | resting, task-hariri | resting                   | resting               | resting, 11 types of tasking |
| Pre-train dataset | UKB, HCP             | UKB, HCP,<br>OpenNeuron   | UKB, HCP              | UKB, HCP                     |
| Pre-train data #  | 61,038               | 64,584                    | 40,162                | 68,251                       |
| Fine-tune dataset | UKB, HCP             | ASD, ADHD,<br>AD, PD, MDD | UKB, HCP, ADNI        | HCP, ASD,<br>AD, PD, SZ      |
| Parameter amount  | 650M                 | 34M                       | 307M                  | 709M                         |

<span id="page-21-1"></span>Table 8: Computational time cost of BrainMoE inference with two existing architectures and the all-in-one BrainMoE on the ABIDE dataset.

| Test time (ms/sample) | BrainMass | BrainJEPA |
|-----------------------|-----------|-----------|
| Single model          | 37.08     | 28.13     |
| BrainMoE              | 157.60    | 133.26    |
| All-in-one            | 287       | 7.21      |

