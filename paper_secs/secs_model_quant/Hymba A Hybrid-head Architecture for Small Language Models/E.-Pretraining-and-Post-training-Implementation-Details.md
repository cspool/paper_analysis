# **E. Pretraining and Post-training Implementation Details**

**Pretraining settings.** We train Hymba-125M/350M/1.5B models on 1.3T tokens, using a mix of DCLM-Baseline-1.0 [\[36\]](#page-11-18), SmolLM-Corpus [\[37\]](#page-11-19), and an internal high-quality dataset for 1T, 250B, and 50B tokens, respectively. We adopt the WSD learning rate scheduler [\[38\]](#page-11-20) with three phases: (1) warmup steps set to 1% of the total steps, (2) a stable phase maintaining the peak learning rate of 3e-3, and (3) a decay phase reducing the learning rate to 1e-5 over 20% of the total steps, while gradually annealing to smaller, higher-quality datasets like SmolLM-Corpus and the internal dataset. We use a sequence length of 2K and a batch size of 2M tokens throughout the training process, which is conducted on 128 NVIDIA A100 GPUs. Details of Hymba-125M/350M/1.5B models are shown in Tab. [11.](#page-20-0)

We also show the training curves of Hymba-1.5B in Fig. [14.](#page-19-1)

**Implementation details of post-training.** We post-trained our 1.5B base model with a two-stage strategy: the first full-finetuning (FFT) stage and another direct preference optimization (DPO) [\[12\]](#page-10-11) training. The learning rates are 5e-5, and 3e-6 for FFT and DPO, respectively. Both FFT and DPO training are carried out for one epoch with a cosine scheduler. The global batch size is set to 1024. To accelerate training, we follow the training recipe [\[61,](#page-13-0) [62,](#page-13-1) [63\]](#page-13-2) to pack the samples and use a block size of 2048. We implement the finetuning and DPO training with the LMFlow toolkit [\[62\]](#page-13-1). In addition to full-finetuning, we also leverage Dora [\[13\]](#page-10-12) to do parameter-efficient finetuning.

**Baselines and downstream tasks.** We compare Hymba-1.5B-Instruct with competitive lightweight instruction-tuned models, i.e., Llama-3.2-1B-Instruct [\[42\]](#page-12-3), OpenELM-1-1B-Instruct [\[51\]](#page-12-12),

<span id="page-20-0"></span>Table 11 | Architecture details of Hymba models of different size.

| Attribute      | 125M | 350M | 1.5B  |
|----------------|------|------|-------|
| Blocks         | 24   | 32   | 32    |
| Hidden Size    | 512  | 768  | 1600  |
| SSM State      | 16   | 16   | 16    |
| Attn. Heads    | 8    | 12   | 25    |
| Query Groups   | 4    | 4    | 5     |
| Num. Full Attn | 3    | 3    | 3     |
| Window Size    | 1024 | 1024 | 1024  |
| MLP Hidden     | 1664 | 2432 | 5504  |
| Tie Embedding  | True | True | True  |
| Parameters     | 125M | 350M | 1.52B |

Qwen2.5-1.5B-Instruct [\[64\]](#page-13-3), and SmolLM-1.7B-Instruct [\[43\]](#page-12-4). We test the instruction-tuned models on MMLU (5-shot), IFEval, GSM8K (5-shot), GPQA (0 shot), and Berkeley Function-Calling Leaderboard v2 (BFCLv2) [\[65\]](#page-13-4). For BFCLv2, we use the official code from Gorilla project [\[65\]](#page-13-4) and evaluate the BFCLv2 live category, including *live\_simple*, *live\_multiple*, *live\_parallel*, *live\_parallel\_multiple*, *live\_relevance*. We exclude *live\_irrelevance*, since we found some baseline models without function calling abilities, could achieve high in the *live\_irrelevance* category (where the model is not required to call function) and very low in other tasks, but still got high overall accuracy although these models are not helpful at all. For the remaining tasks, we directly use the lm-evaluation-harness [\[91\]](#page-14-12).