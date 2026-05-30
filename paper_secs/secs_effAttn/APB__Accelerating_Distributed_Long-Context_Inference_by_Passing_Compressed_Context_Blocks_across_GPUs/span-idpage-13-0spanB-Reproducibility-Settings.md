# <span id="page-13-0"></span>B Reproducibility Settings

#### B.1 Training of the Compressor

We use the retaining heads proposed in [Huang et al.](#page-9-10) [\(2024\)](#page-9-10) as our compressor. Following [Huang et al.](#page-9-10) [\(2024\)](#page-9-10), we set the intermediate size of the retaining heads to 1024. We utilize the first 3000 samples from the LongAlign [\(Bai et al.,](#page-9-21) [2024\)](#page-9-21) dataset to generate the training labels, and we train the retaining heads with the a frozen model backbone for 3000 steps in 1 epoch. The batch size set is 1 and the maximum input length is set to 10240. We use AdamW as the optimizer and the learning rate is set to 5e-4, with β<sup>1</sup> set to 0.9 and β<sup>2</sup> set to 0.95. We apply a linear scheduler whose number of warmup steps is set to 300. Notably, the training loss of the retaining head consists of a regression loss and a smoothing loss, following the setting of [\(Huang et al.,](#page-9-10) [2024\)](#page-9-10), and the balance factor α is set to 0.0025. The gradient clipping value is set to 0.5 to avoid gradient explosion.

<span id="page-13-1"></span>

| n  | 32K  | 64K | 128K | 256K | 512K |
|----|------|-----|------|------|------|
| lb | 4K   | 8K  | 16K  | 32K  | 64K  |
| la | 1K   | 2K  | 4K   | 8K   | 8K   |
| lp | 0.5K | 1K  | 2K   | 4K   | 8K   |

Table 8: Hyperparameters of APB used in Section [4.3.](#page-6-2) n stands for input length, l<sup>b</sup> is the block size, l<sup>a</sup> represents the anchor length, and l<sup>p</sup> represents the passing length. "K" is an abbreviation for 1024.

#### B.2 Inference Hyperparameters

Here, we elaborate on the details of the inference hyperparameters used in Section [4.2,](#page-5-2) [4.3,](#page-6-2) and [4.4.](#page-6-3)

#### B.2.1 End-to-End Benchmark (Section [4.2\)](#page-5-2)

In the end-to-end benchmark, we first test the performance of 3 baselines with the performance of our method. Since methods without approximate attention do not alter the computational outcome of the attention mechanism, we take the result of ULYSSES as the performance benchmark for FUL-LATTN. Then, we evaluate the inference speed of all baselines along with APB .

Performance Evaluation. In the ∞Bench experiments, we evaluate the performance of each method by running all the data from each task. Details of the tasks can be found in [Zhang et al.](#page-11-13) [\(2024b\)](#page-11-13). When conducting the RULER experiments, we generate 500 test samples for each RULER task to evaluate the performance, with details introduced in [Hsieh et al.](#page-9-19) [\(2024\)](#page-9-19). We report the performance of each method under a 128K input length for both benchmarks. For methods with sequence parallelism, we set the sequence parallel size to 8. We use a single machine with 8 GPUs for Llama-3.1-8B and Qwen-2.5-14B, while for Yi-34B, due to its large model size, we employ two machines with layers evenly distributed between them. All the answers are generated with greedy decoding, i.e. with a temperature set to 0. For STARATTN, we set both the block length and the anchor length to 16K. For MINFERENCE, we use the official head configuration for Llama-3.1-8B and generate the configurations for Qwen-2.5-14B and Yi-34B based on the first data entry of SG1 from RULER. For APB, we set anchor length l<sup>a</sup> to 4K and passing length l<sup>p</sup> to 2K. We place the query tokens after the system prompt to embed the query within the anchor blocks without disrupting the model's chat template.

Speed Evaluation. We run the first 20 samples for each task and average the inference speed. To reflect the speed of LLMs in processing a query, we define the inference speed as follows.

$$speed = \frac{\#input \ tokens + \#output \ tokens}{prefill \ time + decoding \ time}$$

We keep all the hyperparameters and distribution settings same as the performance evaluation.

