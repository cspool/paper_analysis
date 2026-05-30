# <span id="page-37-2"></span>**I Parameter Initializations**

Throughout this section, we use *l* to denote the layer index (layer *l* = 0 accepts input embeddings and layer *l* = *L*−1 produces output), and *i* the dimension index (*i* = 0,1,··· ,*D* −1). We set *r*<sup>0</sup> = *l L*−1 and *r*<sup>1</sup> = 1− *l L* as two parameters for simplicity.

The initialization of Eagle is provided as follows:

<span id="page-38-0"></span>

| METHOD             | MODEL                    | FINETUNED?                 | SLC  |
|--------------------|--------------------------|----------------------------|------|
|                    | neural-chat-7b-v3-3      | Yes - DPO                  | 0.57 |
|                    | Mistral-7B-Instruct-v0.2 | Yes - Instruct             | 0.35 |
| Open Generation    | Mistral-7B-v0.1          | No                         | 0.31 |
| Open Generation    | TinyLlama-1.1B-Chat-v1.0 | Yes - Vanilla and DPO      | 80.0 |
|                    | rwkv-4-world-7b          | Partially instruct trained | 0.40 |
|                    | v5-Eagle-7B-HF           | Partially instruct trained | 0.37 |
|                    | neural-chat-7b-v3-3      | Yes - DPO                  | 0.75 |
|                    | Mistral-7B-Instruct-v0.2 | Yes - Instruct             | 0.65 |
| Oracle-Selected    | Mistral-7B-v0.1          | No                         | 0.43 |
| Oracle-Selected    | TinyLlama-1.1B-Chat-v1.0 | Yes - Vanilla and DPO      | 0.36 |
|                    | rwkv-4-world-7b          | Partially instruct trained | 0.73 |
|                    | v5-Eagle-7B-HF           | Partially instruct trained | 0.70 |
|                    | neural-chat-7b-v3-3      | Yes - DPO                  | 0.59 |
|                    | Mistral-7B-Instruct-v0.2 | Yes - Instruct             | 0.25 |
| Induced Generation | Mistral-7B-v0.1          | No                         | 0.33 |
| maacca Generation  | TinyLlama-1.1B-Chat-v1.0 | Yes - Vanilla and DPO      | 0.17 |
|                    | rwkv-4-world-7b          | Partially instruct trained | 0.44 |
|                    | v5-Eagle-7B-HF           | Partially instruct trained | 0.57 |
|                    | neural-chat-7b-v3-3      | Yes - DPO                  | 0.74 |
| Evternal Promnt    | Mistral-7B-Instruct-v0.2 | Yes - Instruct             | 0.84 |
|                    | Mistral-7B-v0.1          | No                         | 0.37 |
| External Prompt    | TinyLlama-1.1B-Chat-v1.0 | Yes - Vanilla and DPO      | 0.22 |
|                    | rwkv-4-world-7b          | Partially instruct trained | 0.78 |
|                    | v5-Eagle-7B-HF           | Partially instruct trained | 0.65 |

<span id="page-38-1"></span>Table 15: Self-Learning Capability Evaluation.

| Dataset     | Eagle-7B | Raven-7b |
|-------------|----------|----------|
| Aggression  | 0.6587   | 0.4063   |
| MathQA      | 0.4760   | 0.4028   |
| Sarcasm     | 0.4679   | 0.4782   |
| TweetSent   | 0.5355   | 0.5541   |
| Unhealthy   | 0.2986   | 0.2834   |
| TweetStance | 0.3933   | 0.3070   |
| Spam        | 0.7290   | 0.4902   |
| ColBER      | 0.4088   | 0.2889   |
| CoLa        | 0.5285   | 0.4677   |
| TextEntail  | 0.7765   | 0.6137   |
| GoEmo       | 0.0956   | 0.0814   |
| PolEmo      | 0.5037   | 0.2639   |
| WNLI        | 0.5257   | 0.4638   |

Table 16: Eagle 7B and Raven 7B reasoning performance comparison based on subsets of selected datasets. The used metric is F1-macro (except for MathQA where accuracy is used instead).

#### • In the Time Mixing module:

- The token-shift coefficients of receptance and gate,  $\mu_r$  and  $\mu_g$ , are initialized to  $1 - \left(\frac{i}{D}\right)^{r_1/2}$  for i over dimension indices.
- The token-shift of key  $\mu_k$  is initialized to  $1 \left(\frac{i}{D}\right)^{r_1}$ .
- The token-shift of value  $\mu_{v}$  is initialized to  $1-\left(\frac{i}{D}\right)^{r_{1}}-0.3r_{0}$ .

   The time\_decay w is initialized to  $-6+5\left(\frac{i}{D-1}\right)^{0.7+1.3r_{0}}$ .

   The "time-first" u is initialized to  $r_{0}\left(1-\frac{i}{D-1}\right)+0.1((i+1) \mod 3)$ .

   The Time Mixing output matrix is initialized to 0.

| Parameters       | 0.4B               | 1.5B/1.6B          | 3B                   | 7B                   |
|------------------|--------------------|--------------------|----------------------|----------------------|
| Max LR           | $4 \times 10^{-4}$ | $3 \times 10^{-4}$ | $2 \times 10^{-4}$   | $1.5 \times 10^{-4}$ |
| Min LR           | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ | $1.5 \times 10^{-5}$ | $1 \times 10^{-5}$   |
| Micro Batch Size | 8                  | 8                  | 4                    | 9                    |
| GPU Count        | 24                 | 48                 | 48                   | 64                   |
| GPU Type         | A100               | A100               | A100                 | H800                 |
| Batch Size       | 786432             | 1572864            | 786432               | 2359296              |

Table 17: Learning Rate Hyperparameters for pretrained Eagle and Finch models

- The WKV GroupNorm weights are initialized with constant value  $((1+l)/L)^{0.7}$ .
- Two-dimensional parameters with the first dimension being larger than the second dimension are initialized with and orthogonal initialization of gain equal to the size of the first dimension divided by the size of the second dimension.
- Other parameters are initialized according to PyTorch default.
- In the Channel Mixing module:
  - The token-shift of both key  $\mu_k$  and receptance  $\mu_r$  are initialized to  $1 \left(\frac{i}{D}\right)^{r_1}$ .
  - The value and receptance matrices  $W_v$ ,  $W_r$  are initialized to 0.
  - Two-dimensional parameters with the first dimension being larger than the second dimension are initialized with and orthogonal initialization of gain equal to the size of the first dimension divided by the size of the second dimension.
  - All other parameters are initialized according to PyTorch default.
- The input embedding is initialized with a uniform distribution of  $\mathcal{U}(-maxLR, maxLR)$ , the maximum learning rate.
- The output head is initialized with an orthogonal initialization of gain 0.5.
- Bias is set to False for all linear layers.

In the Finch architecture, most of the parameters are initialized to the same as Eagle, except for a few changes.

In the Time Mixing block, there are several additional parameters initialized as follows:

- The token shift of input  $\mu_x$  and time decay  $\mu_w$  are initialized to  $1 \left(\frac{i}{D}\right)^{r_1}$ .
- The lora weights of A and B are initialized to uniform distribution of  $\mathcal{U}(-10^{-4}, 10^{-4})$ .

#### <span id="page-39-0"></span>J Architectural Ablations

Our improvements consist of architectural advances, a diverse multilingual corpus, and an optimized efficient tokenizer. To demonstrate that pure architectural advances indeed contribute to overall performance improvement, we ran an ablation where we train a 170 million parameter RWKV-6 model (which has 12 layers with dimension 768) from scratch on the Pile dataset using GPT-NeoX-20B tokenizer (vocabulary size V=50277), which yields 330 billion tokens in total. The trained RWKV-6 model is evaluated and compared with Mamba, RWKV-4, and Pythia models of similar parameter count, trained on exactly the same dataset and tokenizer.

| Model      | lmb.o | lmb.o | hella | piqa | sc16 | arc-e | arc-c | winG        | headqa | obqa  | sciq  | record | copa | avg  |
|------------|-------|-------|-------|------|------|-------|-------|-------------|--------|-------|-------|--------|------|------|
|            | ppl↓  | acc   | acc_n | acc  | acc  | acc   | acc   | acc         | acc    | acc_n | acc_n | acc    | em   | acc  |
| RWKV4-Pile | 29.2  | 33.1  | 32.2  | 64.9 | 59.1 | 47.1  | 23.9  | 51.5        | 28.3   | 29.4  | 77.2  | 61.9   | 64.0 | 47.7 |
| Pythia     | 24.4  | 38.8  | 31.7  | 62.6 | 58.4 | 45.3  | 24.0  | 52.0        | 28.7   | 29.0  | 76.5  | 66.3   | 62.0 | 47.9 |
| Mamba      | 16.0  | 44.2  | 35.3  | 64.4 | 60.4 | 48.1  | 24.3  | <b>52.4</b> | 28.8   | 28.6  | 78.1  | 68.9   | 68.0 | 50.1 |
| RWKV6-Pile | 16.1  | 44.5  | 34.9  | 64.4 | 60.7 | 48.4  | 24.7  | 51.9        | 29.3   | 29.6  | 80.6  | 69.3   | 70.0 | 50.7 |

Table 18: Ablation Results. Labels are the same from Table 4.

