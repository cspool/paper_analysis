# <span id="page-18-0"></span>**C. Pre-training Details of the Llama3.1 Draft Model**

<span id="page-18-3"></span>To serve as the draft model for LLaMA3.1-8b in TriForce, we pretrain a tiny version of 250M parameters with the same tokenizer from LLaMA3.1-8b. The model configuration is listed in Table [11.](#page-18-3) We train the model on Wikipedia (20231101.en) [5](#page-18-4) and part of C4-en[6](#page-18-5) for 1 epoch.

Table 11. Configuration of Llama 3.1 205M.

| hidden size             | 768    |
|-------------------------|--------|
| hidden act              | silu   |
| intermediate size       | 3072   |
| max position embeddings | 2048   |
| num attention heads     | 12     |
| num key value heads     | 12     |
| rope theta              | 500000 |
| vocab size              | 128256 |
|                         |        |

<span id="page-18-4"></span><sup>5</sup><https://huggingface.co/datasets/wikimedia/wikipedia>

<span id="page-18-5"></span><sup>6</sup><https://huggingface.co/datasets/allenai/c4>

