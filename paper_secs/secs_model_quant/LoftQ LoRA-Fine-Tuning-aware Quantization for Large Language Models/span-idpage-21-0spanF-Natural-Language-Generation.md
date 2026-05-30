# <span id="page-21-0"></span>F Natural Language Generation

<span id="page-21-1"></span>We set the batch size as 32 for WikiText-2 and 16 for GSM8K. We train 2 epochs on WikiText-2 and 6 epochs on GSM8K. We select learning rate from{1×10−<sup>5</sup> *,*5×10−<sup>5</sup> *,*7×10−<sup>5</sup> *,*1×10−<sup>4</sup> *,,*3×10−<sup>4</sup> *,*4×10−<sup>4</sup> }. Specific settings are summarized in Table [16](#page-21-1) and Table [17.](#page-21-2)

Table 16: Hyper-parameter setup of LoftQ LLAMA-2-series on GSM8K

| Model       | Hyperparameter | NF4            | NF2            | Mixed-precision |
|-------------|----------------|----------------|----------------|-----------------|
| LLAMA-2-7b  | learning rate  | 10−4<br>×<br>3 | 10−4<br>×<br>3 | 10−4<br>×<br>3  |
| LLAMA-2-13b | learning rate  | 10−4<br>×<br>1 | 10−4<br>×<br>1 | 10−4<br>×<br>3  |

<span id="page-21-2"></span>Table 17: Hyper-parameter setup of LoftQ LLAMA-2-series on WikiText-2

| Model       | Hyperparameter | NF4            | NF2            | Mixed-precision |
|-------------|----------------|----------------|----------------|-----------------|
| LLAMA-2-7b  | learning rate  | 10−4<br>×<br>3 | 10−4<br>×<br>3 | 10−4<br>×<br>3  |
| LLAMA-2-13b | learning rate  | 10−4<br>×<br>1 | 10−4<br>×<br>1 | 10−4<br>×<br>3  |

