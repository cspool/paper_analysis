# <span id="page-34-0"></span>**F New Tokenizer Details**

#### <span id="page-34-1"></span>**F.1 Designation**

To construct the tokenizer's vocabulary, we merge the vocabularies of the following tokenizers and then manually select the tokens for non-European languages.

- **GPT-NeoX-20B [\(Black et al.,](#page-17-5) [2022\)](#page-17-5):** [https://huggingface.co/EleutherAI/](https://huggingface.co/EleutherAI/gpt-neox-20b) [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)
- **GPT2 [\(Radford et al.,](#page-22-5) [2019\)](#page-22-5):** [https://huggingface.co/openai-community/](https://huggingface.co/openai-community/gpt2) [gpt2](https://huggingface.co/openai-community/gpt2)
- **cl100k\_base of tiktoken:** <https://github.com/openai/tiktoken>
- **Llama2 [\(Touvron et al.,](#page-23-5) [2023\)](#page-23-5):** [https://huggingface.co/meta-llama/](https://huggingface.co/meta-llama/Llama-2-7b-hf) [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Bloom [\(Workshop et al.,](#page-24-0) [2023\)](#page-24-0):** [https://huggingface.co/bigscience/](https://huggingface.co/bigscience/bloom) [bloom](https://huggingface.co/bigscience/bloom)

This tokenizer has a vocabulary size of *V* = 65536, numbered from 0 through 65535, where tokens are arranged by their lengths in bytes. Below is a brief overview:

- **Token 0:** Represents the boundary between text documents, known as <EOS> or <SOS>. This token doesn't encode any specific content and is only used for document separation.
- **Tokens 1-256:** Consist of byte encodings (Token *k* encodes byte *k* −1), wherein tokens 1-128 correspond to standard ASCII characters.
- **Tokens 257-65529:** Tokens with a minimum length of 2 bytes in UTF-8, including words, prefixes and suffixes, accented letters, Chinese characters, Hangul, Hiragana, Katakana and emojis. For example, Chinese characters are allocated from token 10250 to 18493.
- **Token 65530-65535:** Reserved tokens for future use.

These designations are intended to enhance the tokenizer's efficiency on the multilingual corpus, as well as on source code of programming languages.

#### <span id="page-34-2"></span>**F.2 Efficiency Experiments**

We test the tokenizer along with Llama2 tokenizer, GPT2's cl50k\_base and GPT4's cl100k\_base on five different languages and programming code. For the five natural languages, we select the first 3GB of data from the CulturaX [\(Nguyen et al.,](#page-21-12) [2023\)](#page-21-12) dataset, and we use StarCoder [\(Li et al.,](#page-20-9) [2023b\)](#page-20-9) for code. The efficiency is measured with the number of tokens and the average character length per token. A tokenizer is considered more efficient if it tokenizes a document in less tokens or having longer average character length per token.

The results are presented in Table [12.](#page-35-4) Generally, our tokenizer is as efficient as GPT4's cl100k\_base tokenizer, and surpasses it on three non-European languages, despite having a smaller vocabulary size (65536 vs 100256).

<span id="page-35-4"></span>

| Language              | English                  |              | Chinese                |              | Arabic                 |              |  |
|-----------------------|--------------------------|--------------|------------------------|--------------|------------------------|--------------|--|
| Num. of chars         | 3918475074               |              | 1056687183             |              | 1765106557             |              |  |
| Tokenizer             | tokens                   | avg len      | tokens<br>avg len      |              | tokens                 | avg len      |  |
| cl50k_base            | 874341786                | 4.48         | 2019239404             | 0.52         | 1722145732             | 1.02         |  |
| cl100k_base           | 855585969                | 4.58         | 1241767292             | 0.85         | 1219229554             | 1.44         |  |
| llama2                | 1016595271               | 3.85         | 1524486994             | 0.69         | 1569786022             | 1.12         |  |
| RWKV vocab            | 878861532                | 4.46         | 997736792              | 1.06         | 1133572680             | 1.56         |  |
|                       |                          |              |                        |              |                        |              |  |
| Language              | Hindi                    |              | Spanish                |              | Code                   |              |  |
| Num. of chars         | 1837327906               |              | 3047372943             |              | 1046274579             |              |  |
| Tokenizer             | tokens                   | avg len      | tokens                 | avg len      | tokens                 | avg len      |  |
|                       |                          |              |                        |              |                        |              |  |
| cl50k_base            | 2637636307               | 0.69         | 1061207448             | 2.87         | 461240625              | 2.27         |  |
| cl100k_base<br>llama2 | 1721299552<br>1883783695 | 1.06<br>0.97 | 831382965<br>938883427 | 3.67<br>3.25 | 269124622<br>369239882 | 3.89<br>2.83 |  |

Table 12: Comparison of tokenization efficiency across five different languages and code.

#### <span id="page-35-0"></span>**F.3 Speed**

The speed of the tokenizer is also an important factor, especially when facing corpus with trillions of tokens, where the tokenizer's speed is likely to become a bottleneck. We conducted experiments to compare the tokenization speeds among common tokenizers. We used Wikipedia's 20220301.en corpus [\(Wikimedia-Foundation,](#page-24-2) [2022\)](#page-24-2) to conduct this test, which is run on an M2 Mac mini machine. The comparison standard is the tokenization speed of the original corpus, expressed in MB/s, to mitigate the impact of the vocabulary size. The results show that the Rust implementation of the RWKV tokenizer has extremely high speed of 90.32 MB per second, and is 9.6 times faster than OpenAI's Tiktoken at the second place. Even comparing with only Python implementations, The original Python implementation of RWKV's tokenizer is significantly faster than Llama2's tokenizer. The experimental results are shown in Table [13.](#page-35-3)

<span id="page-35-3"></span>

| Tokenizer                    | Type            | Speed (MB/s) |
|------------------------------|-----------------|--------------|
| RWKV tokenizer (Rust)        | Greedy matching | 90.32        |
| Tiktoken o200k_base          | BPE             | 9.34         |
| RWKV tokenizer (Python)      | Greedy matching | 5.31         |
| BERT (Devlin et al., 2019)   | WordPiece       | 3.44         |
| Mistral (Jiang et al., 2023) | BPE             | 2.41         |
| Llama2                       | BPE             | 2.40         |

Table 13: Comparison of tokenizer speeds.

