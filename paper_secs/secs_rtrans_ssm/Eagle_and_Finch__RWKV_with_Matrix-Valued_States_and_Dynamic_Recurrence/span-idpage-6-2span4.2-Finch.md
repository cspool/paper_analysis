# <span id="page-6-2"></span>**4.2 Finch**

#### <span id="page-6-3"></span>**4.2.1 Finch Token Shift**

The **d**ata-**d**ependent **l**inear int**erp**olation (ddlerp) between *x<sup>t</sup>* and *xt*−<sup>1</sup> used in Finch Token Shift is defined as:

$$lora_{\square}(x) = \lambda_{\square} + tanh(xA_{\square})B_{\square}$$
(14)

$$ddlerp_{\square}(a,b) = a + (b-a) \odot lora_{\square}(a + (b-a) \odot \mu_x)$$
(15)

where  $\mu_x$  and each  $\lambda_\square$  introduce a trainable vector of dimension D and each  $A_\square \in \mathbb{R}^{D \times 32}$ ,  $B_\square \in \mathbb{R}^{32 \times D}$  introduce new trainable weight matrices. For the special case of  $LoRA_\omega$  seen below we introduce double-sized trainable weight matrices  $A_\omega \in \mathbb{R}^{D \times 64}$ ,  $B_\omega \in \mathbb{R}^{64 \times D}$ . A schematic representation can be found in Figure 1, bottom-right. Please note that future 7B and larger Finch models are expected to further increase the size of these weight matrices by double or more.

This new form of Token Shift enhanced with data-dependence is intended to expand the abilities of the model beyond the RWKV-4/Eagle style of Token Shift so that the amount of new and old data allocated per channel now depends on the input at both current and prior time steps.

#### <span id="page-7-0"></span>4.2.2 Finch Time Mixing

$$\Box_t = \mathrm{ddlerp}_{\Box}(x_t, x_{t-1}) W_{\Box}, \quad \Box \in \{r, k, \nu, g\}$$
 (16)

$$d_t = \text{lora}_d(\text{ddlerp}_d(x_t, x_{t-1})) \tag{17}$$

$$w_t = \exp(-\exp(d_t)) \tag{18}$$

$$wkv_t = \operatorname{diag}(u) \cdot k_t^{\mathrm{T}} \cdot v_t + \sum_{i=1}^{t-1} \operatorname{diag}\left( \bigodot_{j=i+1}^{t-1} w_j \right) \cdot k_i^{\mathrm{T}} \cdot v_i \in \mathbb{R}^{(D/h) \times (D/h)}$$
(19)

$$o_t = \operatorname{concat}\left(\operatorname{SiLU}(g_t) \odot \operatorname{LayerNorm}(r_t \cdot wkv_t)\right) W_o \in \mathbb{R}^D$$
 (20)

The  $wkv_t$  attention calculation can alternatively be written in a recurrent manner:

$$wkv' = s + \operatorname{diag}(u) \cdot k^{\mathrm{T}} \cdot v \tag{21}$$

$$s' = \operatorname{diag}(w) \cdot s + k^{\mathrm{T}} \cdot v \tag{22}$$

Unlike in Eagle,  $w_t$  here is not static across the sequence (dashed arrows in Figure 1, left and topright.). This is the core change to decay in Finch, as each channel of  $w_t$  can now vary independently over time, in a data-dependent manner, whereas previously it was a fixed learned vector.

The new LoRA mechanisms above are used to take learned vectors, as seen in Eagle, and inexpensively augment them with additional offsets determined by the incoming input. Note that the LoRA process itself uses an Eagle style Token-Shifted value as its input, not just the latest token. The new time-varying decay  $w_t$  goes one step further, applying LoRA again afterward. Intuitively, this is a second-order variant of Token-Shifting, allowing each channel of  $w_t$  to vary based on a mix of the current and prior tokens, with the mix itself determined by aspects of both tokens.

## <span id="page-7-1"></span>5 RWKV World Tokenizer

Tokenization is important in language modelling as it conditions the learning relationships between tokens and the generation of new text based on those patterns. The numbers of tokens to build a single semantic chunk are, however, often very unequally distributed against non-European and other underrepresented languages. Byte-pair-encoding (BPE) based tokenizers which are trained with this inequality result in not only lower performances against underrepresented languages but also undue economic costs such as inference Ahia et al. (2023) and continual pre-training with extended vocabulary Lin et al. (2024); Sasaki et al. (2023). To address these problems, we manually select tokens from multiple vocabulary files such that non-European languages are well represented.

To construct the tokenizer's vocabulary, we merge the vocabularies of the following tokenizers and then manually select the tokens for non-European languages.

- GPT-NeoX-20B (Black et al., 2022): https://huggingface.co/EleutherAI/gpt-neox-20b
- GPT2 (Radford et al., 2019): https://huggingface.co/openai-community/gpt2
- cl100k\_base of tiktoken: https://github.com/openai/tiktoken
- Llama2 (Touvron et al., 2023): https://huggingface.co/meta-llama/Llama-2-7b-hf

• **Bloom [\(Workshop et al.,](#page-24-0) [2023\)](#page-24-0):** [https://huggingface.co/bigscience/](https://huggingface.co/bigscience/bloom) [bloom](https://huggingface.co/bigscience/bloom)

This tokenizer has a vocabulary size of *V* = 65536, numbered from 0 through 65535, where tokens are arranged by their lengths in bytes. Below is a brief overview:

- **Token 0:** Represents the boundary between text documents, known as <EOS> or <SOS>. This token doesn't encode any specific content and is only used for document separation.
- **Tokens 1-256:** Consist of byte encodings (Token *k* encodes byte *k* −1), wherein tokens 1-128 correspond to standard ASCII characters.
- **Tokens 257-65529:** Tokens with a minimum length of 2 bytes in UTF-8, including words, prefixes and suffixes, accented letters, Chinese characters, Hangul, Hiragana, Katakana and emojis. For example, Chinese characters are allocated from token 10250 to 18493.
- **Token 65530-65535:** Reserved tokens for future use.

These designations are intended to enhance the tokenizer's efficiency on the multilingual corpus, as well as on source code of programming languages.

This tokenizer is implemented via a Trie (Prefix Tree) to boost speed while maintaining simplicity. Encoding is performed as matching the longest element in vocabulary with an input string from left to right. We note that our tokenizer's vocabulary construction is to mitigate *undue* burden, which naive BPE and related methods cause, on minor languages.

