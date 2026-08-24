# H Model Size and Training Details

We use xlm-roberta-large which has 355M parameters as the feature encoder f<sup>θ</sup> in LLMLingua-2. The training process takes approximately 23 hours on our MeetingBank compression dataset. For LLMLingua-2-small, the feature encoder is the multilingual-BERT which has 110M parameters. It takes roughly 16 hours to train the multilingual-BERT model.

## <span id="page-13-2"></span>I GPU Memory Usage

LLMLingua-2 enjoys a smaller GPU memory overhead because of its lightweight. The peak GPU memory usage of LLMLingua-2 on MeetingBank is only 2.1GB, while LLMLingua and Selective-Context, which utilize LLAMA-2-7B as the SLM, consume 16.6GB and 26.5GB of GPU memory, respectively.

