# <span id="page-10-2"></span>G Scaling language modeling to 400K context

As shown in Figure 6, FocusLLM maintains a low perplexity even with a context length of 400K. Note

that the number of candidate tokens corresponding to 400K is 200, which is far greater than the number of candidate tokens seen during training. This demonstrates that FocusLLM has strong extrapolation capabilities. We can effectively scale to lengths greater than 400K by either using longer sequences during training or by employing a base model with a default context length, which we plan to explore in future work.

