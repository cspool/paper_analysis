# <span id="page-7-2"></span>9 Limitations

While our proposed RWKV model has demonstrated promising results regarding training and memory efficiency during inference, some limitations should be acknowledged and addressed in future work.

First, the linear attention of RWKV leads to significant efficiency gains but still, it may also limit the model's performance on tasks that require recalling minutiae information over very long contexts. This is due to the funneling of information through a single vector representation over many time steps, compared with the full information maintained by the quadratic attention of standard Transformers. In other words, the model's recurrent architecture inherently limits its ability to "look back" at previous tokens, as opposed to traditional self-attention mechanisms. While learned time decay helps prevent the loss of information, it is mechanistically limited compared to full selfattention.

Another limitation of this work is the increased importance of prompt engineering in comparison to standard Transformer models. The linear attention mechanism used in RWKV limits the information from the prompt that will be carried over to the model's continuation. As a result, carefully designed prompts may be even more crucial for the model to perform well on tasks.

The above RWKV property was confirmed by studies on prompt engineering presented in Appendix [L.](#page-24-0) By changing the order of the information

pieces, we were even able to almost double the RWKV performance for some tasks.

