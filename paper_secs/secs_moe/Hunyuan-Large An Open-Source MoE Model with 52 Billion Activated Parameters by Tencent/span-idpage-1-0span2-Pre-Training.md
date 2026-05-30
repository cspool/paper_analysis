# <span id="page-1-0"></span>2 Pre-Training

In this section, we will describe the details of pre-training Hunyuan-Large, including (a) data and tokenizer, where high-quality data largely contributes to the model performance, (b) model structure, consisting of our proposed KV cache compression, expert routing, and expert-specific learning rate scaling strategies, and (c) pre-training recipes, introducing the detailed pre-training schedule as well as our guidebook of explorations on MoE scaling laws. These techniques build the foundation of Hunyuan-Large's remarkable capability in pre-training.

