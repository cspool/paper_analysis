# <span id="page-15-2"></span>A.2.1 LANGUAGE

For finetuning on SuperGLUE, we generally adopt the conventional setup (Raffel et al., 2020; Narang et al., 2021) where we finetune on all SuperGLUE tasks, in a proportional mix, for 200K steps with a batch size of 128. Each example has input length 512 and target decoding length of 62. In our figures, we report the average SuperGLUE accuracy score across 3 runs for each data point.

For finetuning Dense models on SuperGLUE, we use Adafactor with the default, constant learning rate of  $10^{-3}$  and a dropout rate of 0.1 (Raffel et al., 2020; Narang et al., 2021). For finetuning upcycled models, because there are many more parameters, it can be helpful to increase the dropout rate for the experts (Fedus et al., 2022), while using the default dropout rate of 0.1 for all "dense" parameters. For upcycled Base models, we obtained the strongest results for a constant learning rate of  $10^{-4}$  with an expert dropout rate of 0.1. For upcycled Large models, we found slightly stronger results with a learning rate of  $10^{-3}$  and an expert dropout rate of 0.3. Decreasing (or increasing) the learning rate was not helpful for the Dense Base or Large models.

#### <span id="page-15-0"></span>A.2.2 Vision

**Few-shot linear evaluation**. The fewshot evaluation poses classification as a linear regression task, where inputs are frozen representations computed by a pretrained model, and outputs are one-hot vectors representing the ground truth (Dosovitskiy et al., 2021). There are two key changes compared to prior works which used this method (Dosovitskiy et al., 2021; Riquelme et al., 2021):

- Multiple seeds. The evaluation involves randomly selecting N examples per class. To reduce dependency on that choice, we run 5 random seeds, and report the average test accuracy across them.
- Fixed L2 regularization. Prior works considered a range of L2 regularizations. The optimal value was picked based on average test accuracy across all datasets considered for few-shot evaluation. We fix the L2 regularisation at 1024.

**Full finetuning**. We finetune models on ImageNet2012 using SGD and a batch size of 512. We use a cosine decay learning rate schedule with a linear warmup. We sweep over two training schedules: (i) 5k steps, with a warmup of 200 steps, and (ii) 10k steps, with a warmup of 400 steps. Alongside this we sweep over learning rates [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]. For each pretrained model, there are therefore 12 finetuning sweeps; we select based on optimal validation accuracy, and report the corresponding test accuracy.

