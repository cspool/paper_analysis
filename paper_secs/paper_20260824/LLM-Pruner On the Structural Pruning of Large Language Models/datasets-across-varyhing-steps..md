# datasets across varyhing steps.

### 4.3 More Analysis

**Impact of Different Pruning Rates.** We investigate the impact of pruning the LLM at various pruning ratios in Figure 4. We compare our pruning results with the L2 strategy because L2 is also a data-free pruning algorithm. It is observed in the experiment of LLaMA that when the pruning ratio reaches approximately 20%, the magnitude-dependent algorithm experiences a rapid collapse, leading to the loss of information. Conversely, by employing LLM-Pruner, we are able to increase the pruning ratio to around 60% while achieving an equivalent perplexity level. Furthermore, in the case of Vicuna-7B, removing 10% parameters results in a performance decline equivalent to that of LLM-Pruner with 60%. The utilization of LLM-Pruner enables a significant increase in the number of model parameters that can be pruned, thereby substantially reducing computational overhead.

**Tuning on the External Dataset.** To tune the pruned model, we utilize the external dataset Alpaca [47]. The evaluation curves of the pruned model on two zero-shot datasets during the posttraining process are depicted in Figure 5. The results demonstrate a rapid decrease in the perplexity of the pruned model within 300 steps, followed by a gradual increase. We provide a more comprehensive evaluation in Appendix C.4. It is important to note that if the model is trained for an excessive number of steps, it runs the risk of overfitting the external dataset, potentially compromising its performance in other general-purpose tasks.

<span id="page-7-0"></span>Impact of Dependency-based Structured Pruning. To study the importance of dependency-based structural pruning, we conduct an experiment to disrupt dependencies within groups, where each weight matrix  $W_i$  is pruned solely based on the importance score estimated on itself. Table 6 presents the results demonstrating the impact of dependencies in structural pruning. In the absence of dependencies, the model nearly fails in the zero-shot generation and classification tasks. Even with tuning, the model fails to recover, showing a substantial difference compared to the results in dependency-based pruning.

**Impact of Different Aggregation Strategies.** We conduct tests on the aggregation algorithms proposed in Section 3.2. Our experimental results unveil notable discrepancies in model performance across different aggregation strategies, with particular emphasis on the 'Last-only' strategy. Among the evaluated approaches, the 'Max' strategy attains the most favorable outcomes in terms of perplexity, signifying enhanced coherence and fluency in sentence generation. However, it is important to note that the 'Max' strategy exhibits the poorest zero-shot classification results compared to all four strategies. Conversely, the 'Last-only' strategy showcases superior classification performance but

<span id="page-8-0"></span>Table 6: Effect of the dependency-based structural pruning. Average represents the average performance on 7 classification datasets.

|            | Method         | WikiText2↓ | PTB↓     | Average↑ |
|------------|----------------|------------|----------|----------|
| w/o Tuning | w/o dependency | 68378.42   | 79942.47 | 38.32    |
|            | w/ dependency  | 19.09      | 34.21    | 56.69    |
| w/ Tuning  | w/o dependency | 13307.46   | 13548.08 | 38.10    |
|            | w/ dependency  | 17.58      | 30.11    | 59.23    |

Table 7: Impact of different aggregation strategies on group importance estimation. Experiments are performed on LLaMA-7B.

| Method     | WikiText2↓ | PTB↓   | ARC-e↑ | PIQA↑ | OBQA↑ |
|------------|------------|--------|--------|-------|-------|
| Summation  | 66.13      | 164.25 | 40.70  | 63.49 | 34.80 |
| Max        | 62.59      | 144.38 | 39.60  | 63.71 | 34.60 |
| Production | 77.63      | 192.88 | 37.84  | 62.08 | 35.00 |
| Last-only  | 130.00     | 170.88 | 41.92  | 64.75 | 35.20 |

suffers from the poorest generation quality. In our experiments, we make a trade-off by selecting the 'Sum' strategy since it shows both good generalization quality and classification performance.

Comparison with DistilBERT We show the comparison results of DistilBERT and LLM-Pruner on LLaMA-7B in Table 8. LLM-Pruner outperforms DistilBERT by 4.24% on average with even a smaller size. The reason lies in that LLM-Pruner minimizes model disruption during pruning, whereas DistilBERT merely selects one layer out of two. As a result, the model pruned by LLM-Pruner demands less data to recover its performance compared with DistilBERT, consequently achieving superior performance.

<span id="page-8-1"></span>Table 8: DistilBert vs. LLM-Pruner. The average here means the average score on the above seven datasets.

| Pruning Ratio | #Param | Average |
|---------------|--------|---------|
| DistilBert    | 3.50B  | 44.64   |
| LLM-Pruner    | 3.35B  | 48.88   |

**Scratch Training vs. Pruning.** We compare LLM-Pruner with StableLM-3B<sup>4</sup> with a similar parameter size. To ensure fairness, both models are fine-tuned on the Alpaca dataset. The experimental results of these two models are shown in the Table 9. LLM-Pruner crafts lightweight LLMs with low resources, and even can sometimes achieve better performance than LLMs from scratch training. However, we also acknowledge that the LLaMA-3B obtained by LLM-Pruner will not always outperform other 3B models from scratch training, due to the huge gap in the size of training corpus.

Table 9: Scratch Training (StableLM-3B) vs. LLaMA-3B (by LLM-Pruner)

<span id="page-8-3"></span>

| Pruning Ratio   #Param                | Latency   BoolQ                  | PIQA | HellaSwag | WinoGrande     | ARC-e | ARC-c | OBQA   Average                 |
|---------------------------------------|----------------------------------|------|-----------|----------------|-------|-------|--------------------------------|
| StableLM-3B   3.6B<br>LLaMA-3B   3.6B | 31.69s   48.78<br>37.96s   61.41 |      |           | 54.62<br>55.01 |       |       | 27.40   45.84<br>37.40   50.30 |

Case Study. We provide some examples of sentences generated by the model compressed using LLM-Pruner in Table 10. We made efforts to ensure a minimal overlap between these generated sentences and the information contained in the tuning corpus, which demonstrates that the information originates from the original model rather than the tuning corpus. We provide additional examples in the Appendix, including the generated sentences of the model without post-training. From the cases in Table 10, it is evident that the sentences generated by the compressed model are comparable to those produced by the original model. They exhibit fluency, relevance, and informativeness regarding the given topic. Nevertheless, during our experiments, we observed that the pruned model's performance deviates from that of the original model, particularly when generating lengthy sentences. Occasionally, it may generate sentences that are meaningless or contain repetitive tokens.

## 5 Conclusion

In this paper, we propose LLM-Pruner, a structured pruning approach for large language models. LLM-Pruner aims to compress sizable language models in a task-agnostic manner while minimizing the dependency on the original training corpus and preserving the linguistic capabilities of LLMs. LLM-Pruner accomplishes this by iteratively examining each neuron within the model as a trigger for identifying dependency groups, thereby constructing the LLM's dependency graph. Subsequently, LLM-Pruner assesses the importance of these groups using both parameter-wise and weight-wise estimation. Finally, we utilize LoRA for fast recovery and adjustment of the pruned model. We evaluate the efficacy of LLM-Pruner on three distinct models—LLaMA, Vicuna, and ChatGLM—utilizing

<span id="page-8-2"></span><sup>&</sup>lt;sup>4</sup>https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b

<span id="page-9-0"></span>Table 10: Some examples of generations obtained with the original model and the pruned model.

| Model   | Original Model                                                                                                                                                                                                                                                                                                                                                                                                                                           | Pruned Model                                                                                                                                                                                                                                                                                                                                                                                                               |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LLaMa   | The universe is the entirety of space, time, matter, and energy<br>that exists. It includes all matter and energy, whether visible<br>or not, and in fact the whole universe is invisible. There are<br>many theories regarding its origin and ultimate fate                                                                                                                                                                                             | The universe is the entirety of space, time, matter, and<br>energy that exists. The laws that make up physics — elec<br>trons orbiting atoms, magnets attracting other objects,<br>planets orbiting the Sun—have been consistent over                                                                                                                                                                                      |
| Vicuna  | 10 steps to build an ios app: 1. Research and develop the idea.<br>2. Decide on the platform you want to use (iOS, Android,<br>or both). 3. Create a plan and budget. 4. Hire a team of<br>developers. 5. Design the user interface (UI). 6. Develop the<br>user interface (UI). 7. Test and iterate on the app. 8. Submit<br>the app to the app store. 9. Optimize the app for performance<br>and user experience (UX). 10. Market and promote the app. | 10 steps to build an ios app: 1. Plan your app: decide what<br>it should do, what it will look like, and what it will cost<br>to build it. 2. Hire a developer: A good iOS developer<br>will help you design the user interface, create the app, and<br>make sure it's bug-free. 3. Create a user interface: This is<br>where the magic happens. Your app's user interface will<br>be designed and mocked up at this point |
| ChatGLM | Write a poetry about love and peace. In the midst of life's<br>hustle and bustle, I find your love a soothing peace. It is a<br>feeling that cannot be conveyed, A connection that cannot<br>be distanced. With your smile, your eyes, your embrace, I<br>feel a sense of belonging that only you can give. It is a love<br>that lasts forever, A love that cannot be defeat                                                                             | Write a poetry about love and peace. Love and peace, two<br>such wonderful things\ A pure and endless desire in my<br>heart\ And both of them, I must seek for\ A long, long<br>time, I know\ Love, I know, is a feeling of being\ A<br>perfect partner, in every sense\ And peace, I need it, so<br>much, one day\ A long, long way, my heart will go                                                                     |

various zero-shot datasets. Our experimental results indicate that LLM-Pruner successfully prunes the model, reducing computational burden while retaining its zero-shot capabilities. Nevertheless, considerable performance degradation occurs when employing high pruning rates, such as the removal of 50% of LLaMA's parameters, resulting in a substantial decline in model performance. Additionally, we observe instances in which the model generates incoherent sentences. Addressing the challenges associated with compressing LLMs at higher pruning rates remains a challenging task.

