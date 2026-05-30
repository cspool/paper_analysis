# A Cross capabilities: additional details and experiments

## A.1 Further experiment details

Russian expert model training To enhance cross capabilities in mathematical skills for Russian, we train an additional expert specifically on Russian data. The expert training setup follows the same procedure outlined in [Section 3.1.](#page-4-1) For training data, we utilize the Russian subset of the multilingual dataset previously used for the multilingual expert, as described in [Section 3.2.](#page-4-2)

Merged models training As an additional baseline, we continually pretrain the strongest dense model, the seed model, on the same Russian math pretraining data used for the merged models. All experiments share the following training configuration, except for BTM and Model Soup which does not require further training:

- Learning rate schedule: we warm up from 0 to 5e − 6 over 1000 steps, then undergoes a cosine decay to 10% of the peak learning rate. The merged models are trained for a total of 2000 steps. One exception is the expert routing model is trained for 1000 steps in total with a constant learning rate of 5e-4. This was chosen upon tuning the hyperparameters.
- Batch size: we use a batch size of 1M tokens.
- Token count: All models were trained on 2B tokens of Russian mathematics over 2000 training steps. The exception is expert routing, which only trained on 1B tokens over 1000 steps, as we did not see performance improvement with further training.

## <span id="page-17-1"></span>A.2 Results on merging and upcycling models without in-domain data

|                  |       |       | Flores |           |
|------------------|-------|-------|--------|-----------|
|                  | GSM8K | En/Ru | Ru/En  | Ru-MGSM   |
| Dense models     |       |       |        |           |
| Seed Model       | 10.5  | 22.8  | 32.8   | 12.8      |
| Math Expert      | ∗20.5 | 10.2  | 28.9   | 10.8      |
| Russian Expert   | 9.48  | ∗32.3 | ∗34.6  | 9.60      |
| Expert upcycling |       |       |        |           |
| BTX Sample       | 18.3  | 30.4  | 34.0   | 10.0      |
| BTX Soft         | 18.0  | 30.0  | 33.9   | 12.4      |
| BAM              | ∗20.5 | 30.6  | 34.5   | 10.8      |
| Expert merging   |       |       |        |           |
| Model Soup       | 17.5  | 14.7  | 32.3   | ∗<br>13.2 |
| BTM              | ∗20.5 | ∗32.3 | ∗34.6  | 9.60      |
| Expert Routing   | ∗20.5 | 32.3  | ∗34.6  | 9.60      |
| BAM Adapter      | 18.1  | 30.9  | 34.1   | 12.8      |
| BTS              | 19.0  | 31.6  | 33.0   | 10.0      |

Table 9 Cross capability performance of merged models without in-domain data. We evaluate the seed model, Russianlanguage, and Math experts on Russian MGSM [\(Shi et al.,](#page-15-10) [2022\)](#page-15-10) and compare performance with merged and upcycled models. We do not use any in-domain training data during the merging or upcycling training process. The results indicate that a small amount of cross capability data is necessary for merged or upcycled models to effectively learn cross capabilities.

In [Table 9,](#page-17-1) we show results on merging and upcycling models without in-domain data. The merging phase is instead trained on a data mixture composed of 50% of math expert and 50% of Russian expert's continue pretraining data mixture.

We observe that despite being trained with more tokens during the merging phase, all baseline methods does not significantly outperform the seed model on the cross capability task Russian MGSM. This indicates that in-distribution data is essential.

