# <span id="page-15-0"></span>4.2 Distillation

Deploying massive neural networks with billions, or trillions, of parameters is inconvenient. To alleviate this, we study distilling [\(Hinton et al.,](#page-36-3) [2015\)](#page-36-3) large sparse models into small dense models. Future work could additionally study distilling large models into smaller sparse models.

Distillation techniques. In Table [6](#page-16-1) we study a variety of distillation techniques. These techniques are built off of [Sanh et al.](#page-38-8) [\(2019\)](#page-38-8), who study distillation methods for BERT models. We find that initializing the dense model with the non-expert weights yields a modest improvement. This is possible since all models are FLOP matched, so non-expert layers will have the same dimensions. Since expert layers are usually only added at every or every other FFN layer in a Transformer, this allows for many of the weights to be initialized with trained parameters. Furthermore, we observe a distillation improvement using a mixture of 0.25 for the teacher probabilities and 0.75 for the ground truth label. By combining both techniques we preserve ≈ 30% of the quality gains from the larger sparse models with only ≈ 1/20th of the parameters. The quality gain refers to the percent of the quality difference between Switch-Base (Teacher) and T5-Base (Student). Therefore, a quality gain of 100% implies the Student equals the performance of the Teacher.

| Technique                                 | Parameters | Quality (↑)     |
|-------------------------------------------|------------|-----------------|
| T5-Base                                   | 223M       | -1.636          |
| Switch-Base                               | 3,800M     | -1.444          |
| Distillation                              | 223M       | (3%)<br>-1.631  |
| + Init. non-expert weights from teacher   | 223M       | (20%)<br>-1.598 |
| + 0.75 mix of hard and soft loss          | 223M       | (29%)<br>-1.580 |
| Initialization Baseline (no distillation) |            |                 |
| Init. non-expert weights from teacher     | 223M       | -1.639          |

<span id="page-16-1"></span>Table 6: Distilling Switch Transformers for Language Modeling. Initializing T5-Base with the non-expert weights from Switch-Base and using a loss from a mixture of teacher and ground-truth labels obtains the best performance. We can distill 30% of the performance improvement of a large sparse model with 100x more parameters back into a small dense model. For a final baseline, we find no improvement of T5-Base initialized with the expert weights, but trained normally without distillation.

Achievable compression rates. Using our best distillation technique described in Table [6,](#page-16-1) we distill a wide variety of sparse models into dense models. We distill Switch-Base versions, sweeping over an increasing number of experts, which corresponds to varying between 1.1B to 14.7B parameters. Through distillation, we can preserve 37% of the quality gain of the 1.1B parameter model while compressing 82%. At the extreme, where we compress the model 99%, we are still able to maintain 28% of the teacher's model quality improvement.

Distilling a fine-tuned model. We conclude this with a study of distilling a finetuned sparse model into a dense model. Table [8](#page-17-1) shows results of distilling a 7.4B parameter Switch-Base model, fine-tuned on the SuperGLUE task, into the 223M T5-Base. Similar to our pre-training results, we find we are able to preserve 30% of the gains of the sparse model when distilling into a FLOP matched dense variant. One potential future avenue, not considered here, may examine the specific experts being used for fine-tuning tasks and extracting them to achieve better model compression.

### <span id="page-16-0"></span>4.3 Multilingual Learning

In our final set of downstream experiments, we measure the model quality and speed tradeoffs while pre-training on a mixture of 101 different languages. We build and benchmark off the recent work of mT5 [\(Xue et al.,](#page-39-2) [2020\)](#page-39-2), a multilingual extension to T5. We pre-train on the multilingual variant of the Common Crawl data set (mC4) spanning 101 languages introduced in mT5, but due to script variants within certain languages, the mixture contains 107 tasks.

In Figure [7](#page-18-0) we plot the quality improvement in negative log perplexity for all languages of a FLOP-matched Switch model, mSwitch-Base to the T5 base variant, mT5-Base. After

|                                | Dense  |        |        | Sparse |        |        |
|--------------------------------|--------|--------|--------|--------|--------|--------|
| Parameters                     | 223M   | 1.1B   | 2.0B   | 3.8B   | 7.4B   | 14.7B  |
| Pre-trained Neg. Log Perp. (↑) | -1.636 | -1.505 | -1.474 | -1.444 | -1.432 | -1.427 |
| Distilled Neg. Log Perp. (↑)   | —      | -1.587 | -1.585 | -1.579 | -1.582 | -1.578 |
| Percent of Teacher Performance | —      | 37%    | 32%    | 30 %   | 27 %   | 28 %   |
| Compression Percent            | —      | 82 %   | 90 %   | 95 %   | 97 %   | 99 %   |

Table 7: Distillation compression rates. We measure the quality when distilling large sparse models into a dense baseline. Our baseline, T5-Base, has a -1.636 Neg. Log Perp. quality. In the right columns, we then distill increasingly large sparse models into this same architecture. Through a combination of weight-initialization and a mixture of hard and soft losses, we can shrink our sparse teachers by 95%+ while preserving 30% of the quality gain. However, for significantly better and larger pre-trained teachers, we expect larger student models would be necessary to achieve these compression rates.

| Model             | Parameters | FLOPS | SuperGLUE (↑) |
|-------------------|------------|-------|---------------|
| T5-Base           | 223M       | 124B  | 74.6          |
| Switch-Base       | 7410M      | 124B  | 81.3          |
| Distilled T5-Base | 223M       | 124B  | (30%)<br>76.6 |

<span id="page-17-1"></span>Table 8: Distilling a fine-tuned SuperGLUE model. We distill a Switch-Base model finetuned on the SuperGLUE tasks into a T5-Base model. We observe that on smaller data sets our large sparse model can be an effective teacher for distillation. We find that we again achieve 30% of the teacher's performance on a 97% compressed model.

pre-training both versions for 1M steps, we find that on all 101 languages considered, Switch Transformer increases the final negative log perplexity over the baseline. In Figure [8,](#page-18-1) we present a different view and now histogram the per step speed-up of using Switch Transformer over the mT5-Base.[9](#page-17-2) We find a mean speed-up over mT5-Base of 5x and that 91% of languages achieve at least a 4x speedup. This presents evidence that Switch Transformers are effective multi-task and multi-lingual learners.

