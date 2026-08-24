# <span id="page-4-0"></span>3 Supervised fine-tuning for LLMs on CLIPPER data

Having shown that CLIPPER produces synthetic data of high quality, we now investigate the effects of training on such data. We apply supervised fine-tuning (SFT) to three models on our dataset: ProLong-512K-8B-Base (Gao et al., 2024b), <sup>11</sup> Llama-3.1-8B-Instruct (Dubey et al., 2024), and Qwen2.5-7B-Instruct (Team, 2024). Our top model, Llama-CLIPPER, achieves nearly three times the test set performance of Llama-Instruct—boosting accuracy from 27.9% to 76%—while showing substantial gains in long-context reasoning and narrative

<span id="page-4-1"></span><sup>&</sup>lt;sup>9</sup>Chosen to mitigate potential self-biases (Xu et al., 2024b; Panickssery et al., 2024; Li et al., 2025).

<span id="page-4-3"></span><span id="page-4-2"></span><sup>&</sup>lt;sup>10</sup>See detailed cost analysis in §A.4.

<sup>&</sup>lt;sup>11</sup>Despite the name, this model has undergone instruction tuning before. The ProLong team ran continual pre-training on Llama-3-8B-Instruct to get this model.

<span id="page-4-4"></span><sup>&</sup>lt;sup>12</sup>We will now refer to these models as ProLong-Base, Llama-Instruct, and Qwen-Instruct.

understanding on tasks like NoCha, NarrativeQA, and MuSR. Moreover, all of our models outperform all existing <10B models on the NoCha benchmark.

## **3.1 Training setup**

**Data splits and hyperparameters:** We divide our dataset into three parts: 16K claims (8K true/false pairs) for training, 2K for validation, and 1K for testing. Notably, the books in the test set do not overlap with those in the training or validation sets. For each entry, we combine the book text and claim to form the user prompt, and include the chain of thought reasoning along with the final answer as the assistant's message (see Figure [16\)](#page-32-0). A learning rate of 1e-6 and a batch size of 16 yield the best performance on our dev set.[13](#page-5-0) We fine-tune Qwen-Instruct, Llama-Instruct, and ProLong-Base using this configuration for one epoch.

**Ablation on the effect of claim scope:** Our dataset consists of 8K book-level and 8K chapter-level claims. We fine-tune ProLong-Base separately on each claim scope subset, resulting in ProLong-CLIPPER-chapter and ProLong-CLIPPER-book.

**Ablation on the effect of data length:** Prior work shows that fine-tuning on short texts can improve long-context performance in tasks like QA and summarization [\(Dubey et al.,](#page-10-2) [2024;](#page-10-2) [Gao et al.,](#page-10-6) [2024b\)](#page-10-6). Since our dataset contains long documents averaging 90K tokens, we test whether short-text fine-tuning also helps with long-context claim verification. We use WritingPrompts [\(Fan et al.,](#page-10-7) [2018\)](#page-10-7), a dataset of 300K stories averaging 742 tokens, and extract claims directly without generating outlines or summaries.[14](#page-5-1) We collect 19K claims and train on ProLong-Base to get ProLong-WritingPrompts.[15](#page-5-2)

## **3.2 Evaluation**

Beyond claim verification, we expect that training on our synthetic dataset will also improve performance on related tasks. Therefore, we include both reasoning and narrative understanding benchmarks that vary in input lengths and tasks.

**Claim verification:** To measure accuracy, we calculate the percentage of cases in which a model correctly verifies both true and false claims within a given pair.

- **CLIPPER-test** contains 1,000 true/false claim pairs drawn from 53 books, evenly split between book-level and chapter-level claims.
- **NoCha** [\(Karpinska et al.,](#page-11-1) [2024\)](#page-11-1) consists of 1,001 true/false claim pairs about recent fiction books (up to 336k tokens). These claims, crafted by annotators familiar with the books, are much harder to verify compared to those in CLIPPER-test.

**General narrative understanding:** We use three existing benchmarks as detailed below.

- **NarrativeQA** [\(Kocisk](#page-11-3) ˇ y et al. ´ , [2018\)](#page-11-3) is a long-form Q&A benchmark that requires models to process entire books or movie scripts to answer provided questions. The benchmark consists of 1,572 stories and summaries as well as 46,675 human-written questions. We use the HELMET implementation [\(Yen et al.,](#page-14-3) [2024\)](#page-14-3) for this benchmark.
- ∞**Bench QA** [\(Zhang et al.,](#page-14-4) [2024\)](#page-14-4) is a long-form Q&A benchmark that requires models to answer 351 questions about novels. We use the HELMET implementation but use GPT-4o's judgment as a metric instead of ROUGE F1 (see [§C.1](#page-31-0) for explanation).

<span id="page-5-0"></span><sup>13</sup>We perform hyperparameter tuning on learning rates of 1e-5, 1e-6, and 1e-7, along with batch sizes of 16 and 32. Tuning is done for one epoch on a subset of 2K training samples. Due to high GPU costs (each epoch takes 5 hours), we only conduct hyperparameter tuning on ProLong-Base only.

<span id="page-5-2"></span><span id="page-5-1"></span><sup>14</sup>We use a prompt similar to the one in [§2.3.](#page-3-0)

<sup>15</sup>After doing hyperparameter tuning on 2K training samples, we decide on the learning rate of 1e-5 and batch size of 16 as the best training configurations. We tested learning rates of 1e-5, 1e-6, 1e-7 and batch sizes of 8, 16, 32, 64.

| Models                         | CLIPPER-test | NoCha | NarrativeQA | MuSR  | ∞Bench QA |
|--------------------------------|--------------|-------|-------------|-------|-----------|
| Qwen2.5-7B-Instruct            | 51.0%        | 24.1% | 40.3%       | 41.2% | 35.3%     |
| Llama-3.1-8B-Instruct          | 27.9%        | 16.5% | 47.7%       | 40.3% | 47.8%     |
| ProLong-512K-8B-Instruct       | 34.5%        | 16.9% | 44.0%       | 42.3% | 42.6%     |
| ♣ Qwen2.5-7B-CLIPPER           | 73.9%        | 32.4% | 46.0%       | 45.2% | 42.3%     |
| 🔦 Llama-3.1-8B-CLIPPER         | 76.0%        | 32.2% | 49.0%       | 43.6% | 46.5%     |
| ProLong-512K-8B-CLIPPER        | 75.0%        | 32.3% | 49.0%       | 44.5% | 38.5%     |
| ProLong-512K-8B-WritingPrompts | 63.0%        | 24.1% | 31.0%       | 45.2% | 35.8%     |

<span id="page-6-1"></span>Table 2: Model accuracy on claim verification (CLIPPER-test, NoCha) and narrative understanding benchmarks (NarrativeQA, MuSR, ∞Bench QA). Fine-tuning models using CLIPPER improves performance on claim verification and narrative understanding.

➤ MuSR (Sprague et al., 2024) includes 756 algorithmically generated problems such as murder mysteries, object placement questions, and team allocation optimization. We use the LM Harness (Gao et al., 2024a) implementation.

