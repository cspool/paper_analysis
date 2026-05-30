# A Artifact Appendix

#### A.1 Abstract

This appendix describes the steps to set up and conduct Posit8 and FP8 inference and fine-tuning experiments on various Transformer models. Additionally, it guides on reproducing the quantitative results presented in the paper. For executing these artifacts, a CUDA-enabled GPU is necessary. The experiments also require Python 3.9+ and PyTorch 2.0+. The inference and fine-tuning results are expected to align closely with the figures reported in the paper. However, minor discrepancies up to half a percent may occur due to variations in hardware and software.

#### A.2 Artifact check-list (meta-information)

- Models: MobileBERT (15M & 25M), DistillBERT (66M), BERT (110M & 340M), Whisper (39M & 244M & 1.6B), GPT-2 (762M & 1.5B), and LLaMA2 (7B & 13B). All the models except MobileBERTtiny are available from Hugging Face and will be downloaded when running inference or fine-tuning scripts. MobileBERTtiny is accessible from the [artifact code](https://github.com/jeffreyyu0602/quantized-training/tree/main/models/mobilebert_tiny) [repository.](https://github.com/jeffreyyu0602/quantized-training/tree/main/models/mobilebert_tiny)
- Datasets: GLUE, SQuAD v1.1, LibriSpeech, and WikiText-103. All the datasets will be downloaded from Hugging Face when running the script.
- Run-time environment: Python 3.9+ and PyTorch 2.0+.
- Hardware: CUDA-enabled GPU with 16 GB of dedicated VRAM for most models, and 32 GB of VRAM (can be split across two GPUs) for LLMs.
- Metrics: GLUE tasks use accuracy (higher is better), SQuAD uses F1 score (higher is better), speech recognition uses word error rate (WER) (lower is better), and LLMs use loss or perplexity (lower is better).
- Output: Numerical results, e.g. accuracy, are printed in the log file, whose location can be specified by the command line arguments.
- Experiments: The README.md file in the GitHub repository provides detailed instructions for running all experiments. The results may vary depending on the specific hardware and software versions used. Variations of up to a few tenths of a percent (0.1%) from the figures reported in the paper are considered acceptable.
- How much disk space required (approximately)?: 128 GB of disk space is enough for all experiments.
- How much time is needed to prepare workflow (approximately)?: About 10 to 15 minutes.
- How much time is needed to complete experiments (approximately)?: The time required for experiments varies depending on the specific task and hardware used. On a machine with an NVIDIA RTX 4090 GPU, one quantized inference experiment usually completes within 10 minutes, while fine-tuning experiments can take from 30 minutes to more than a day, depending on the size of the model and the dataset.
- Publicly available?: Yes, the artifact code is available on GitHub: <https://github.com/jeffreyyu0602/quantized-training>.
- Code licenses (if publicly available)?: MIT License.

#### A.3 Description

A.3.1 How to access. The code repository for this submission can be downloaded from [https://github.com/jeffreyyu060](https://github.com/jeffreyyu0602/quantized-training)2/ [quantized-training](https://github.com/jeffreyyu0602/quantized-training).

- A.3.2 Hardware dependencies. CUDA-enabled NVIDIA GPUs and a machine with > 32 GB memory.
- A.3.3 Software dependencies. Evaluation of artifacts requires a machine with Python 3.9+ and PyTorch 2.0+ installed. We tested the artifacts on Ubuntu 23.10 with Python 3.9 and PyTorch 2.0, and found them to work.
- A.3.4 Datasets. We evaluate MobileBERT and BERT models using the GLUE and SQuAD v1.1 datasets, LLMs (e.g., LLaMA2 and GPT-2) on the WikiText dataset, and Whisper models on the LibriSpeech dataset. These datasets are available from Hugging Face and will be downloaded when running inference or fine-tuning scripts.
- A.3.5 Models. For inference experiments, we use models trained by either the original authors or third parties, except for MobileBERTtiny which we fine-tuned ourselves and uploaded to the artifact code repository. These models can be found on the Hugging Face website with links provided below, and will be automatically downloaded when running the quantized inference experiments.
  - [MobileBERT](https://huggingface.co/csarron/mobilebert-uncased-squad-v1) fine-tuned by third-party on SQuAD v1.1
  - [DistillBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad)base fine-tuned by Hugging Face on SQuAD v1.1
  - [BERT](https://huggingface.co/csarron/bert-base-uncased-squad-v1)base fine-tuned by third-party on SQuAD v1.1
  - [BERT](https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad)large fine-tuned by Google on SQuAD v1.1
  - [Whisper](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013) trained on 680k hours of labelled speech data by OpenAI
  - [GPT-2](https://huggingface.co/openai-community) pretrained on a very large corpus of English data by OpenAI
  - [LLaMA2](https://huggingface.co/meta-llama) pre-trained on a very large corpus of English data by Meta

To access LLaMA2, users must first request access through [Meta's official website.](https://llama.meta.com/llama-downloads/) Following this, users need to create an account and apply for access to the model checkpoints on the [Hugging Face website.](https://huggingface.co/meta-llama/Llama-2-7b) After gaining access, users can log in from the terminal through Hugging Face CLI, as detailed in [Hugging Face documentation.](https://huggingface.co/docs/huggingface_hub/en/quick-start)

For fine-tuning experiments, we use pre-trained model checkpoints provided by Hugging Face, except for MobileBERTtiny which is accessible from the artifact code repository.

- [MobileBERT](https://huggingface.co/google/mobilebert-uncased) from Google
- [RoBERTa](https://huggingface.co/FacebookAI/roberta-base)base from Meta
- [RoBERTa](https://huggingface.co/FacebookAI/roberta-large)large from Meta

