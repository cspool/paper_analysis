# A.4 Installation

First, ensure Python 3.9+ is installed on the system. For installing PyTorch, please refer to the official [PyTorch website,](https://pytorch.org/get-started/locally/) since installation commands vary by environment. Clone the quantized-training artifact repository to the local machine:

- \$ git clone https://github.com/jeffreyyu0602/ quantized-training
- \$ cd quantized-training

```
$ mkdir logs
```

Then, install quantized-training and all the required packages listed in requirements.txt.

```
$ pip install -e .
$ pip install -r requirements.txt
```

## A.5 Experiment workflow

A set of Python scripts will generate the numbers and tables in the paper. The complete instructions can be found in the README.md file included within the artifact repository.

#### A.6 Evaluation and expected results

A.6.1 SQuAD inference experiments. Table 2 consists of quantized inference results (F1 scores) on SQuAD v1.1 for MobileBERTtiny, MobileBERT, BERTbase, BERTlarge, and DistillBERTbase. The results can be replicated by running the following commands:

```
python examples/question_answering/run_squad.py [--
    log_file <PATH_TO_LOG_FILE>]
```

By default, the log file is saved as logs/squad.log. The script will produce a squad\_f1.csv which has the same format as Table 2.

The inference results will be affected by the specific device the experiments are run on, CUDA version, and software versions (e.g. PyTorch version). Therefore, it is very hard to get bit-wise identical results. The F1 scores in squad\_f1.csv may differ slightly from those in Table 2, with variations of up to a few tenths of a percent (0.1%).

A.6.2 Fine-tuning experiments. Table 7 presents the results of fine-tuning MobileBERTtiny, MobileBERT, RoBERTabase, and RoBERTalarge on the GLUE and SQuAD v1.1 datasets. The hyperparameters used for training—such as batch size, learning rate, number of training epochs, and LoRA parameters are specified in the training script run\_quantized\_training. py. Users can reproduce all the results using the commands detailed in asplos\_experiments.sh. The commands have the following format:

```
python run_quantized_training.py --model <MODEL> --
    task <TASK> --run_job <JOB_NAME> [--seed 42] [--
    log_file <LOG_FILE>] [--op_fusion <LAYER>]
```

Available models for selection include:

- models/mobilebert\_tiny
- google/mobilebert-uncase
- roberta-base
- roberta-large

For task, options include mnli, qnli, mrpc, sst2, and squad. For job name, selection options are:

- bf16: run BFloat16 training
- posit8: run Posit8 quantized training

- posit8-approx-shifted: run Posit8 quantized training with posit approximation
- fp8: run FP8 quantized training

Users can optionally specify a random seed for reproducibility and a filename to redirect logs to.

For example, MobileBERT LoRA Posit8 results on MRPC can be replicated by running:

```
python run_quantized_training.py --model google/
    mobilebert-uncased --task mrpc --run_job posit8
    --seed 42 --log_file logs/mobilebert-mrpc.log
```

The above command takes approximately 30 minutes to run on an NVIDIA RTX 4090 GPU. The final accuracy should match closely with the figure in Table 7.

Low-precision training can sometimes lead to numerical instability and non-finite gradients. In such instances, users have several options: resuming training from a previous checkpoint, lowering the learning rate, or selecting a different random seed. Additionally, when fine-tuning Mobile-BERT with Posit8 or FP8, training stability can be achieved by fusing the last layer. For GLUE tasks, the relevant option is --op\_fusion classifier, and for SQuAD the option is --op\_fusion qa\_outputs.

Similar to inference experiments, replicating bit-wise identical results is challenging due to variations in hardware and software. For fine-tuning, conducting multiple runs with different random seeds is advised to eliminate outlier runs. Variations in results of up to half a percent (0.5%) are considered normal.

