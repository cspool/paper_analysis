# H. Supplementary Material

This Supplementary document is organized as follows:

- A.1 [Reproducibility Statement](#page-10-1)
- A.2 [Implementation Details](#page-10-2)
- A.3 [Model Details](#page-10-3)
- A.4 [Extended Ablations](#page-11-0)
- A.5 [HERMES vs. MA-LMM vs. MovieChat](#page-12-0)
- A.6 [A Note on Latency](#page-13-3)
- A.7 [More Qualitative Results](#page-13-1)
- A.8 [Error Analysis: When does HERMES fail and why?](#page-13-2)
- A.9 [How is our approach related to cognitive processes?](#page-14-0)

## <span id="page-10-1"></span>H.1. Reproducibility Statement

To facilitate the reproducibility of our work, we will make our code, pretrained models, default hyperparameters, and preprocessed annotations publicly available. Detailed hyperparameters for each dataset are also provided in Table [9.](#page-11-2) Our model demonstrates efficient performance, completing inference on the MovieChat-1k test set in 13 minutes (22 FPS) using a single V100 GPU (32 GB), and training on the MovieChat-1k dataset in less than 12 minutes with 8x 32 GB GPUs. In contrast to recent LLM-based approaches that necessitate extensive and costly multi-stage pretraining on increasingly large datasets, our model is designed for accessibility, thereby lowering the barrier for researchers without access to high-end computing resources. We achieve high performance while maintaining accessibility by leveraging existing pretrained weights and implementing our trainingfree ECO and SeTR, resulting in a model where finetuning is optional. We also demonstrate the applicability of our modules to existing video models, and are planning to submit pull requests to integrate our modules into these models.

For fully-supervised results, QFormers and adapter are fine-tuned on the respective dataset's training split. For plugin experiments, ECO and SeTR are inserted into target architectures at inference time, with zero additional training, demonstrating true plug-and-play capability.

## <span id="page-10-2"></span>H.2. Implementation Details

To ensure the reproducibility of our results, we provide training and inference details in Table [9.](#page-11-2) These settings are mostly consistent across different datasets. In the table, LR is the learning rate, and Keep Ratio is the SeTR keep ratio. Episodes refer to the number of episodes to which we compress the input frames (i.e., the capacity of ECO). The number of frames (N) represents the quantity of frames retained from the original video to serve as input to the model. These frames are selected by applying a regular stride over the original video's frame sequence, where the stride length is determined by the ratio of original frame count to N. *Max Epoch = 20* means we run the program for 20 epochs, performing evaluation after each epoch, and then pick the model with the highest validation accuracy. MovieChat-1k (G) and MovieChat-1k (B) denote global and breakpoint modes, respectively. All models were trained on 8 V100 GPUs (32GB VRAM each). We test on VideoMME using the zero-shot setting by applying our modules to two different models, the same parameters were used across models for consistency.

