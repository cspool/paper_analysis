# <span id="page-5-2"></span>4.1. Experimental Settings

Base Models We select Llama2-7B and Llama2-13B [\(Tou](#page-11-14)[vron et al.,](#page-11-14) [2023\)](#page-11-14) as our base models, each with a maximum context length of 4096 tokens. For inference with context lengths shorter than 4096 tokens (in Sec [4.2\)](#page-5-0), we retain the original model weights and fine-tune only the convolutional heads. To extend the context window to 32768 tokens during long context fine-tuning (in Sec [4.3\)](#page-5-1), we utilize positional interpolation for initialization [\(Chen et al.,](#page-9-4) [2023a\)](#page-9-4).

Convolutional Heads We insert convolutional heads layer-wise to capture the diverse token relationships across layers. Each convolutional head possesses one layer of 1-D convolutional kernels: its input feature dimension is the dimension of key and value matrices, while the output feature dimension is the target memory size. We set the kernel size to be 21 by default, as more choices will be validated in Sec [5.3.](#page-8-1) Within the same layer, all attention heads would share the same set of convolutional kernel parameters. Thus, for Llama2-7b, a 32-layer model with 256-dimension KV states, we only add 22 million parameters for compressing raw KV states to a memory of 128 tokens.

Compression Details For post-hoc compression without modifying pre-trained LLMs, we experiment compressing the sequence of length up to 4096, to fit in the memory size of 128, 256, 512, leading to the compression ratio of 32 : 1.

For experiments on context length extending, we by default set 512 as the memory size. We will validate more choices ranging from 128 to 1024 in Sec [5.1.](#page-7-0)

Training Details We use RedPajama [\(Computer,](#page-9-19) [2023\)](#page-9-19) as our training dataset. For post-hoc compression experiments, we only tune compression heads for 200 steps without modifying the pre-trained LLM. For context length extending, we fine-tune the convolutional heads and LoRA adapters (rank 8), and also allow modifying the embedding and normalization layers, all following [Chen et al.](#page-9-6) [\(2023b\)](#page-9-6).

For all experiments, we use the learning rates of 5 × 10−<sup>5</sup> for LoRA adapters, embedding and normalization layers and 5 × 10<sup>−</sup><sup>2</sup> for convolutional heads, with linear learning rate schedule. We use the batch size of 128, and chunk size of 512. All experiments are run on A6000 (48GB memory) to intentionally test our efficacy with small-memory GPUs, and we use per-device batch size as 1.

### <span id="page-5-0"></span>4.2. Post-hoc Token Compression of Pre-trained Models

At inference, we validate LoCoCo on representative downstream tasks, under target memory sizes varying from 128 to 512. We select the reading comprehension dataset RACE [\(Lai et al.,](#page-10-22) [2017\)](#page-10-22) (2, 4, 6 shots), the closed-book question answering dataset TriviaQA [\(Joshi et al.,](#page-10-23) [2017\)](#page-10-23) (50 shots), and the common sense reasoning dataset: HellaSwag [\(Zellers](#page-11-15) [et al.,](#page-11-15) [2019\)](#page-11-15) (10, 20, 40 shots), WinoGrande [\(Sakaguchi](#page-10-24) [et al.,](#page-10-24) [2021\)](#page-10-24) (70 shots), and ARC easy and challenge [\(Clark](#page-9-20) [et al.,](#page-9-20) [2018\)](#page-9-20) (40 shots). Note that we deliberately keep the sequence length of each task within the maximum sequence length of the pre-trained Llama-2 [\(Touvron et al.,](#page-11-14) [2023\)](#page-11-14).

Using the Llama-2-7b [\(Touvron et al.,](#page-11-14) [2023\)](#page-11-14) as the base model, we compare our approach with H2O [\(Zhang et al.,](#page-11-2) [2023b\)](#page-11-2), a recent token dropping method. As in Figure [2,](#page-6-0) LoCoCo shows exceptional performance on various tasks, especially on tasks whose average sequence length is long.

LoCoCo be further applied onto any long-context model. We insert convolutional heads on the top of ChatGLM3- 6B-32k [\(Du et al.,](#page-9-21) [2021\)](#page-9-21), a representative long-context pretrained model. We evaluate the model on SCROLLS [\(Sha](#page-10-5)[ham et al.,](#page-10-5) [2022\)](#page-10-5), a popular long-context dataset, and Table [1](#page-6-1) again demonstrates our effectiveness over H2O.

