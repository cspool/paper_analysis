# E. Instruction Set Extension

The SPARK encoding scheme does not require the introduction of new data types specifically for multiply-accumulate instructions. Instead, it leverages the existing instruction set architecture, ensuring compatibility and easy integration with the current system design. SPARK encoding mechanism operates with a fixed bit length, allowing for the concatenation of high and low precision values based on encoding rules. Consequently, the original load/store instructions remain applicable and unchanged, maintaining compatibility with the existing system design.

After performing INT-based quantization, the specific type of each layer is determined, enabling the seamless substitution of the original integer-based versions with SPARK-encoded numbers. This approach ensures that the encoding scheme can be readily incorporated into existing systems without requiring extensive modifications.

#### V. EVALUATION

#### A. Methodology

**Benchmark.** We use two representative groups of models, CNNs and attention-based models, including computer vision

TABLE III EVALUATED MODEL, DATASET, AND THEIR ACCURACY.

| Type          | CNN-based |          |          | Attention-based |       |
|---------------|-----------|----------|----------|-----------------|-------|
| Model         | VGG16     | ResNet18 | ResNet50 | ViT             | BERT  |
| Dataset       | ImageNet  |          |          |                 | SST-2 |
| FP-32 Acc.(%) | 71.59     | 69.76    | 76.15    | 84.19           | 90.45 |
| SPARK Acc.(%) | 71.38     | 69.69    | 76.05    | 84.23           | 91.14 |

and natural language processing tasks, as reported in Table III. For CNN models, we evaluate VGG-16, ResNet-18 and ResNet-50 on ImageNet dataset. We use pre-trained network models from Pytorch Model Zoo as the basis for validation on accuracy. The network structure is derived from the model in the torchvision. For attention-based models, we evaluate BERT-Base with the GLUE dataset suite. We also evaluate ViT (vision transformer), which is a recent Transformer-based model and has achieved excellent results for vision tasks.

Baselines. We implement the SPARK encoding framework in PyTorch. We evaluate six baselines compared against SPARK, including: 1) Eyeriss [3], which is a spatial energyefficient dataflow architecture. It uses coarse-grained quantization of INT16 throughout the network. We will use it as the baseline for standard NN accuracy; 2) BitFusion [38], which is an accelerator featured by the composable MAC unit. It employs the model typologies proposed in prior work [9], [37], [39]. If the algorithm permits, it can change quantization at the layer granularity. If a coarse-grained quantization of INT16 is applied throughout the network, it can retain the accuracy as Eyeriss; 3) OLAccel [35], which is a state-ofthe-art accelerator based on the outlier-aware low-precision computation; 4) AdaFloat, which requires an 8-bit float to maintain the original model accuracy; 5) ANT [11], which is a hardware-friendly quantization accelerator that combines power of two data type and INT type for low bit-width; 6) Olive [10], which is outlier-aware quantization accelerator. For comparison, we take part of results as reported in their paper. For models that are not available in their paper, we reproduce their experiments and report the results.

Accelerator Implementation. We implement the ANT decoder, PE and encoder described in Section IV with the Verilog RTL. Meanwhile, we use the 28 *nm* TSMC technology library and Synopsys Design Compiler [5] to study the area and energy of those components that we designed. In addition,

TABLE IV ACCURACY LOSS (%) AND BITWIDTH COMPARISON BETWEEN SPARK AND OTHER DESIGNS WITHOUT FINETUNING.

| Model     | SPARK           | ANT          | BiScaled     |
|-----------|-----------------|--------------|--------------|
| VGG16     | 0.08 (5.33 bit) | 0.71 (6 bit) | 1.66 (6 bit) |
| ResNet50  | 0.81 (5.11 bit) | 0.89 (6 bit) | 5.51 (6 bit) |
| ResNet152 | 0.77 (5.21 bit) | 0.95 (6 bit) | 4.84 (6 bit) |

for a fair comparison, we use the same global buffer capacity (5 MB) and memory bandwidth for all these accelerators and use CACTI [1] to estimate it that can satisfy our design goals. We use DeepScaleTool to scale all designs to the 28 *nm* process for the iso-area comparison. Under 200MHz frequency with 28nm technology library, we verify that the encoder/decoder bandwidth is about 50 GB/s, which is larger than the peak bandwidth requirement (∼25 GB/s) [4] of PE pages. So, SPARK can sustain a non-blocking processing with decoding/encoding support. To evaluate the performance of our proposed SPARK architecture, we develop a cycleaccurate simulator to simulate the PE array with decoders and output accumulation together with the encoder. Meanwhile, the SPARK architecture can be extended to a larger number of PEs under the same area budget.

