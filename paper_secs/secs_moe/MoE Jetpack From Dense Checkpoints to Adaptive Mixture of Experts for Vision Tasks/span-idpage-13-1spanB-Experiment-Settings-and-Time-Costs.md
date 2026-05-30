# <span id="page-13-1"></span>B Experiment Settings and Time Costs

In this section of the appendix, we provide a comprehensive description of the training settings used in our experiments. Tab. [6](#page-13-4) outlines the standard training configuration utilized across our experiments. Tab. [7](#page-13-4) details the dataset-specific training configurations, capturing variations in batch size, warmup epochs, total training epochs, and drop path rates for each dataset employed in our experiments.

Our experiments were conducted on RTX 4090 GPU. Training V-JetMoE-T on the CIFAR-100 dataset (60,000 images) required 2.5 GPU hours while training on the ImageNet-1K dataset (1,281,167 images) required 120 GPU hours. Training C-JetMoE-F on CIFAR-100 also required 2.5 GPU hours and 156 GPU hours on ImageNet-1K. For V-JetMoE-S, training on CIFAR-100 required 8 GPU hours and 200 GPU hours on ImageNet-1K. Compared to the original dense models (ViT-Tiny, ConvNeXt-Femto, ViT-Small), our method achieves nearly equivalent training times.

For all the experiments presented in our paper, we required 3, 300 GPU hours for training. In total, we spent approximately 8, 000 GPU hours for exploration and validation of our work.

<span id="page-13-4"></span>Table 6: Our basic recipe for model training.

| Training Setting          | Configuration |
|---------------------------|---------------|
| image resolution          | 224 × 224     |
| optimizer                 | AdamW[52]     |
| base learning rate        | 4 × 10−3      |
| weight decay              | 0.05          |
| optimizer momentum β1, β2 | = 0.9, 0.999  |
| batch size                | 4096          |
| training epochs           | 300           |
| learning rate schedule    | cosine decay  |
| warmup epochs             | 50            |
| warmup schedule           | linear        |
| randaugment [53]          | (9, 0.5)      |
| mixup [54]                | 0.8           |
| cutmix [55]               | 1.0           |
| random erasing [56]       | 0.25          |
| label smoothing [57]      | 0.1           |
| layer scale [58]          | 1 × 10−6      |

Table 7: Hyper-parameter setting on ViT-T.

| Setting | Batch<br>Size | Warmup<br>Epochs | Training<br>Epochs | Drop<br>Path Rate |
|---------|---------------|------------------|--------------------|-------------------|
| C-10    | 512           | 50               | 300                | 0.1               |
| C-100   | 512           | 50               | 300                | 0.1               |
| Pets    | 512           | 100              | 600                | 0.1               |
| Flowers | 512           | 100              | 600                | 0.1               |
| STL-10  | 512           | 50               | 300                | 0                 |
| Food101 | 512           | 50               | 300                | 0.1               |
| DTD     | 512           | 100              | 600                | 0.2               |
| IN1k    | 4096          | 50               | 300                | 0                 |

