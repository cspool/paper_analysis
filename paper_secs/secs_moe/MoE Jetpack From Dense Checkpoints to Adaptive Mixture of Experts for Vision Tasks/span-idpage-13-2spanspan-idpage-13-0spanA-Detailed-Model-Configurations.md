# <span id="page-13-2"></span><span id="page-13-0"></span>A Detailed Model Configurations

In this section, we present the detailed model configurations for the main experiments in Sec. [4](#page-5-2) in Tab. [5.](#page-13-3) We refer to pre-trained dense checkpoints as predecessors and the derived MoE models as successors. We use ImageNet-21k pre-trained predecessor from timm with our Checkpoint Recycling algorithm to generate initialized weights for the successor.

Table 5: Configurations for Models.

<span id="page-13-3"></span>

| Configuration           |                      | Successors           | Predecessors |              |  |
|-------------------------|----------------------|----------------------|--------------|--------------|--|
| Model                   | V-JetMoE-T           | C-JetMoE-F           | ViT-S/16     | ConvNext-T   |  |
| FLOPs (G)               | 1.1                  | 1.1                  | 1.1          | 1.1          |  |
| Initialization          | Checkpoint Recycling | Checkpoint Recycling | ImageNet-21k | ImageNet-21k |  |
| MoE Layers              | 7:12                 | 10:18                | -            | -            |  |
| Core Expert Number      | 98                   | [98, 24]             | -            | -            |  |
| Universal Expert Number | 196                  | [196, 48]            | -            | -            |  |

