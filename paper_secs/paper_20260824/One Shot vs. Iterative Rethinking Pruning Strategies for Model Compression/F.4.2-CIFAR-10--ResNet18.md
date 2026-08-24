# *F.4.2 CIFAR-10 / ResNet18*

The same ResNet-18 model as for CIFAR-100 was used. Before pruning, the model was trained for 226 epochs with the same configuration as the CIFAR-100 experiment. Before pruning, the top-1 accuracy was 94.14%.

Table 4: Pruning Hybrid Scheduler Parameters 0.92 - 0.99

| One-shot step | Iterative step | Target pruning value |
|---------------|----------------|----------------------|
| 0.5           | 0.01997        | 0.92                 |
| 0.6           | 0.0191         | 0.92                 |
| 0.7           | 0.01893        | 0.92                 |
| 0.8           | 0.0181         | 0.92                 |
| 0.5           | 0.04831        | 0.92                 |
| 0.6           | 0.04706        | 0.92                 |
| 0.7           | 0.04848        | 0.92                 |
| 0.8           | 0.04172        | 0.92                 |
| 0.5           | 0.08679        | 0.92                 |
| 0.6           | 0.09191        | 0.92                 |
| 0.7           | 0.07948        | 0.92                 |
| 0.8           | 0.06192        | 0.92                 |
| 0.5           | 0.01968        | 0.96                 |
| 0.6           | 0.01922        | 0.96                 |
| 0.7           | 0.01987        | 0.96                 |
| 0.8           | 0.01919        | 0.96                 |
| 0.5           | 0.04629        | 0.96                 |
| 0.6           | 0.04838        | 0.96                 |
| 0.7           | 0.04895        | 0.96                 |
| 0.8           | 0.04265        | 0.96                 |
| 0.5           | 0.0976         | 0.96                 |
| 0.6           | 0.08539        | 0.96                 |
| 0.7           | 0.0955         | 0.96                 |
| 0.8           | 0.08348        | 0.96                 |
| 0.5           | 0.01961        | 0.99                 |
| 0.6           | 0.01958        | 0.99                 |
| 0.7           | 0.01994        | 0.99                 |
| 0.8           | 0.01897        | 0.99                 |
| 0.5           | 0.04696        | 0.99                 |
| 0.6           | 0.04823        | 0.99                 |
| 0.7           | 0.04775        | 0.99                 |
| 0.8           | 0.04127        | 0.99                 |
| 0.5           | 0.09171        | 0.99                 |
| 0.6           | 0.09413        | 0.99                 |
| 0.7           | 0.08206        | 0.99                 |
| 0.8           | 0.1            | 0.99                 |

#### *F.4.3 ImageNet / ResNet18*

We used the ResNet-18 model from the PyTorch torchvision library with pretrained weights. During fine-tuning in the pruning phase, images were resized to 256 × 256 and cropped to 224 × 224. Before pruning, the top-1 accuracy was 68.91%.

## *F.4.4 CIFAR-100 / EfficientNet*

We used the EfficientNet V2-S model from the PyTorch torchvision library with pretrained weights (IMAGENET1K\_V1). Before pruning, the model was trained for 132 epochs with the following configuration:

- SGD optimizer with learning rate: 0.1, momentum: 0.9, and weight decay: 0.0005
- Linear scheduler for 10 epochs with start factor: 0.01
- After 10 epochs: CosineAnnealingWarmRestarts with T<sup>0</sup> = 10, Tmult = 2, and ηmin = 1 × 10<sup>−</sup><sup>5</sup>

- Early stopping with patience of 80 epochs
- Images resized to 128 × 128

Images were resized during fine-tuning in the pruning phase. Before pruning, the top-1 accuracy was 87.53%.

#### *F.4.5 CIFAR-10 / EfficientNet*

The same EfficientNet V2-S model as for CIFAR-100 was used. Before pruning, the model was trained for 152 epochs with the same configuration as the CIFAR-100 experiment. Before pruning, the top-1 accuracy was 97.88%.

## *F.4.6 CIFAR-100 / ViT*

We used the ViT small patch16 224 model from the timm library with pretrained weights (vit small patch16 224 augreg in1k). Before pruning, the model was trained for 18 epochs with the following configuration:

- SGD optimizer with learning rate: 0.1, momentum: 0.9, and weight decay: 0.0005
- Linear scheduler for 10 epochs with start factor: 0.01
- After 10 epochs: CosineAnnealingWarmRestarts with T<sup>0</sup> = 10, Tmult = 2, and ηmin = 1 × 10<sup>−</sup><sup>5</sup>
- Early stopping with patience of 50 epochs
- Images resized to 224 × 224

Images were resized during fine-tuning in the pruning phase. Before pruning, the top-1 accuracy was 88.16%.

## *F.4.7 CIFAR-10 / ViT*

The same ViT small patch16 224 model as for CIFAR-100 was used. Before pruning, the model was trained for 19 epochs with the same configuration as the CIFAR-100 experiment. Before pruning, the top-1 accuracy was 98.11%.