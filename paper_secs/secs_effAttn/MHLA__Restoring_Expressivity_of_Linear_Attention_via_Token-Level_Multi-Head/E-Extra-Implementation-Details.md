# **E Extra Implementation Details**

Image Classification. For training of DeiT, we replace the class token with average pooling and train all baselines under identical settings to ensure fair comparison. We additionally add CPE [\[10\]](#page-11-8) with a kernel size of 3, following previous works for a fair comparison. For VLT, we strictly follow the setup in [\[22\]](#page-12-2). All models are trained for 300 epochs with a batch size of 1024 and a peak learning rate of 1e-3. For models with an input size of 224, we pad the input size to 256 for better splitting of heads. The head number M is set to 16 for DeiT modes. For VLT models, the sequence length for the two linear attention layers is {3136, 784}. So we set the head number M to {49, 16} for the two layers respectively.

