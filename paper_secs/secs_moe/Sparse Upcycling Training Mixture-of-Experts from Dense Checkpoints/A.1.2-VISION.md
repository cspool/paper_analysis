# A.1.2 VISION

Dense ViT models are pretrained on JFT300M Sun et al. (2017). We train with Adafactor (Shazeer & Stern, 2018), and decoupled weight decay (magnitude 3 on head and 0.03 on body) following Zhai et al. (2022). We use a batch size of 4096. The learning rate schedule consists of a learning warmup of 10 000 steps, followed by reverse square root decay with timescale 100 000 steps and ending with a linear cooldown to 0 over 50 000 steps. We use a fixed peak learning rate of  $4 \cdot 10^{-4}$ . Models employing a patch size of 32 (i.e. B/32, L/32) were trained for a total of 14 epochs, while those employing a patch size of 16 (i.e. B/16, L/16) were trained for 28 epochs. By default, we begin upcycling half-way through the total number of epochs in each case. This roughly corresponds to the number of epochs used to train the corresponding variant in ViT (Dosovitskiy et al., 2021). For B/16, for example, we assume that a dense checkpoint trained for 14 epochs is given, and we either continue training or apply upcycling for another 14 epochs.

## <span id="page-15-1"></span>A.2 MODEL TRANSFER

