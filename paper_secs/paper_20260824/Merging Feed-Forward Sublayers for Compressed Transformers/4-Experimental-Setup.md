# 4 Experimental Setup

<span id="page-3-1"></span>To test its extensibility, we apply our compression method to a diverse set of Transformer-based models. Specifically, we use GPT-2 [\(Radford et al.,](#page-9-13) [2019\)](#page-9-13), a vision transformer (ViT) [\(Dosovitskiy](#page-8-11) [et al.,](#page-8-11) [2020\)](#page-8-11), and a Transformer-based MT model from OPUS-MT [\(Tiedemann and Thottingal,](#page-10-9) [2020\)](#page-10-9). We select these models in order to cover a diversity of Transformer model types (decoder-only, encoder, encoder-decoder) and different modalities. We additionally include experiments on OLMo-7B using QLoRA [\(Groeneveld et al.,](#page-8-12) [2024\)](#page-8-12).

For each main setting, we list the model used, the example data for computing alignments, and finally the data used for recovery fine-tuning and evaluation. Additional hyperparameters are included in Appendix [C,](#page-11-1) and dataset details in Appendix [D.](#page-12-0)

#### 4.1 Language modeling

For our experiments, we use GPT-2 Large which was trained on extensive English text [\(Radford](#page-9-13) [et al.,](#page-9-13) [2019\)](#page-9-13). For computing example activations, we use ~10k tokens from the validation set of the Wikitext103 dataset [\(Merity et al.,](#page-9-14) [2017\)](#page-9-14). Finally, we use the train and test sets from the Wikitext103 for fine-tuning and evaluation, respectively.

Because we use Wikitext103 for recovery finetuning, we also fine-tune the uncompressed GPT-2 baseline model for fair comparison. Because we have access to the training data for our machine translation and ViT models, we do not provide a fine-tuned baseline for those as the data we use already appears in their original training data.

We fine-tune our GPT-2 models for up to 100k steps with batches of 2048 tokens. We select the best model based on validation perplexity and report average test perplexity with a sliding window of 512 tokens.

<span id="page-3-2"></span><sup>3</sup> For an extension of this method to SwiGLU [\(Shazeer,](#page-10-8) [2020\)](#page-10-8), see Appendix [A.](#page-11-0)

<span id="page-3-3"></span><sup>4</sup>An algorithm summarizing our selection method can be found in Appendix [B.](#page-11-2)

## 4.2 Image classification with ViT

We use a vision transformer (ViT) for our image classification experiments, with resolution of 224x224, and patch size of 16x16 [\(Dosovitskiy](#page-8-11) [et al.,](#page-8-11) [2020\)](#page-8-11). ViT is a 12-layer Transformer encoder model pre-trained on ImageNet-21k, and subsequently fine-tuned on ImageNet-1k. ImageNet-1k is a classification task where images belong to one of 1000 categories [\(Russakovsky et al.,](#page-10-10) [2015\)](#page-10-10). For computing activations, we use ~10k patches from the ImageNet-1k validation set. Evaluation results are computed on original validation labels.

We fine-tune our ViT models on ImageNet-1k for up to 50k steps with a batch size of 128, and report accuracy.

#### 4.3 Machine translation

For our experiments on machine translation, we use a 12-layer Chinese-English Transformer-based translation model from an OPUS-MT release [\(Tiedemann and Thottingal,](#page-10-9) [2020\)](#page-10-9). For computing activations, we use ~10k tokens from the Tatoeba validation set[5](#page-4-0) [\(Tiedemann,](#page-10-11) [2020\)](#page-10-11). For fine-tuning, we use the original training data released by the Tatoeba translation challenge, sourced from OPUS [\(Tiedemann,](#page-10-12) [2012\)](#page-10-12).

We apply our method to both the encoder and decoder separately, constituting two anchors. However, we search windows in sync, meaning that the same window from the encoder and decoder are merged, but separately.

We fine-tune our translation models for up to 100k steps with a batch size of 64 sentences. We use sacrebleu to compute BLEU scores for evaluation [\(Papineni et al.,](#page-9-15) [2002;](#page-9-15) [Post,](#page-9-16) [2018\)](#page-9-16), and pymarian to compute COMET scores[6](#page-4-1) [\(Rei et al.,](#page-9-17) [2022;](#page-9-17) [Gowda et al.,](#page-8-13) [2024\)](#page-8-13).

#### 4.4 Layer pruning baseline

Recent work on the structured pruning of Transformers has seen many methods that remove full layers from a model and then optionally fine-tune the compressed model [\(Ma et al.,](#page-9-18) [2023;](#page-9-18) [Men et al.,](#page-9-7) [2024;](#page-9-7) [Gromov et al.,](#page-8-6) [2024;](#page-8-6) [Yang et al.,](#page-10-13) [2024b\)](#page-10-13). We focus on a structured pruning baseline as many unstructured pruning methods do not realize memory savings unless they achieve 1) high sparsity ratios and 2) use specialized sparse libraries to store

sparse weights. On the other hand, our method easily realizes compression due to weight tying.

We implement layer-pruning as a baseline; many layer-pruning methods rely on similarity measures to choose a set of adjacent layers to prune. However, we avoid any specific similarity techniques and instead choose the best subset after evaluation much like our own technique, via a sliding window and testing on validation data. After selecting the best pruned model, we then fine-tune the model with the same specifications as our method. In all, this encapsulates a strong, structured pruning baseline that generalizes many layer-pruning based techniques.

#### 4.5 OLMo-7B QLoRA Extension

We additionally apply our method alongsize 4-bit QLoRA to OLMo-7B for a downstream summarization task, namely SamSum [\(Groeneveld et al.,](#page-8-12) [2024;](#page-8-12) [Gliwa et al.,](#page-8-14) [2019\)](#page-8-14). SamSum is an English dialogue summarization dataset, and OLMo-7B is a 32-layer English language model trained on the open Dolma dataset [\(Soldaini et al.,](#page-10-14) [2024\)](#page-10-14). This model uses SwiGLU FFs; we discuss extending our method to this variation in Appendix [A.](#page-11-0) We compute features on ~10k tokens from Dolma, and select the best pre-tune model using Wikitext-103. We report results on SamSum using ROUGE-1, ROUGE-2, and ROUGE-Lsum [\(Lin,](#page-9-19) [2004\)](#page-9-19).

