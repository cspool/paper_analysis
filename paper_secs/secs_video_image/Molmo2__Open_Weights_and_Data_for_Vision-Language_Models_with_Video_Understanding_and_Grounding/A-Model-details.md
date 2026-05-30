# **A Model details**

We present additional details about image encoding, hyperparameters, and implementation choices.

**Image crops**. Our method of encoding images largely follows Molmo [\[29\]](#page-18-2), including the use of overlapping crops. Unlike Molmo, we do not pad crops with black. Instead, we resize them to 378 (even if that means changing the aspect ratio), following how SigLIP 2 [\[139\]](#page-23-17) was trained. If the number of image patches is not evenly divisible by the pooling size, the bottom and far-right image patches are pooled with a reduced number of patches.

**Video frames.** We use torchcodec[3](#page-26-1) to extract frames from videos. We extract frames at S fps and the last frame. If that leads to more than F frames, we instead extract frames uniformly, including the first and last frames. For tracking, during training, we always sample videos at S fps and trim both videos and point tracks to a maximum of F frames instead. This ensures that points, which are annotated for S fps, remain aligned with the sampled frames. We include the last frame since it is typically what is shown when the video ends and, therefore, can have special importance to users. Frames are extracted based on timestamps (instead of frame indices) to handle variable fps videos.

**Formatting.** Videos and image tokens are always inserted first, right after the BOS token. We insert different start and end special tokens for videos, tokens from a multi-crop image, and tokens for the low-resolution single-crop version of the image. Frames are interleaved with text timestamps written as seconds to one decimal point, and multi-images are interleaved with "Image 1", "Image 2", etc., labels. Text is added after the image/video tokens following the Qwen3 [\[169\]](#page-24-0) prompt template without thinking tokens.

**Pointing.** Our pointing format provides points in an HTML-like format, with the coordinates stored in a compact string. For each frame or image with points, the string contains an image index (for image input, starting at 1) or a frame timestamp (for video, shown in seconds with one decimal point), followed by a list of point coordinates. The points each have an object index, which is unique for each distinct object being pointed at, and x and y coordinates that are normalized to be between 0 and 1000. Object indices are sequential, starting at 1. The object indices both facilitate counting, because the final object index represents the total count, and enable tracking by identifying repeating objects. Points are sorted by time/frame index and then by x and y coordinates. Values are space-separated, with semi-columns indicating a new frame/image. We elect to use this format over a format like JSON since it dramatically reduces the number of tokens needed to represent points.

An example output for a pointing and tracking task are shown below (new lines added for clarity):

<points coords="1 1 555 169;2 3 649 154 4 709 162;5 5 758 175 6 808 183 7 852 187"> Inline text

<span id="page-26-1"></span><sup>3</sup> <https://pytorch.org/blog/torchcodec/>

```
</points>
<tracks coords="0.0 1 635 522;0.5 1 606 490 2 511 124;1.0 2 515 164;1.5 2 520 168">
Inline text
</tracks>
```

Where image indices and frame timestamps are in blue, object indices are in purple, and x and y coordinates are in green. The first example points to an object in images 1, 2, and 5. The second one tracks two different objects through several frames. The "Inline text" is used to describe what is being pointed at.

**Hyperparameters.** Hyperparameters for the Molmo2 models are shown in Table 12. The connector MLP uses the same intermediate dimension as the LLM, so its size depends on the LLM; otherwise, they are the same across all models. All models use the SigLIP 2 So400m/14 384px ViT [139].

Implementation. Our implementation uses PyTorch with Fully Sharded Data Parallel (FSDP) 2 [185]. We use PyTorch's Scaled Dot Product Attention (SDPA), not FlashAttention [28, 27], since it does not support custom attention masks. We use torch.compile to improve throughput and ensure that the shapes in the LLM and ViT are static so the model can be statically compiled, which we find essential for maximizing throughput.

To improve throughput, we also utilize PyTorch's Automatic Mixed Precision (AMP) module<sup>4</sup>, which enables most operations to run in half-precision with bfloat16 numbers. Computations for layer normalization [9] and Rotary Position Embedding (RoPE) [131] are still carried out in full precision.

When computing gradients, each GPU computes a gradient on a small mini-batch of examples, after which the gradients are averaged across all devices. We always compute the per-device gradient by dividing the total loss on that device by the *average* number of loss tokens across all devices, not the number of loss tokens on that particular device. This avoids a subtle bias that effectively up-weights examples with a small number of loss tokens  $(e.g., with short responses)^5$  [51].

During fine-tuning, mixing is done within each batch so that the batches contain examples from a variety of datasets. We truncate examples that are longer than the max sequence length. This occurs in < 0.1% of cases, usually due to videos with both subtitles and a large number of annotations. We find training to be stable, without loss spikes or NaNs.

