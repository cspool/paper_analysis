# *B. ML Inference Applications*

Table [II](#page-6-3) lists the applications we evaluate and their key parameters. Llama is the Llama-3.1-8b [\[9\]](#page-13-16) open-source large language model. It uses a transformer [\[41\]](#page-13-26) DNN with two phases: a highly parallel prefill phase that encodes the input prompts, and a sequential decoding phase to generate the output tokens. The length of each phase and overall performance (throughput and latency) are determined by the input prompt length, batch size, and the number of output tokens.

SD (Stable Diffusion) [\[34\]](#page-13-17) is a popular image generation model. It combines a convolution neural network (CNN) [\[22\]](#page-13-27) and a transformer to encode both image and text data together to generate an output image over multiple iterations. Its performance depends on the resolution of the output image.

VITS [\[20\]](#page-13-19) is a popular text-to-speech model. It combines a transformer and 1D CNN to generate a voice output from text input. Its performance depends mostly on the input length.

Resnet is the Resnet-50 [\[13\]](#page-13-18) image classifier. It reads fixedresolution images and its performance depends only on the batch size. Its compute demand is lower than the rest.

For each application, we consider two levels of requests: *High* and *Low*. *High* uses large batches, high resolution, or long input prompts, demanding more processing, while *Low* uses the opposite.

In our setup, each node runs one application, and in *PGmulti*, all the nodes in a sub-cluster run the same application. Among the processors assigned to an application, half run *Low* requests and half run *High* requests. This induces heterogeneity within and across nodes. We do this because, in practice, requests of similar compute demand are often *batched* and assigned to the same processor [38]. In each processor, requests arrive based on a Poisson process such that, without power constraints, *High* and *Low* induce an average CPU utilization of 60% and 30%, respectively. We report the average and the 95<sup>th</sup> percentile tail (P95) response time.

