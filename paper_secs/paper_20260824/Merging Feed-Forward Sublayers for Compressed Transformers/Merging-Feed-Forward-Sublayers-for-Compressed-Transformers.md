# Merging Feed-Forward Sublayers for Compressed Transformers

Neha Verma<sup>1</sup> Kenton Murray<sup>1</sup>,<sup>2</sup> Kevin Duh<sup>1</sup>,<sup>2</sup> <sup>1</sup>Center for Language and Speech Processing <sup>2</sup>Human Language Technology Center of Excellence Johns Hopkins University {nverma7, kenton}@jhu.edu, kevinduh@cs.jhu.edu

## Abstract

With the ubiquity of large deep learning models and their growing number of use cases, the need for high-quality compression techniques is growing in order to deploy these models widely across diverse hardware and memory settings. In this work, we present a novel approach to model compression by *merging* parameter groups within a model, rather than pruning away less important parameters. Specifically, we select, align, and merge separate feed-forward sublayers in Transformer models, and test our method on language modeling, image classification, and machine translation. With our method, we demonstrate performance comparable to the original models while combining more than a third of model feedforward sublayers, and demonstrate improved performance over a strong layer-pruning baseline. For instance, we can remove over 21% of total parameters from a vision transformer, while maintaining 99% of its original performance. Additionally, we observe that some groups of feed-forward sublayers exhibit high activation similarity, which may help explain their surprising mergeability.

