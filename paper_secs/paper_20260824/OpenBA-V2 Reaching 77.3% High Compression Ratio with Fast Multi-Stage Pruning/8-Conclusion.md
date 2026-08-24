# 8 Conclusion

We release OpenBA-V2, an encoder-decoder Transformer model with 3.4B parameters. OpenBA-V2 is derived from the 15B OpenBA model by compressing and continually pre-training. During

the compressing stage, we achieve a compression ratio of 77.3% through multi-stage compression combined with recovery training, with minimal loss in model performance. In the continual pretraining stage, we optimize the UL2 objective and reduce the number of padding tokens in UL2 from about 40% to close to 0, which significantly increases the training efficiency and reduces the waste of resources while bringing almost no loss of model performance. OpenBA-V2 leverages a more diverse dataset and employs multiple levels of filtering strategies to enhance text quality. Overall, OpenBA-V2 demonstrates notable competitiveness among open-source models of similar size.

