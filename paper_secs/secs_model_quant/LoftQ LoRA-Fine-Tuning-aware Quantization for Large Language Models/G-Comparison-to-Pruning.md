# G Comparison to Pruning

Pruning is also a widely used compression method. Here we compare LoftQ with the state-of-theart pruning method [Li et al.](#page-15-11) [\(2023\)](#page-15-11). We show the comparison in Table [18.](#page-22-0) We can see our method significantly outperforms the pruning methods on DeBERTaV3-base model. We also remark that LoftQ can consistently reduce the memory of both training and storage. In contrast, pruning requires training the entire full-precision matrix, which implies that it can not achieve any memory savings during the training stage.

