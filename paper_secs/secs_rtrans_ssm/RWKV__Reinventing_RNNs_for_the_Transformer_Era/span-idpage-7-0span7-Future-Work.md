# <span id="page-7-0"></span>7 Future Work

There are several promising directions for future work on the RWKV architecture. Work can be done to increase model expressivity by enhancing the time-decay formulations and exploring initial model states while maintaining efficiency.

The RWKV computational efficiency can be further improved by applying a parallel scan in the wkv<sup>t</sup> step to reduce the computational cost to O(B log(T)d).

The mechanisms used in RWKV can be applied to encoder-decoder architectures, potentially replacing the cross-attention mechanism. This could be applicable in seq2seq or multimodal settings, thereby enhancing efficiency during both training and inference.

RWKV's state (or *context*) can be leveraged for interpretability, predictability in sequence data, and safety. Manipulating the hidden state could also guide behavior and allow greater customizability through prompt tuning.

The RWKV architecture is not perfect, and can be improved via many aspects, such as modifying the formulae or implementing larger internal states. Larger states can enhance the model's memory to previous context and improve performance over various tasks.

