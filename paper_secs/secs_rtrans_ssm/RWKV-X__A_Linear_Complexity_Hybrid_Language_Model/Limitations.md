# Limitations

While RWKV-X demonstrates strong performance and efficiency in long-context language modeling, several limitations remain. First, its sparse attention mechanism, based on top-k chunk selection, is heuristic and may overlook some semantically relevant dependencies. Second, in our current implementation, sparse attention decoding is slower than that of vanilla RWKV. Further engineering efforts are required to optimize the implementation.

