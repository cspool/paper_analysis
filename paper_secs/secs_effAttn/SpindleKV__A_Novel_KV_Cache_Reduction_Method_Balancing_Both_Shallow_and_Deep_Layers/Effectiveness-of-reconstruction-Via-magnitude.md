# Effectiveness of reconstruction Via magnitude

In SpindleKV, we record the magnitude for each key and value, and reconstruct them to their original magnitude after indexing them from the Code-Book. We compare the performance of SpindleKV with and without the reconstruction operation on LLaMA2-7b-chat. The results in Table [6](#page-8-3) indicate reconstruct operation can effectively reserve model's capability with only a slight memory consumption of the magnitude.

