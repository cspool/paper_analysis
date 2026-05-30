# G. Future Offset Design Ablation

Our scoring function evaluates key importance at multiple future offsets D. We ablate two aspects: the range/number of offsets, and the spacing strategy (Table [E\)](#page-17-0).

Increasing the maximum distance from 128 to 4096 improves accuracy from 41.7% to 48.8% (+7.1%), confirming that future offsets are beneficial. Geometric spacing ({1, 2, 4, . . .}) dramatically outperforms linear spacing (45.8% vs. 28.7%, −17.1%), as near-distance positions require denser sampling.

