# *C. Necessity of Two-staged Compression*

While the coarse-grained step provides the largest compression gains by removing entire irrelevant functions, the finegrained compression step is crucial for balancing the trade-off between compression overhead and task model cost. Users can disable the fine-grained step for faster, cheaper compression when using less expensive models. However, for powerful but costly APIs like Claude-3.7-Sonnet, the precise pruning from the fine-grained step becomes critical, yielding substantial cost reductions that justify the additional computational overhead. This adaptive design allows LongCodeZip to accommodate different deployment scenarios and cost constraints.

