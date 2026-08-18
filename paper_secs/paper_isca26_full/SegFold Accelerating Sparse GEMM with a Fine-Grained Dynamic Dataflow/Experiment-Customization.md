# *Experiment Customization*

Every experiment script supports --jobs N for parallelism control, --config PATH for alternative configurations, and --timeout SEC for per-run time limits. A single matrix can be evaluated directly:

```
./csegfold/build/csegfold \
  --config configs/segfold.yaml \
  --mtx-file benchmarks/data/suitesparse/
    ca-GrQc/ca-GrQc.mtx
```