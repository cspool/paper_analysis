# *Experiment Workflow*

A single command reproduces every experiment:

```
./scripts/run_all.sh
```

Outputs are placed in output/ae\_<timestamp>/. Each figure or table can also be generated independently via a standalone script:

- ./scripts/run\_figure\_overall.sh → Figure 8
- ./scripts/run\_figure\_nonsquare.sh → Figure 9
- ./scripts/run\_figure\_mapping.sh → Figure 10
- ./scripts/run\_figure\_breakdown.sh → Figure 11
- ./scripts/run\_figure\_crossbar\_width.sh → Figure 12(a)
- ./scripts/run\_figure\_window\_size.sh → Figure 12(b)
- ./scripts/run\_k\_reordering.sh → §IV-C ablation result

