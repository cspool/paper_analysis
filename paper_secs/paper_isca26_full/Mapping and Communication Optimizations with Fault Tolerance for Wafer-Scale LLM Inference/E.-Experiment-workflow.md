# *E. Experiment workflow*

\$ pip install -r requirements.txt

We provide a single script to run all the simulation experiments. The command is shown below:

```
# Run all the simulations
# with configurable parallel jobs (default 16)
$ bash run_all.sh 16
```

This generates configurations, runs all experiments, plots all 12 figures, and collects them into output/.

*a) Quick validation (*<*3 hours).:* We also provide a quick validation script that covers six representative figures:

```
# Run quick validation subset
$ bash run_quick_test.sh 16
```

This validates core functionality with AllGather communication (Fig. 8a), power/fault mapping (Fig. 10c), workload heatmap (Fig. 11), runtime breakdown (Fig. 15), and GPT-NeoX end-to-end (Fig. 12, GPT only).

