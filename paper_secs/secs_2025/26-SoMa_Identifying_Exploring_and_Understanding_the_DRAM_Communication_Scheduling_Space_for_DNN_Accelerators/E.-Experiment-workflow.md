# *E. Experiment workflow*

*1) Overall and DSE:* Once the SoMa framework is built, you can reproduce the Overall (Fig. 6) and DSE (Fig. 7) experiments with the command below.

```
$ ./run.sh --eta
```

The "run.sh" takes parameters from "args.txt" as input for each soma instance, including compute power, DRAM bandwidth, storage directory, and the random seed. For each configuration, both our method and the baseline use the same seed. By default, the "run.sh" utilizes all CPU cores, with each core running a separate SoMa process. Each SoMa process outputs the corresponding results and logs to "results/overall" and "results/dse". When using all 192 cores, it takes around 2 days to run on an Intel Xeon Platinum 8260 server. Due to the long runtime, we recommend using tools like "nohup" or "screen" to prevent disconnection due to inactivity, ensuring that all experiments complete successfully.

After all experiments are completed, we use "get results.sh" to extract the results from the raw outputs.

```
$ ./get_results.sh
```

"get results.sh" will use Python scripts under folder "pyscripts" to generate four files: "overall.csv", "stats.log", "dse.csv", and "Fig7 heatmaps DSE.svg". "overall.csv" contains all the data presented in Fig. 6, while "stats.log" includes all the data analyzed and calculated in the Sec. VI-B. "dse.csv" contains all the data related to Fig. 7, and "Fig7 heatmaps DSE.svg" reproduces Fig. 7.

- *2) Comparison with Baselines:* The comparison with the baseline in Fig. 6 can be found in "stats.log". For detailed data of each case, you can refer to "overall.csv". The results of the DSE experiment are available in "Fig7 heatmaps DSE.svg", with the detailed data in "dse.csv", which is also humanreadable, just like "overall.csv".
- *3) SoMa Compiler Workflow:* In "Compiler-IR", we present open-sourced files that showcase the workflow of the end-toend SoMa-based compiler developed for our high-performance commercial AI accelerator, the ZEBU FPGA-based Verification Platform, and the corresponding results. For related information, please refer to "Compiler-IR/Readme.md". (This section is for material demonstration only and does not require execution or reproduction.)

While we are currently unable to release the full source code of the whole compiler due to IP flow restrictions, we believe the provided scheduling engine at the core of SoMa, along with the materials and documentation, effectively offers a clear understanding of the entire workflow of the SoMabased compiler. Additionally, we are committed to establishing a small-scale cloud platform after the chip tape-out and related testing are completed. This platform will allow users to access the open-sourced compiler based on SoMa, with the flexibility to modify or even replace our scheduler (as long as the output is converted into IR format), enabling translation into chip-executable instructions. (This commitment has also been included in Sec. V-F of the paper.)

