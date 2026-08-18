# *G. Evaluation*

The log files contain information on IPC, the numbers for each cache access type at different cache levels, and the DRAM utilization information. The memory access file contains information on the time, the addresses and the types of the cache accesses. A file containing the time and the addresses of all successfully issued prefetches is also generated for detailed timing analysis.

1) Use the ./sim\_analyze/process\_log.py script to process the data to extract speedup, prefetch coverage, prefetch accuracy and DRAM utilization. Refer to the README.md file for instructions on how to use the script. The command will ask if processing CVP traces. If not processing CVP traces, input the path to the weights file, which can be found in ./sim analyze/weights.csv in this repository. The script will output a csv file that

- shows speedups, prefetch coverage, prefetch accuracy and DRAM utilization.
- 2) Use the ./sim\_analyze/process\_conv.py script to check for simulation convergence.

