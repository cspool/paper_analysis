# Artifact Execution

1.1: The benchmark commands for 1.<sup>1</sup> are python exps-1.0 /run\_tasks.py 0 and python exps-1.0/run\_tasks2.py. The first argument of run\_tasks.py specifies benchmarking without NCU profiling. After executing the scripts, experimental data will be saved in checkpoints/cache.

1.2: The measured data is collected by executing python exps-1.0/

run\_tasks.py 1, where the first argument specifies benchmarking with NCU profiling.

1.3: All data required for 1.<sup>3</sup> has been generated during the experiments of 1.1.

1.<sup>4</sup> The experimental data will be collected by executing command MNs=0 MNe=1000 python L1\_THRES=1.3 L2\_THRES=1.4 run\_tasks.py.

# Artifact Execution

The baseline results can be reproduced by executing the command python exps-1.0/run\_tasks2.py and the primary experiments are conducted by running python exps-1.0/run\_tasks.py 1. Besides, To generate the corresponding figures and tables, follow these instructions:

```
cd exps-1.0
# fig9
python fig9.py
# table3
python table3.py
# fi10
python fig10.py
# fi11
python fig11.py
```

The resulting images and tables will be saved in the exps-1.0/imgs directory.

