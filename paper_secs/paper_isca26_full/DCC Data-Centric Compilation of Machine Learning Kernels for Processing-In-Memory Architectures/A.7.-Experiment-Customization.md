# A.7. Experiment Customization

In kernel performance experiments, users can customize the workload configuration, including batch size, input and output token sizes and number of heads by adding following function call at the end of the src/test\_configs.py script, where the name parameter should be set to "ATTN", "GEMV", "RED", "VA", or "RELU" depending on the target ML kernel:

custom\_kernel\_sizes(name, batch, input, output, nhead)

For LLM inference performance experiments, users can customize the workload configuration, including batch size, input and output token sizes, by first adding following function call at the end of the src/test\_configs.py script:

```
custom_inference_sizes(batch, input, output)
```

and then by adding following command (in one line) at the end of the run\_experiemtns.sh script:

```
conda run -n dcc -live-stream python3
run_inference.py -batch=batch -lin=input
-lout=output
```

Note that the plotting scripts do not support custom experiment configurations. To avoid errors, please comment out the last two lines in the run\_experiments.sh script, and update the corresponding plotting script accordingly to match the custom configurations.# A.7. Experiment Customization

In kernel performance experiments, users can customize the workload configuration, including batch size, input and output token sizes and number of heads by adding following function call at the end of the src/test\_configs.py script, where the name parameter should be set to "ATTN", "GEMV", "RED", "VA", or "RELU" depending on the target ML kernel:

custom\_kernel\_sizes(name, batch, input, output, nhead)

For LLM inference performance experiments, users can customize the workload configuration, including batch size, input and output token sizes, by first adding following function call at the end of the src/test\_configs.py script:

```
custom_inference_sizes(batch, input, output)
```

and then by adding following command (in one line) at the end of the run\_experiemtns.sh script:

```
conda run -n dcc -live-stream python3
run_inference.py -batch=batch -lin=input
-lout=output
```

Note that the plotting scripts do not support custom experiment configurations. To avoid errors, please comment out the last two lines in the run\_experiments.sh script, and update the corresponding plotting script accordingly to match the custom configurations.