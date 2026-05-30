# A.5 Experiment Workflow

To compile DL models into programs and obtain program execution traces from the Elk simulator, we provide a one-click script "benchmark\_scripts/generate\_data\_from\_sim.py" for you to launch all test cases in one place. However, the script may take more than 30 hours to finish, so we recommend using tmux:

tmux

Then, run the one-click script within the new tmux terminal:

```
python3.10 benchmark_scripts/generate_data_from_sim.py
```

To return from the tmux terminal without pausing the script, press "Ctrl+B" and then press "D" on your keyboard. To attach back to the original tmux terminal where the script is running, use:

```
tmux attach -t 0
```

For more tips on using tmux, refer to [https://tmuxcheatsheet.com.](https://tmuxcheatsheet.com)

A.5.1 Handle Errors. If the script encounters an error, the most common cause is that the artifact runs on too many CPU cores and overflows the main memory. In such events, (1) go to "launch.py", (2) change the "CORE\_REDUCE" macro in line 22 to a larger value (e.g., CORE\_REDUCE=8), and (3) rerun the script:

```
python3.10 benchmark_scripts/generate_data_from_sim.py
```

The script should automatically skip any completed test cases and resume from the failed one.

