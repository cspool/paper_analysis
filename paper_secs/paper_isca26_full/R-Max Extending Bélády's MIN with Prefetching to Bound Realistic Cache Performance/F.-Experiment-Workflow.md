# *F. Experiment Workflow*

Please use the following steps to run the simulations:

- 1) Clone the repository.
- 2) Setup the simulator with vcpkg to install dependencies.
- 3) Use ./sim\_compile/compile\_all.sh to compile the binaries for all configurations.
- 4) Create a directory for the benchmark suite that one wishes to run. Change directory to the created directory.
- 5) Use ./sim\_run/generate\_commands.sh <trace directory> <binary directory> command to setup the directory structure for the pending simulations. The command will generate two job files: phase\_1\_jobs.txt and phase\_2\_jobs.txt. Each line in each job file represents a single job that can run in parallel. Note that if running CVP-1 traces, the script asks for the file that has the length of each trace. The trace length file is located at ./sim\_run/cvp\_public\_trace\_length.txt.
- 6) Submit all jobs from phase 1 jobs.txt. Wait for all of them to finish.
- 7) Run ./sim\_run/copy\_translations.sh in the root level of the previously created directory in step 4 to copy the corresponding translations and/or prefetches issued by SPP or Berti. The translations will be used by R-Max, MIN replacement policy, SPP-Max and Berti-Max.
- 8) Submit all jobs from phase 2 jobs.txt. Wait for all of them to finish.

