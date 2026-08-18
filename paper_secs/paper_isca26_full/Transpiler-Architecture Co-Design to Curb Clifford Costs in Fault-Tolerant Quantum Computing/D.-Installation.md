# *D. Installation*

Build NWQEC from the artifact root directory:

cmake -S . -B build -DCMAKE\_BUILD\_TYPE=Release cmake --build build -j

## *E. Experiment workflow*

The artifact provides automated scripts to reproduce the experimental results reported in the paper.

For convenience, the full workflow can be executed with a single script:

./run\_all.sh

This script performs a fresh build of the NWQEC binary, runs all experiments on the benchmark circuits, collects metrics into CSV files under results/, and generates the corresponding PDF figures under figures/.

Individual figures can also be reproduced using dedicated top-level scripts (one per figure). For example:

./plot\_fig\_5.sh

Each figure script both runs the necessary experiment and generates the final plotted figure. If the corresponding CSV results already exist, the experiment is skipped to save time, and only the figure is generated. Passing the --force-collect flag forces the script to re-run the experiment and regenerate the data.

Detailed usage instructions and script options are provided in the artifact README.

