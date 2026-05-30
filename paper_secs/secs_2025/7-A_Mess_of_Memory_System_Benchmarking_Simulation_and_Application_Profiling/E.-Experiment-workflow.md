# *E. Experiment workflow*

For each experiment (each folder in Figures 17 or 18), the workflow for running the Mess benchmark/simulation is provided in the runner.sh script. For Figure 6.b of the main manuscript (trace-driven DRAMsim3 simulation), the full workflow takes less than one day.

#### *F. Evaluation and expected results*

To replicate each experiment, we run replicate.sh script. This script compiles necessary codes, executes the workflow (i.e., runner.sh), and processes the output raw data. The artifact also includes all the raw measurements and final processed .csv data. For the DRAMsim3 example, the replicate.sh script inside DRAMsim3 directory (Figure 19) executes the following commands:

```
# u n z i p t r a c e f i l e s
cd t r a c e I n p u t
f o r f i l e i n * . z i p ; do
      u n zi p " $ f i l e "
done
cd . .
# c o m pil e DRAMsim3
cd DRAMsim3 mn5
m k di r b u i l d
cd b u i l d
cmake . .
make
cd . . / . .
# r u n t h e e x p e r i m e n t
. / r u n n e r . s h
# ##################
# P o st − p r o c e s s i n g #
# ##################
# g e n e r a t e r e s u l t s . c s v f i l e
p yt h o n 3 main . py .
# g e n e r a t e o u t p u t . p df ( b a n dwi dt h −−l a t e n c y c u r v e s )
p yt h o n 3 c o n v e r t . py
```

The easiest way to validate the result is visually by examining the generated curves (e.g., output.pdf in our example). However, if one wants to evaluate the results in more detail, the results.csv file can be compared to results\_original.csv; rows with the same rw ratio and pause values should have a very close latency and bandwidth.

