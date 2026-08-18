# *F. Evaluation and expected results*

The artifact should regenerate the following figure PDFs in figures/:

- Fig. 5: QFT operation-weight distribution
- Fig. 14: QFT gate parallelism
- Fig. 17: Clifford reduction on benchmark set
- Fig. 20: large-QFT reduction trend
- Fig. 21: Synthetiq comparison
- Fig. 22: TRASYN comparison

Numeric outputs are generated as CSV files in results/. The figures should match trends and relative values reported in the paper. Minor numerical or formatting differences may occur due to environment or library variations.

## *G. Experiment customization*

Scripts support standard environment-variable overrides for the Python interpreter and the NWQEC binary path. The README documents available script flags for forced regeneration and figure-specific customization.

