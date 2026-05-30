# *F. Evaluation and expected results*

After executing "./get results.sh", you can use the following script to compare the results with the expected ones:

\$ compare.sh

"compare.sh" uses the "diff" command to compare "overall.csv", "stats.log", and "dse.csv". The "Fig7 heatmaps DSE.svg" file may not be byte-for-byte identical due to font or library version differences, but the data used, namely "dse.csv", should be identical. If all files match, it will output a message like "All files match the expected results!". Otherwise, it will report "Some files do not match the expected results." and highlight the differences.

