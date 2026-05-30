# <span id="page-13-0"></span>A Basic Algorithm

We now describe a simple algorithm that finds the optimal quantization values using the dynamic program, with pseudo-code given by Algorithm 3. After initialization (lines 2-4), the algorithm iteratively computes  $MSE[i,\cdot]$  given  $MSE[i-1,\cdot]$  (lines 5-7) and traces back the optimal quantization values given the solution (lines 8-12).

