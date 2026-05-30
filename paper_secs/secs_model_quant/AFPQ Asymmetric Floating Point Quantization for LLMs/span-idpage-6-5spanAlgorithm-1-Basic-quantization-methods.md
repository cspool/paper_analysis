# <span id="page-6-5"></span>Algorithm 1 Basic quantization methods

//  $weight_{max}$  represents the maximal value in each weight group, range represents the range of quantization formats

// INT-based algorithm

def INTQuant:

**for** weight group in weight tensor to quantize **do**  $scale = \frac{weight_{max} - weight_{min}}{2}$  $zeropoint = \begin{bmatrix} -weight_{min} \end{bmatrix}$  $weight\_4bit = \left[\frac{weight}{scale}\right] + zeropoint$ 

end

def INTDequant:

for weight group in weight tensor to quantize do  $weight_q = scale(weight_4bit-zeropoint)$ end

## // FP-based symmetric algorithm

def FPSYMQuant:

for weight group in weight tensor to quantize do  $scale = \frac{max(weight_{max}, |weight_{min}|)}{r}$ range/2 $weight\_4bit = \left[\frac{weight}{scale}\right]$ 

end

def FPSYMDequant:

for weight group in weight tensor to quantize do  $| weight_q = scale * weight_4bit$ 

end

// FP-based asymmetric algorithm

def FPASYMQuant:

**for** weight group in weight tensor to quantize **do** 

$$scale_{pos} = \frac{weight_{max}}{range/2}$$

$$scale_{neg} = \frac{-weight_{min}}{range/2}$$

$$weight\_4bit = \left[\frac{weight_{pos}}{scale_{pos}}\right] + \left[\frac{weight_{neg}}{scale_{neg}}\right]$$

def FPASYMDequant:

for weight group in weight tensor to quantize do  $weight\_q = scale_{pos} * weight\_4bit_{pos} +$  $scale_{neg} * weight\_4bit_{neg}$ end

<span id="page-6-4"></span><sup>&</sup>lt;sup>5</sup>https://github.com/TimDettmers/ bitsandbytes

<span id="page-7-0"></span>Table 7: WikiText-2 perplexity and MMLU average accuracy on LLaMA2 models after FP4 RTN quantization

|             |          | LLaMA2-7B |       |       |       | LLaMA2-13B |       |       |       | LLaMA2-70B |       |       |       |
|-------------|----------|-----------|-------|-------|-------|------------|-------|-------|-------|------------|-------|-------|-------|
|             |          | g-1       | g256  | g128  | g64   | g-1        | g256  | g128  | g64   | g-1        | g256  | g128  | g64   |
|             | FP16     | 5.47      |       |       |       | 4.88       |       |       |       | 3.32       |       |       |       |
| WikiText-2↓ | INT4     | 6.12      | 5.75  | 5.72  | 5.67  | 5.20       | 5.02  | 4.98  | 4.97  | 3.67       | 3.49  | 3.46  | 3.44  |
| WIKITEXI-2↓ | FP4-sym  | 5.89      | 5.73  | 5.70  | 5.67  | 5.11       | 5.03  | 5.02  | 5.01  | 3.54       | 3.47  | 3.46  | 3.44  |
|             | FP4-asym | 5.82      | 5.71  | 5.70  | 5.67  | 5.09       | 5.02  | 5.01  | 4.99  | 3.52       | 3.47  | 3.45  | 3.43  |
|             | FP16     | 46.58     |       |       |       | 55.38      |       |       |       | 69.58      |       |       |       |
| MMLU(%)↑    | INT4     | 40.31     | 43.67 | 45.28 | 45.59 | 52.92      | 54.09 | 54.33 | 54.44 | 67.82      | 68.43 | 68.32 | 68.53 |
| WINTEO(76)  | FP4-sym  | 44.14     | 44.25 | 43.74 | 44.04 | 53.77      | 54.17 | 54.83 | 54.62 | 68.14      | 68.72 | 68.71 | 68.90 |
|             | FP4-asym | 45.25     | 44.61 | 45.15 | 44.55 | 54.23      | 54.47 | 54.70 | 54.99 | 68.74      | 68.65 | 68.86 | 69.06 |

<span id="page-7-1"></span>Table 8: WikiText-2 perplexity and MMLU average accuracy on LLaMA2 models after FP3 RTN quantization

|             |          | LLaMA2-7B |       |       |       | LLaMA2-13B |       |       |       | LLaMA2-70B |       |       |       |
|-------------|----------|-----------|-------|-------|-------|------------|-------|-------|-------|------------|-------|-------|-------|
|             |          | g-1       | g256  | g128  | g64   | g-1        | g256  | g128  | g64   | g-1        | g256  | g128  | g64   |
|             | FP16     |           | 4.88  |       |       |            | 3.32  |       |       |            |       |       |       |
| WikiText-2↓ | INT3     | 542.80    | 7.10  | 6.66  | 6.40  | 10.68      | 5.67  | 5.52  | 5.39  | 7.53       | 4.11  | 3.98  | 3.85  |
| WIKITEXt-2↓ | FP3-sym  | 1621.90   | 7.16  | 6.89  | 6.64  | 12.76      | 5.82  | 5.66  | 5.54  | 8.43       | 4.22  | 4.11  | 4.00  |
|             | FP3-asym | 18.72     | 6.89  | 6.63  | 6.48  | 7.72       | 5.69  | 5.57  | 5.41  | 5.93       | 4.11  | 4.01  | 3.89  |
|             | FP16     | 46.58     |       |       |       | 55.38      |       |       |       | 69.58      |       |       |       |
| MMLU(%)↑    | INT3     | 25.22     | 37.46 | 38.50 | 40.06 | 27.79      | 48.91 | 51.23 | 50.77 | 34.39      | 64.77 | 65.05 | 66.16 |
| WINDO(N)    | FP3-sym  | 23.73     | 31.75 | 36.55 | 33.08 | 27.13      | 48.66 | 49.76 | 49.89 | 32.32      | 64.65 | 65.17 | 65.91 |
|             | FP3-asym | 27.32     | 35.42 | 40.33 | 40.24 | 36.15      | 50.09 | 50.72 | 51.60 | 49.74      | 64.62 | 66.14 | 66.41 |

by using look-up tables (LUTs). Then these values can be further dequantized using the methods in Algorithm1.