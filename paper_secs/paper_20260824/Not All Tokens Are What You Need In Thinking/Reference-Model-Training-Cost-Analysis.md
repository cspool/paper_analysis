# Reference Model Training Cost Analysis

The key parameters for estimating the FLOPs in reference (RM) computation are:

• Model parameters: 7 × 10<sup>9</sup>

• Total tokens processed: 8 × 10<sup>6</sup>

• Training epochs: 3

• Sequence length: 4,096

The FLOPs for the training process are calculated as:

FLOPstrain ≈ 6×Parameters×Sequence Length×Total Tokens (9)

yielding a total computational cost of 4.13 × 10<sup>18</sup> FLOPs.

## Inference Efficiency Gains

For our reference model Qwen-7B-Instruct using CTS-14B at compression ratio 0.7, we observe:

• Average token reduction per inference: 5,300 (across MATH-500, AIME24, and GPQA Diamond)

Using the inference FLOPs estimation from Kaplan et al. (2020):

$$FLOPs_{inf} \approx 2 \times Parameters \times Sequence Length$$
 (10)

## Cost-Benefit Analysis

- FLOPs saved per inference: 1.5 × 10<sup>14</sup>
- Break-even point: 27,500 inference calls

$$N_{\text{break-even}} = \frac{\text{FLOPs}_{\text{train}}}{\text{FLOPs}_{\text{inf}}} \approx 27,500$$
 (11)

Table 6: Experimental results of various compression methods on Qwen2.5-7B-Instruct, showing accuracy, average reasoning CoT tokens, and compression ratio (actual ratio).

| Methods   | Ratio (Actual) | MATH500    |          | AIME24     |          | GPQA Diamond |          |
|-----------|----------------|------------|----------|------------|----------|--------------|----------|
|           |                | Accuracy ↑ | Tokens ↓ | Accuracy ↑ | Tokens ↓ | Accuracy ↑   | Tokens ↓ |
| Original  | 1.0            | 82.4       | 7244     | 20         | 24396    | 37.8         | 17038    |
|           | 0.9(0.89)      | 72.6       | 6804     | 13.3       | 23903    | 39.8         | 14470    |
|           | 0.8(0.81)      | 58.6       | 2969     | 6.7        | 4194     | 34.3         | 4421     |
| LLMLingua | 0.7(0.73)      | 57.2       | 2542     | 6.7        | 3692     | 32.8         | 3236     |
|           | 0.6(0.62)      | 55.0       | 2178     | 3.3        | 3084     | 33.3         | 3203     |
|           | 0.5(0.55)      | 51.4       | 2226     | 3.3        | 3603     | 30.8         | 2462     |
|           | 0.9(0.88)      | 78.6       | 6997     | 23.3       | 24094    | 38.8         | 17263    |
|           | 0.8(0.80)      | 72.8       | 8172     | 10.0       | 25223    | 39.3         | 18365    |
| TokenSkip | 0.7(0.71)      | 64.2       | 9984     | 6.6        | 27946    | 32.3         | 21219    |
|           | 0.6(0.62)      | 54.6       | 11496    | 3.3        | 28802    | 26.2         | 21371    |
|           | 0.5(0.50)      | 37.4       | 13595    | 0          | 29470    | 31.3         | 21012    |
|           | 0.9(0.87)      | 82.8       | 6497     | 20         | 24769    | 43.4         | 17272    |
| CTS       | 0.8(0.81)      | 81.2       | 6886     | 23.3       | 27006    | 39.3         | 17961    |
|           | 0.7(0.74)      | 78.0       | 5109     | 13.3       | 15929    | 42.4         | 13937    |
|           | 0.6(0.66)      | 70.8       | 2198     | 10.0       | 3550     | 32.3         | 3055     |
|           | 0.5(0.58)      | 70.6       | 2039     | 6.7        | 2993     | 32.8         | 3187     |

Table 7: Experimental results of various compression methods on Llama-3.1-8B-Instruct, showing accuracy, average reasoning CoT tokens, and compression ratio (actual ratio).

| Methods   | Ratio (Actual) | MATH500    |          | AIME24     |          | GPQA Diamond |          |
|-----------|----------------|------------|----------|------------|----------|--------------|----------|
|           |                | Accuracy ↑ | Tokens ↓ | Accuracy ↑ | Tokens ↓ | Accuracy ↑   | Tokens ↓ |
| Original  | 1.0            | 63.2       | 12538    | 0.033      | 31013    | 0.363        | 19069    |
|           | 0.9(0.89)      | 54.8       | 9811     | 0          | 27327    | 30.3         | 11727    |
|           | 0.8(0.81)      | 39.6       | 3917     | 3.3        | 5609     | 37.4         | 5471     |
| LLMLingua | 0.7(0.73)      | 36.2       | 3134     | 0          | 3602     | 25.3         | 3323     |
|           | 0.6(0.62)      | 39.0       | 2863     | 0          | 4334     | 32.8         | 3548     |
|           | 0.5(0.55)      | 34.8       | 3172     | 0          | 4188     | 30.3         | 3438     |
|           | 0.9(0.88)      | 56.4       | 11985    | 0          | 27882    | 35.8         | 17846    |
|           | 0.8(0.80)      | 51.2       | 13145    | 3.3        | 30443    | 30.3         | 17960    |
| TokenSkip | 0.7(0.71)      | 44.6       | 14354    | 3.3        | 34249    | 27.7         | 19265    |
|           | 0.6(0.62)      | 32.4       | 15453    | 0          | 23319    | 29.7         | 20800    |
|           | 0.5(0.50)      | 23.3       | 16013    | 0          | 22318    | 26.7         | 22010    |
|           | 0.9(0.87)      | 60.6       | 12047    | 3.3        | 29144    | 32.3         | 18503    |
| CTS       | 0.8(0.81)      | 58.4       | 12134    | 6.7        | 24906    | 40.4         | 18981    |
|           | 0.7(0.74)      | 55.0       | 9987     | 3.3        | 25933    | 32.3         | 16571    |
|           | 0.6(0.66)      | 50.8       | 2808     | 0          | 3781     | 26.7         | 3492     |
|           | 0.5(0.58)      | 45.5       | 2625     | 0          | 3478     | 29.2         | 3080     |