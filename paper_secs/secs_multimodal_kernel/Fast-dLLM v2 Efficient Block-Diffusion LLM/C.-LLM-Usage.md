# C. LLM Usage

During manuscript preparation, we used large language models —strictly for language polishing of paragraphs and sentences (grammar, flow, and tone). These tools were not used to generate ideas, design experiments, or determine conclusions. All technical content, methodology, and interpretations were written, verified, and approved by the authors. To reduce risks of factual drift or citation errors, we required human review of every model-edited sentence and cross-checked all references against primary sources. The authors take full responsibility for the accuracy and integrity of the manuscript.

### Table 5 | Single-turn Dialogue Cases of Fast-dLLM v2 (7B).

Math

**User** How many vertical asymptotes does the graph of  $y = \frac{2}{x^2 + x - 6}$  have?

Assistant To determine the number of vertical asymptotes of the function  $y = \frac{2}{x^2 + x - 6}$ , we need to identify the values of x that make the denominator zero, as these values will cause the function to approach infinity, indicating a vertical asymptote.

First, we factor the denominator  $x^2 + x - 6$ . We look for two numbers that multiply to -6 and add up to 1. These numbers are 3 and -2. Therefore, we can factor the quadratic as follows:

$$x^{2} + x - 6 = (x + 3)(x - 2)$$

So the function can be rewritten as:

$$y = \frac{2}{(x+3)(x-2)}$$

The denominator is zero when x + 3 = 0 or x - 2 = 0. Solving these equations, we get:

$$x=-3 \quad \text{and} \quad x=2$$

Since the function is undefined at x=-3 and x=2, there are vertical asymptotes at these points. Therefore, the function  $y=\frac{2}{x^2+x-6}$  has two vertical asymptotes.

The number of vertical asymptotes is  $\boxed{2}$ 

Code

**User** Write a recursive Python function to solve the Tower of Hanoi problem and print each move.

**Assistant** 

```
def tower_of_hanoi(n, source, target, auxiliary):
    if n > 0:
        tower_of_hanoi(n - 1, source, auxiliary, target)
        print(f"Move disk {n} from {source} to {target}")
        tower_of_hanoi(n - 1, auxiliary, target, source)

tower_of_hanoi(3, 'A', 'C', 'B')
```

<span id="page-15-0"></span>This function solves the Tower of Hanoi problem using recursion. It takes four parameters: n (the number of disks), source (the starting peg), target (the destination peg), and auxiliary (the helper peg). The function prints each move in the form of "Move disk n from source to target".

## Table 6 | Multi-turn Dialogue Cases of Fast-dLLM v2 (7B).

<span id="page-16-0"></span>*Daily life*

User When was NVIDIA founded?

Assistant NVIDIA was founded in 1993.

User Using the year you just gave me, calculate how many years NVIDIA has been in business as of 2025.

Assistant NVIDIA has been in business for 32 years as of 2025.

User How many years will it be until NVIDIA's 40th anniversary?

Assistant NVIDIA's 40th anniversary will be in 2033, which is 8 years from now.

*Math*

User A delivery truck can carry 120 boxes in one trip. If a warehouse needs to send out 960 boxes, how

many trips will the truck need?

Assistant To determine how many trips the delivery truck will need to send out 960 boxes, we start by noting the capacity of the truck in one trip. The truck can carry 120 boxes in one trip. We need to find out how many trips are required to transport 960 boxes.

We can set up the problem as a division problem:

Number of trips = 
$$\frac{\text{Total number of boxes}}{\text{Capacity of the truck per trip}}$$

Substituting the given values into the equation, we get:

Number of trips = 
$$\frac{960}{120}$$

Next, we perform the division:

$$\frac{960}{120} = 8$$

Therefore, the truck will need 8 trips to send out 960 boxes.

User If each trip takes 45 minutes, how long will it take to deliver all boxes in total? Give your answer in hours and minutes.

Assistant To determine the total time required to deliver all the boxes, we need to follow these steps:

- 1. Calculate the number of trips required.
- 2. Determine the total time taken for all the trips.
- 3. Convert the total time from minutes to hours and minutes.

## Step 1: Calculate the number of trips required.

The warehouse needs to send out 960 boxes, and the truck can carry 120 boxes per trip. Therefore, the number of trips required is:

Number of trips 
$$=$$
  $\frac{960}{120} = 8$ 

#### Step 2: Determine the total time taken for all the trips.

Each trip takes 45 minutes, so the total time taken for 8 trips is:

Total time = 
$$8 \times 45 = 360$$
 minutes

#### Step 3: Convert the total time from minutes to hours and minutes.

There are 60 minutes in an hour, so we convert 360 minutes to hours by dividing by 60:

$$360 \div 60 = 6 \text{ hours}$$

Therefore, the total time required to deliver all the boxes is:

6 hours