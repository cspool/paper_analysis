# A RULES DERIVATION

At any point during the out-of-order simulation, the state includes each agent's location and the step they have executed. For a given state, let dist(A, B) represent the distance between agents A and B, radius p denote the radius of an agent's perception, and max vel denote the maximum speed of agent movement and information propagation per step. According to Section [3.2,](#page-3-0) a valid execution needs to make sure the following condition holds at any state:

$$\forall$$
 agents  $A,B,$  and their current steps  $Step_A$  and  $Step_B,$  if  $Step_A \neq Step_B,$  then  $dist(A,B) > radius\_p + (|Step_A - Step_B| - 1) \times max\_vel$ 

To satisfy the condition, we derive the following simulation conditions to ensure the state remains valid. Notice that our simulation conditions are over-estimations, which is sound in correctness but not necessarily complete.

Starting from any valid state, given any two agents A and B at steps Step<sup>A</sup> and StepB, respectively, we derive the simulation conditions of A by case study. Let A′ denote the new state of A after one more step so that dist(A′ , B) denotes the new distance between A and B after the next step of A.

• Assume steps Step<sup>A</sup> = StepB. A is allowed to proceed to the next step if a valid state is reached after one further step of A. Formally, there should be

$$dist(A', B) > (Step_A - Step_B + 1 - 1) \times max\_vel + radius\_p$$

Since 
$$dist(A', B) \ge dist(A, B) - max\_vel$$
, we need:

$$dist(A, B) - max\_vel > radius\_p$$

Therefore, on the other side, A and B must stay at the same step if:

$$dist(A, B) \leq max\_vel + radius\_p$$

which means they are *coupled* and can either wait together or proceed together.

• Assume steps Step<sup>A</sup> > StepB. There should be dist(A ′ , B) >(Step<sup>A</sup> − Step<sup>B</sup> + 1 − 1) × max vel

+ radius p

Since dist(A′ , B) ≥ dist(A, B)−max vel, we need:

$$dist(A, B) - max\_vel$$
  
>  $radius\_p + (Step_A - Step_B) \times max\_vel$ 

Therefore, on the other side, A got *blocked* by B if

$$dist(A, B) \le (Step_A - Step_B + 1) \times max\_vel + radius\_p$$

• Assume steps Step<sup>A</sup> < StepB. There should be

$$dist(A', B) > (Step_B - Step_A - 1 - 1) \times max\_vel + radius\_p$$

Since dist(A′ , B) ≥ dist(A, B)−max vel, we need:

$$\begin{aligned} dist(A,B) - max\_vel \\ > (Step_B - Step_A - 2) \times max\_vel + radius\_p \end{aligned}$$

$$\Rightarrow dist(A, B) > (Step_B - Step_A - 1) \times max\_vel + radius\_p$$

which is the same valid condition of the current state. Therefore, A is not *blocked* by any future agents.