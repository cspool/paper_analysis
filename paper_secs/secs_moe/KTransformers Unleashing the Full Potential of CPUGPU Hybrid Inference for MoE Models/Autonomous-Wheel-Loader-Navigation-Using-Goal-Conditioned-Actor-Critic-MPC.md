# Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

Aleksi Maki-Penttil ¨ a¨ *Tampere University* aleksi.maki-penttila@tuni.fi

Naeim Ebrahimi Toulkani *Tampere University* naeim.ebrahimitoulkani@tuni.fi

Reza Ghabcheloo *Tampere University* reza.ghabcheloo@tuni.fi

*Abstract*— This paper proposes a novel control method for an autonomous wheel loader, enabling time-efficient navigation to an arbitrary goal pose. Unlike prior works which combine high-level trajectory planners with Model Predictive Control (MPC), we directly enhance the planning capabilities of MPC by incorporating a cost function derived from Actor-Critic Reinforcement Learning (RL). Specifically, we first train an RL agent to solve the pose reaching task in simulation, then transfer the learned planning knowledge to an MPC by incorporating the trained neural network critic as both the stage and terminal cost. We show through comprehensive simulations that the resulting MPC inherits the time-efficient behavior of the RL agent, generating trajectories that compare favorably against those found using trajectory optimization. We also deploy our method on a real-world wheel loader, where we demonstrate successful navigation in various scenarios.

Autonomous Vehicle Navigation, Optimization and Optimal Control, Motion and Path Planning

## I. INTRODUCTION

Wheel loaders are versatile material-moving machines widely used in industries such as mining, construction and waste management. Their articulated frame steering mechanism enables short radius turns and operation in confined spaces. However, the high level of skill required to fully utilize these capabilities leads to significant variability in productivity among human operators [\[7,](#page-6-0) [8\]](#page-6-1). Furthermore, many tasks performed by wheel loaders are repetitive and take place in extreme conditions, such as intense heat or cold, making automation highly desirable.

Prior work on autonomous wheel loader navigation has frequently employed Model Predictive Control (MPC) due to its planning capabilities [\[22,](#page-6-2) [23\]](#page-6-3). However, when paired with a prediction horizon suitable for real-time execution, MPC may fail to solve complex planning tasks on its own. Therefore, most works incorporate a separate high-level trajectory planner, and employ MPC as a reference-tracking controller. For instance, [\[22\]](#page-6-2) uses an RRT\* based planner with adaptive MPC, while [\[23\]](#page-6-3) integrates an optimization-based planner with Linear Parameter Varying MPC (LPV-MPC). Although both of these works successfully demonstrate the goal reaching capabilities of their method in obstacle-free simulations, they inherently balance a trade-off between optimality and real-time capability of the high-level trajectory planner.

This work was supported in part by Academy of Finland (SA) 345517, under SA-NSF joint call on artificial intelligence and wireless communication, and in part by NSF through the RI Grant 2133656. The work is also part of the FEMMa project (2801/31/2021), funded by Business Finland.

![](_page_0_Figure_11.jpeg)

<span id="page-0-0"></span>Fig. 1. Overview of the proposed control system.

Specifically, following outdated trajectories from a nonreal-time planner, like the one in [\[23\]](#page-6-3), will likely lead to suboptimal performance. This is attributed to inevitable deviations from the planned trajectory due to modeling errors. Real-time capable planners, such as the one in [\[22\]](#page-6-2), address this issue by providing a constant stream of upto-date trajectories, thereby minimizing the accumulated deviation from the planned trajectory. However, their realtime execution rate, achieved through sampling or motion discretization, often results in highly suboptimal trajectories.

We propose an alternative control approach which omits the use of a high-level planner, and improves the planning capabilities of the MPC directly. Similar to [\[17\]](#page-6-4), we leverage the fact that a Reinforcement Learning (RL) critic learns to encode all of the information needed to plan approximately optimal behavior. Thus, by integrating a trained RL critic as an MPC cost function, we enable long-term planning even with a prediction horizon suitable for real-time control.

Our method, illustrated in Fig. [1,](#page-0-0) first trains the Actor-Critic RL agent in simulation using an acceleration limited kinematic model of the wheel loader. The trained critic and kinematic model are then used to formulate a nonlinear MPC problem which generates reference velocities for the front body unit and center joint of the wheel loader. These references are subsequently tracked by a low-level feedbackcontroller, which commands the steering hydraulics and diesel engine. This cascade control approach ensures precise navigation despite the use of a simplified model in the MPC. The main contributions of this paper are summarized as:

- We approximate the time-optimal solution of a wheel loader pose reaching task by solving a Goal-Augmented MDP with a Lyapunov-based RL algorithm, trained in simulation using a constrained kinematic model.
- We transfer the planning knowledge of the Actor-Critic RL agent into a nonlinear MPC by formulating suitable terminal and stage costs based on the critic.
- We demonstrate the effectiveness of our solution on a real wheel loader, where the RL actor would be unsafe for direct use. A comprehensive study, supported by simulations, shows that our method reaches goal poses faster than a baseline trajectory optimization routine.

Compared to prior work on RL wheel loader control [4, 21], we support a broader range of applications by enabling navigation to an arbitrary goal pose, instead of only executing specific maneuvers. Our approach also enhances safety by enforcing constraints through MPC. Unlike the conventional Actor-Critic MPC in [17], we use a Lyapunov-based RL algorithm [26] to train a critic that serves as a sampling-based Lyapunov function, providing a mean cost stability guarantee. Moreover, we use the critic not only as a terminal cost in our MPC, but also derive a suitable stage cost from it. Lastly, we propose the use of a gradient penalty during RL training to enhance the MPC optimization landscape.

#### II. PRIOR WORK

#### A. Actor-Critic Model Predictive Control

Prior works [17, 18] have explored different ways of using RL critics as MPC cost functions. In [18], the authors propose a novel Actor-Critic RL algorithm in which the actor consists of a neural network followed by a differentiable MPC. The actor is trained end-to-end, such that the actor network learns to output the stage-wise coefficients of a quadratic cost for the MPC. However, they are unable to enforce state constraints due to inherent limitations in the differentiable MPC solver [2]. In contrast, the authors of [17] train a standard Actor-Critic RL agent and integrate the critic as a terminal cost in a nonlinear MPC problem, which is then solved using a standard Sequential Quadratic Programming (SQP) solver. This allows enforcing state constraints, but due to the highly nonlinear neural network critic, they observe difficulties in solving the underlying optimization problem.

## B. Lyapunov neural networks

Stability analysis using neural Lyapunov functions has received significant attention in the recent years [13, 5, 26, 27, 28]. Some approaches, like [27, 28], use Counterexample-Guided Inductive Synthesis (CEGIS) to construct Lyapunov functions, which provide strong guarantees through verification methods like Satisfiability Modulo Theory (SMT), but struggle in high-dimensional settings. In contrast, methods based on Actor-Critic RL, such as [13, 5, 26], naturally extend to high-dimensional tasks, but rely on weaker forms of Lyapunov theory, such as almost Lyapunov stability [15] or sampling-based Lyapunov stability, which either tolerate small stability violations or focus on stability in expectation.

#### III. PRELIMINARIES

We consider a goal-augmented Markov Decision Process (MDP) [14] with state space  $s \in \mathcal{S} \subseteq \mathbb{R}^n$ , action space  $a \in \mathcal{A} \subseteq \mathbb{R}^n$  and a fixed goal  $g \in \mathcal{S}$ . The state evolves according to the state transition function  $s_{t+1} \sim p(s_{t+1}|s_t, a_t)$ , where the action is sampled from a goal-conditioned behaviour policy  $a_t \sim \pi(a_t|s_t,g)$ . The reward function is defined as  $r(s_t,a_t,g): \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$ , and the associated cost is defined as  $c(s_t,a_t,g) = -r(s_t,a_t,g)$ . Solving the MDP corresponds to finding the policy which minizes the discounted infinite-horizon cost  $J_\pi = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t c(s_t,a_t,g) \mid s_0 = s \right]$ , where  $\gamma \in (0,1]$  is the a discount factor.

We define the stationary distribution of the state transition as  $\mathcal{P}_{\pi}(s_{t+1}|s_t,g) = \int \pi(a|s_t,g)p(s_{t+1}|s,a)da$ . The stationary distribution of the state is subsequently defined as  $\mathcal{S}_{\pi} = \lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T} \mathcal{T}_{\pi}(s_t|s_0,g)$ , where:

$$\mathcal{T}_{\pi}(s_{t+1}|s_0, g) = \int \mathcal{P}_{\pi}(s_{t+1}|s_t, g) \mathcal{T}_{\pi}(s_t|s_0, g) ds_t.$$
 (1)

We assume a  $\cos c(s, a, g)$  defined as a norm between the state and the goal, making it desirable to drive the cost to zero. This enables us to analyze the asymptotic behaviour of the MDP system via the mean cost stability framework [13].

<span id="page-1-2"></span>Definition 3.1: (Mean cost stability [13]). A MDP system is said to be stable in the mean cost under  $\pi$  when:

$$\lim_{t \to \infty} \mathbb{E}_{s_t} \left[ c_{\pi}(s_t, g) \right] = 0 \tag{2}$$

holds for any  $s_0 \in \{s \mid c_{\pi}(s,g) \leq b\}$ , where  $b \geq 0$  is a constant and  $c_{\pi}(s_t,g) = \mathbb{E}_{a_t \sim \pi(a_t|s_t,g)}[c(s_t,a_t,g)].$ 

<span id="page-1-3"></span>Theorem 3.1: (Sampling-based Lyapunov stability [26]). The mean cost stability of a system can be shown through a function  $L(s, g): S \times S \to \mathbb{R}$  when it satisfies the conditions:

<span id="page-1-5"></span><span id="page-1-4"></span>
$$k_l c_{\pi}(s, g) \le L(s, g) \le k_u c_{\pi}(s, g), \tag{3a}$$

$$L(s,g) \ge c_{\pi}(s,g) + \lambda \mathbb{E}_{s' \sim \mathcal{P}_{\pi}} [L(s',g)],$$
 (3b)

$$-k \left( \mathbb{E}_{s \sim \mathcal{S}_{\pi}} \left[ L(s, g) - \lambda \mathbb{E}_{s' \sim \mathcal{P}_{\pi}} \left[ L(s', g) \right] \right] \right)$$

$$\geq \mathbb{E}_{s \sim \mathcal{S}_{\pi}} \left[ \mathbb{E}_{s' \sim \mathcal{P}_{\pi}} \left[ L(s', g) \right] - L(s, g) \right],$$
(3c)

where  $k_l, k_u > 0$  and  $k, \lambda \in (0, 1]$  are constants. When the conditions in (3) are met, then L(s, g) is a valid sampling-based Lyapunov function within  $S_{\pi}$ .

#### <span id="page-1-1"></span>IV. METHODOLOGY

We begin with a high-level overview of our approach before detailing its application to our specific problem. Our approach focuses on controlling a nonlinear system:

$$x_{t+1} = f(x_t, u_t),$$
 (4)

where both the states x and controls u are subject to constraints. Our goal is to drive the system to a goal state  $x_g$  and maintain it there indefinitely. To achieve this, we use the description of the nonlinear system to construct a goal-conditioned MDP, where we assign s=x, a=u,  $g=x_g$ , and define the state transitions of the MDP as deterministic

<span id="page-1-0"></span><sup>&</sup>lt;sup>1</sup>Equivalent to the closed loop dynamics of a stochastic control system.

using  $s_{t+1} = f(s_t, a_t)$ . We then approximate the optimal solution of the MDP through a Lyapunov-based Actor-Critic RL algorithm [26], which trains a policy that stabilizes the MDP system according to Definition 3.1. However, while the RL policy is able to solve the control task in simulation, it is unable to take into account actuator limits and other constraints, which complicates real world deployment.

Because of this, we leverage the fact that the RL algorithm also produces a critic that is compliant with Theorem 3.1, thereby providing a sampling-based Lyapunov function which can be used to synthesize another controller. Specifically, we integrate the critic as both the stage and terminal cost of a nonlinear MPC, allowing it to inherit the RL agents knowledge on how the system can be stabilized. However, unlike the RL policy, the MPC can enforce constraints, thereby making it better suited for real-world control.

#### A. Wheel loader kinematic model

A typical wheel loader is an articulated steering machine, composed of two body units, front and rear as depicted in Fig. 2. Steering is achieved by adjusting the center joint angle  $\beta$  between the two body units using a hydraulic actuator. We denote the pose of the front body unit as  $(x_f, y_f, \theta_f)$ , and its longitudinal velocity as  $v_f$ . The symbols  $L_f$  and  $L_r$  are machine specific parameters, which represent the distance from the center joint to the front and rear axle, respectively.

We adapt the wheel loader kinematic model proposed in [11] to our purpose. Our model of the system is given by:

<span id="page-2-4"></span>
$$\dot{x} = f_c(x, u) = \begin{bmatrix} v_f \cos(\theta_f) \\ v_f \sin(\theta_f) \\ \frac{L_r \dot{\beta} + v_f \sin(\beta)}{L_f \cos(\beta) + L_r} \\ \beta \\ u_1 \\ u_2 \end{bmatrix}, \tag{5}$$

with  $x = [x_f, y_f, \theta_f, \beta, \dot{\beta}, v_f]^T$  and  $u = [u_1, u_2]^T$ . The control  $u_1$  is the angular acceleration of the center joint, while  $u_2$  is the longitudinal acceleration of the front body.

To ensure the simplified kinematic model remains valid, we enforce the following state constraints:

<span id="page-2-2"></span>
$$\begin{bmatrix} -\beta_{max} \\ -\dot{\beta}_{max} \\ -v_{f,max} \end{bmatrix} \le \begin{bmatrix} \beta \\ \dot{\beta} \\ v_{f} \end{bmatrix} \le \begin{bmatrix} \beta_{max} \\ \dot{\beta}_{max} \\ v_{f,max} \end{bmatrix}, \tag{6}$$

where the constraint bounds  $\beta_{max}$ ,  $\dot{\beta}_{max}$  and  $v_{f,max}$  were determined experimentally using the real wheel loader.

Finally, we obtain the discrete-time dynamics  $x_{t+1} = f(x_t, u_t)$  via the 4th order Explicit Runge-Kutta method:

<span id="page-2-1"></span>
$$x_{t+1} = f(x_t, u_t) = \frac{\Delta t}{6} (k_1 + 2k_2 + 2k_3 + k_4),$$
 (7)

where  $\Delta t$  denotes the difference in time between consecutive states  $x_t$  and  $x_{t+1}$ , and  $k_1$ ,  $k_2$ ,  $k_3$ ,  $k_4$  are defined as:

$$k_1 = f_c(x, u),$$
  $k_2 = f_c(x + \frac{\Delta t}{2}k_1, u),$  (8a)

$$k_3 = f_c(x + \frac{\Delta t}{2}k_2, u), \quad k_4 = f_c(x + \Delta t k_3, u).$$
 (8b)

## B. Reinforcement Learning environment design

We train the RL agent in a simulated environment, which uses the discretized kinematic model (7) for state transitions. We enforce the constraint (6) through a clamping mechanism, and set  $\dot{\beta}=0$  when trying to exceed the joint limit.

The cost of the environment is defined as a p-norm using the error vector  $e = [e_{xy}, e_{\theta}, \beta, \dot{\beta}, v_f]^T$  and weights W:

<span id="page-2-3"></span>
$$c(s, a, g) = ||We||_{0.25},$$
 (9)

where the position and heading errors,  $e_{xy}$  and  $e_{\theta}$ , are defined based on a goal state  $g = [x_g, y_g, \theta_g, 0, 0, 0]^T$  as follows:

$$e_{xy} = \sqrt{(x_f - x_q)^2 + (y_f - y_q)^2},$$
 (10a)

$$e_{\theta} = \arctan_2 \left( \sin(\theta_f - \theta_g), \cos(\theta - \theta_g) \right).$$
 (10b)

Importantly, the use of p=0.25 in (9) makes the cost behave in a sparse manner, i.e. giving a nearly constant cost to states which have not converged to the goal. This ensures that the optimal solution of the associated MDP reaches the goal as quickly as possible, since that is the only way to significantly reduce the magnitude of the cost. Other norms, such as the squared Euclidean norm may encourage solutions to initially approach the goal as fast possible, but since the cost values diminish very rapidly when getting closer, there is not enough consistent incentive for time-optimal convergence.

## C. Actor and critic neural networks

We implement our actor and critic, denoted by  $\pi_{\phi}(s,g)$  and  $L_{\psi}(s,a,g)$ , as feedforward neural networks parameterized by  $\phi$  and  $\psi$ . The actor is designed to parameterize a Gaussian action distribution, and the samples from this distribution are bounded by first applying the tanh function and then scaling by the maximum achievable accelerations  $\ddot{\beta}_{max}$  and  $a_{f,max}$ . The critic is constructed as:

$$L_{\psi}(s, a, g) = Q_{\psi}(s, a, g)Q_{\psi}(s, a, g)^{T}, \tag{11}$$

where  $Q_{\psi}(s,a,g)$  is a feedforward neural network parameterized by  $\psi$ . This ensures positive outputs, which is necessary to satisfy (3a) due to the choice of c(s,a,g). We also include an encoder in both networks, which converts all positions into relative positions with respect to the goal and encodes heading angles using sine and cosine values.

![](_page_2_Figure_27.jpeg)

<span id="page-2-0"></span>Fig. 2. Depiction of the wheel loader kinematic model.

#### D. Lyapunov-based Reinforcement Learning

We train our RL agent using the Adaptive Lyapunov-based Actor-Critic (ALAC) [26] algorithm, briefly summarized below. In ALAC, the critic is trained to minimize the mean squared error to a target critic  $L_{target}\left(s,a,g\right)$ :

$$J_c(\psi) = \mathbb{E}_{\mathcal{D}} \left[ \frac{1}{2} \left( L_{\psi}(s, a, g) - L_{target}(s, a, g) \right)^2 \right], \quad (12)$$

where  $J_c(\psi)$  denotes the critic loss and  $\mathcal{D}$  is an off-policy replay buffer. The target critic is defined as:

$$L_{target}(s, a, g) = c(s, a, g) + \gamma \bar{L}_{\bar{\psi}}(s', a', g), \qquad (13)$$

where  $\bar{L}_{\bar{\psi}}$  is identical to  $L_{\psi}$ , except its parameters  $\bar{\psi}$  are defined to be an exponentially moving average of  $\psi$ :

$$\bar{\psi}_{t+1} = (1-\tau)\bar{\psi}_t + \tau\psi_t,$$
 (14)

with  $\tau \in (0,1]$ . The above process corresponds to training the following sampling-based Lyapunov function candidate:

$$L_{\psi}(s,g) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^{t} c(s_{t}, a_{t}, g) \mid s_{0} = s \right],$$
 (15)

which satisfies (3a) and (3b) [26]. However, given the difficulties observed in [17] when optimizing over a neural network critic, we use the following augmented critic loss:

$$\hat{J}_c(\psi) = J_c(\psi) + \rho \mathbb{E}_{\mathcal{D}} \left[ (1 - ||\nabla L_{\psi}(s, a, g)||_2)^2 \right], \quad (16)$$

where the gradient penalty term weighted by  $\rho \ge 0$  encourages the critic to be 1-Lipschitz [12], thereby providing a smoother optimization landscape for the downstream MPC.

Finally, the actor guides the critic to satisfy the remaining condition (3c) by minimizing the following loss:

$$J(\phi) = \mathbb{E}_{\mathcal{D}} \left[ \lambda_e \left( \log \pi_{\phi}(s, g) + \mathcal{H} \right) + \lambda_l \Delta \mathcal{L}_{\psi} \right], \tag{17}$$

where  $\lambda_e$ ,  $\lambda_l \in [0,1]$  are Lagrange variables,  $\mathcal{H}$  is the target entropy and  $\Delta \mathcal{L}_{\psi}$  signifies the sample violation of (3c):

$$\Delta \mathcal{L}_{\psi} = L_{\psi}(s', \pi_{\phi}(s', g)) - L_{\psi}(s, a, g) + k \left( L_{\psi}(s, a, g) - \lambda L_{\psi}(s_{t+1}, \pi_{\phi}(s', g)) \right).$$
(18)

During training, the Lagrange variables  $\lambda_e$  and  $\lambda_l$  are adjusted to maximize their corresponding losses:

$$J(\lambda_e) = \lambda_e \mathbb{E}_{\mathcal{D}} \left[ \log \pi_\phi(s, q) + \mathcal{H} \right], \tag{19a}$$

$$J(\lambda_l) = \lambda_l \mathbb{E}_{\mathcal{D}} \left[ \Delta \mathcal{L}_{\psi} \right]. \tag{19b}$$

Intuitively,  $\lambda_e$  converges to 0 when we satisfy the minimum entropy constraint  $-\log \pi(s,g)_{\phi} \geq \mathcal{H}$ , which has been shown to improve exploration and robustness [6]. Similarly,  $\lambda_l$  trends towards 0 when (3c) is satisfied for certain k and  $\lambda$ . Consequently, when  $\lambda_l < 1$  then  $L_{\psi}$  is a valid sampling-based Lyapunov function for at least  $(s,g) \in \mathcal{D}$ . After each training step the parameters  $k = 1 - \lambda_l$  and  $k = \min(\lambda_l, \gamma)$  self-adjust to prevent premature convergence.

## E. Model Predictive Control problem formulation

Our nonlinear MPC, to be described below, is formulated as the following multiple-shooting optimization problem:

$$\min_{x,u} l_f(x_N, g) + \sum_{n=0}^{N-1} l(x_n, u_n, g), \tag{20a}$$

$$s.t x_0 = x_{initial}, (20b)$$

<span id="page-3-2"></span>
$$x_{i+1} = f(x_i, u_i),$$
  $i = 0, ..., N-1$  (20c)

(6), (22), 
$$i = 1, .., N$$
 (20d)

$$(21), i = 0, ..., N - 1 (20e)$$

where N is the prediction horizon. The predictive model of the MPC is the discretized kinematic model (7), with state  $x = [x_f, y_f, \theta_f, \beta, \dot{\beta}, v_f]^T$  and controls  $u = [u_1, u_2]^T$ , which are the time derivatives of the last two state variables. However, the inputs to our low-level controller are velocities and not accelerations, that is, they are  $(\dot{\beta}_{cmd}, v_{f,cmd})$ . Therefore, at each sample time, we select  $\dot{\beta}$  and  $v_f$  from the first solution state  $x_1$  and send them to the low-level controller, which subsequently commands the real-world machine.

For all stages of MPC, we enforce the state constraint in (6) along with the following constraint on the controls:

<span id="page-3-1"></span>
$$\begin{bmatrix} -\ddot{\beta}_{max} \\ -a_{f,max} \end{bmatrix} \le \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} \le \begin{bmatrix} \ddot{\beta}_{max} \\ a_{f,max} \end{bmatrix}. \tag{21}$$

We also take into account the presence of circular obstacles by enforcing the following constraint at each stage:

<span id="page-3-0"></span>
$$(x_f - x_o^i)^2 + (y_f - y_o^i)^2 \ge (R_o^i)^2,$$
 (22)

where obstacle i is centered at  $(x_o^i, y_o^i)$  and has radius  $R_o^i$ . This formulation assumes that the radius of the obstacle is expaned to include the radius of the wheel loader front body.

Similar to [17], we define the MPC terminal cost  $l_f(x_N, g)$  as the RL critic, with the actions replaced by a zero vector:

$$l_f(x_N, q) = L_{\psi}(x_N, \mathbf{0}_{2\times 1}, q).$$
 (23)

However, we deviate from [17] in defining the stage cost. The stage cost of MPC problem for stage n is defined as:

$$l(x_n, u_n, g) = \Delta t \tilde{L}(x_n, u_n, g), \tag{24}$$

where  $\tilde{L}(x_n, u_n, g)$  is a second order Taylor approximation of the RL critic around the previous MPC solution  $(x^*, u^*)$ :

$$\tilde{L}(x_n, u_n, g) = \frac{\partial L(z_{n+1}^*, g)}{\partial z_n} (z_{n+1}^* - z_n) + 0.5 \frac{\partial^2 L(z_{n+1}^*, g)}{\partial z_n^2} (z_{n+1}^* - z_n)^2,$$
(25)

with  $L(z_{n+1}^*,g)=L_{\psi}\left(x_{n+1}^*,u_{n+1}^*,g\right)$  and the definitions:

$$z_{n+1}^* = \begin{bmatrix} x_{n+1}^* \\ u_{n+1}^* \end{bmatrix}, \quad z_n = \begin{bmatrix} x_n \\ u_n \end{bmatrix}. \tag{26}$$

Without the stage cost defined above, we observed indecisive behaviour from the machine during the non-terminal stages. However, the additional stage cost derived from the critic increases the computational complexity significantly, which was alleviated by the use of a Taylor approximation.

#### V. EXPERIMENTS AND RESULTS

#### A. Experimental setup

We train our RL agent using PyTorch [10], and base our implementation on stable-baslines3 [16]. The actor and critic networks consist of layers with (48,96,144,96,48) hidden units, employing the SoftPlus activation function. The RL agent is trained until convergence to  $\lambda_l=0.8$ , indicating that the critic is a sampling-based-Lyapunov function.

We formulate the MPC problem in (20) using CasADi [3] and Acados [24], using L4CasADi [19, 20] to integrate the neural network critic. The resulting optimization problem is solved using a SQP-RTI scheme, where we utilize the HPIPM [9] QP solver. The solver is warm started using the solution from the previous iteration. During our experiments we use N=10 and  $\Delta t=0.2~s$ , which were chosen to balance the trade-off between real-time capability, length of the prediction horizon and discretization error.

We conduct the real-world experiments using a small Avant 635 wheel loader, shown in Fig. 3, which has been retrofitted for autonomous operation. During these experiments, all computations are performed on an onboard NVIDIA Jetson AGX Orin developer kit. The real-world actuators have an input delay of roughly 200 ms, which we compensate by propagating the MPC initial state forwards by the same amount using (5). The motion constraints used for RL, MPC and the baseline are given in Table I.

## B. Baseline trajectory optimization

The time-efficacy of our method is evaluated through comparisons to the following trajectory optimization procedure:

$$\min_{x,u} \int_{t-0}^{T} \sqrt[4]{e(t)} + \beta(t)^2 + \ddot{\beta}(t)^2 + a_f(t)^2 dt, \quad (27a)$$

$$s.t \quad x(0) = x_{initial}, \ x(T) = x_{goal}, \tag{27b}$$

$$\dot{x} = f_c(x(t), u(t)), \tag{27c}$$

where the dynamics  $f_c(x(t), u(t))$  are given by the kinematic model (5), and the error term e(t), designed to encourage convergence to the goal before t = T, is defined as:

$$e(t) = c_1(x_g - x_f(t))^2 + c_1(y_g - y_f(t))^2 + c_2(1 - \cos(\theta_g - \theta_f(t)))^2 + \epsilon,$$
(28)

where  $\epsilon$ ,  $c_1$ ,  $c_2 > 0$  are constants. We formulate (27) using Casadi [3] and discretize it using direct collocation with a sampling time of 200 ms. The resulting optimization problem is solved using IPOPT [25] with  $T=25\ s$ . This baseline aims to reflect the ideal performance achievable with a traditional control approach, disregarding tracking errors.

| $L_f = L_r$   | 0.6 m  | $\dot{\beta}_{max}$  | 33 deg/s             |
|---------------|--------|----------------------|----------------------|
| $\beta_{max}$ | 40 deg | $\ddot{\beta}_{max}$ | $33 \text{ deg/}s^2$ |
| $v_{f,max}$   | 1 m/s  | $a_{f_{max}}$        | 1 m/s <sup>2</sup>   |

TABLE I

<span id="page-4-1"></span>MOTION CONSTRAINTS USED DURING THE EXPERIMENTS.

![](_page_4_Picture_17.jpeg)

Fig. 3. The Avant 635 wheel loader used in our experiments.

#### <span id="page-4-0"></span>C. Highlighted scenarios

We analyze three specific scenarios in detail: (a) a short loading cycle, (b) a compact 180-degree turn, and (c) navigation through multiple obstacles. Scenarios (a) and (b) were evaluated using the real wheel loader. Scenario (c) had to be simulated using the kinematic model (5), since successful execution required a prediction horizon of N=20, which was too long for real-time execution on the Jetson.

As shown in Figure 5, the Actor-Critic MPC closely matches the baseline's convergence time in scenario (a) and slightly outperforms it in the more challenging scenario (b). The velocity tracking error for scenarios (a) and (b) is illustrated in Figure 4, which highlights that the Actor-Critic MPC achieves this competitive performance despite the presence of minor actuator delays and velocity tracking errors. The difference in execution time is also significant: the baseline trajectory optimization takes over 5 seconds on a desktop AMD Ryzen 3900x CPU, while each iteration of the Actor-Critic MPC problem is solved in less than 100 milliseconds on the much weaker CPU of the Jetson. However, using N=20 for scenario (c) increases the MPC solve time to 200-300 milliseconds, which prevents successful real-world use in obstacle-rich environments.

<span id="page-4-2"></span>![](_page_4_Figure_22.jpeg)

<span id="page-4-3"></span>Fig. 4. Commanded and estimated velocities for scenario (b).

![](_page_5_Figure_0.jpeg)

<span id="page-5-0"></span>Fig. 5. The three highlighted scenarios. (a) Short loading cycle (b) Compact 180-degree turn (c) Navigation through multiple obstacles. Scenarios (a) and (b) were evaluated in the real world, while scenario (c) was conducted using simulations. The time t signifies the first time instant when ||x − g|| < 0.1. For scenario (c) we illustrate trajectories for both N = 10 and N = 20 to highlight the dependence on a sufficiently long prediction horizon.

## *D. Simulation study*

To provide a more comprehensive analysis, we use [\(5\)](#page-2-4) to simulate 128 additional pose reaching scenarios, which are shown in Fig. [6.](#page-5-1) We apply our MPC to these scenarios and measure the time taken to converge within ||x − g|| < 0.1. These convergence times are then compared to those computed for the baseline. The results are drawn to a histogram in Fig. [7](#page-5-2) and key statistics are listed in Table [II.](#page-5-3) Notably, our MPC successfully completes all scenarios, converging 23.80 % faster on average than the baseline. This advantage becomes evident now that a broader set of scenarios are evaluated, and the real-world modeling errors are eliminated.

![](_page_5_Figure_4.jpeg)

<span id="page-5-1"></span>Fig. 6. The scenarios used in our supplementary analysis. Each black arrow represents a goal pose, while the orange arrow depicts the initial pose.

| Metric                       | Mean ± std       | Median  |
|------------------------------|------------------|---------|
| Convergence time of baseline | (14.33 ± 3.72) s | 13.90 s |
| Convergence time of MPC      | (10.92 ± 2.45) s | 10.60 s |
| Improvement over baseline    | 23.80 %          | 23.74 % |

TABLE II

<span id="page-5-3"></span>CONVERGENCE TIME STATISTICS FOR THE 128 SIMULATED SCENARIOS.

![](_page_5_Figure_9.jpeg)

Fig. 7. Histogram of convergence times for the 128 simulated scenarios.

## <span id="page-5-2"></span>VI. CONCLUSION AND FUTURE WORK

We present a novel control scheme based on the Actor-Critic MPC framework, and apply it to solve a wheel loader pose reaching task in a time-efficient manner. Our simulated experiments show that the trajectories generated by our method compare favorably against those found using nonlinear trajectory optimization. Our approach also demonstrates some robustness by succeeding in real-world tests despite the presence of actuator delay and velocity tracking errors.

However, we encounter difficulties in maintaining a realtime control rate when obstacle avoidance is required. Additionally, the MPC occasionally fails to solve certain obstaclefree scenarios in the real world, which we attribute to two main factors: the large number of local minima in the optimization problem, due to the highly nonlinear MPC cost, and the significant actuator delay on our experimental platform, which complicates the control task.

Although our empirical experiments indicate that using a Lyapunov-based RL algorithm along with a gradient penalty is advantageous, their significance needs to be studied more rigorously in future work. Additionally, exploring Control Barrier Functions (CBFs) to shorten the required prediction horizon for obstacle avoidance, similar to [\[29,](#page-6-27) [1\]](#page-6-28), is another promising direction for future research.

## REFERENCES

- <span id="page-6-28"></span>[1] Hossein Abdi et al. "Model Predictive Control Barrier Functions: Guaranteed Safety with Reduced Conservatism and Shortened Horizon". In: *2024 American Control Conference (ACC)*. IEEE. 2024, pp. 1652– 1657.
- <span id="page-6-9"></span>[2] Brandon Amos et al. "Differentiable mpc for end-toend planning and control". In: *Advances in neural information processing systems* 31 (2018).
- <span id="page-6-21"></span>[3] Joel A E Andersson et al. "CasADi – A software framework for nonlinear optimization and optimal control". In: *Mathematical Programming Computation* 11.1 (2019), pp. 1–36.
- <span id="page-6-5"></span>[4] Carl Borngrund et al. "Autonomous navigation of wheel loaders using task decomposition and reinforcement learning". In: *2023 IEEE 19th International Conference on Automation Science and Engineering (CASE)*. IEEE. 2023, pp. 1–8.
- <span id="page-6-11"></span>[5] Ya-Chien Chang and Sicun Gao. "Stabilizing neural control using self-learned almost lyapunov critics". In: *2021 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE. 2021, pp. 1803–1809.
- <span id="page-6-18"></span>[6] Benjamin Eysenbach and Sergey Levine. "Maximum entropy RL (provably) solves some robust RL problems". In: *arXiv preprint arXiv:2103.06257* (2021).
- <span id="page-6-0"></span>[7] Bobbie Frank, Lennart Skogh, and Mats Alakula. "On ¨ wheel loader fuel efficiency difference due to operator behaviour distribution". In: *2nd International Commercial Vehicle Technology Symposium, CVT*. 2012, pp. 1–18.
- <span id="page-6-1"></span>[8] Bobbie Frank et al. "On increasing fuel efficiency by operator assistant systems in a wheel loader". In: *International conference on advanced vehicle technologies and integration (VTI 2012), Changchun, China*. 2012.
- <span id="page-6-25"></span>[9] Gianluca Frison and Moritz Diehl. "HPIPM: a highperformance quadratic programming framework for model predictive control". In: *IFAC-PapersOnLine* 53.2 (2020), pp. 6563–6569.
- <span id="page-6-19"></span>[10] Jacob Gardner et al. "Gpytorch: Blackbox matrixmatrix gaussian process inference with gpu acceleration". In: *Advances in neural information processing systems* 31 (2018).
- <span id="page-6-16"></span>[11] Reza Ghabcheloo and Mika Hyvonen. "Modeling and motion control of an articulated-frame-steering hydraulic mobile machine". In: *2009 17th Mediterranean Conference on Control and Automation*. IEEE. 2009, pp. 92–97.
- <span id="page-6-17"></span>[12] Ishaan Gulrajani et al. "Improved training of wasserstein gans". In: *Advances in neural information processing systems* 30 (2017).
- <span id="page-6-10"></span>[13] Minghao Han et al. "Actor-critic reinforcement learning for control with stability guarantee". In: *IEEE Robotics and Automation Letters* 5.4 (2020), pp. 6217–6224.
- <span id="page-6-15"></span>[14] Minghuan Liu, Menghui Zhu, and Weinan Zhang. "Goal-conditioned reinforcement learning: Problems

- and solutions". In: *arXiv preprint arXiv:2201.08299* (2022).
- <span id="page-6-14"></span>[15] Shenyu Liu, Daniel Liberzon, and Vadim Zharnitsky. "Almost Lyapunov functions for nonlinear systems". In: *Automatica* 113 (2020), p. 108758.
- <span id="page-6-20"></span>[16] Antonin Raffin et al. "Stable-baselines3: Reliable reinforcement learning implementations". In: *Journal of Machine Learning Research* 22.268 (2021), pp. 1–8.
- <span id="page-6-4"></span>[17] Rudolf Reiter et al. "AC4MPC: Actor-Critic Reinforcement Learning for Nonlinear Model Predictive Control". In: *arXiv preprint arXiv:2406.03995* (2024).
- <span id="page-6-8"></span>[18] Angel Romero, Yunlong Song, and Davide Scaramuzza. "Actor-critic model predictive control". In: *2024 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE. 2024, pp. 14777–14784.
- <span id="page-6-23"></span>[19] Tim Salzmann et al. "Learning for casadi: Data-driven models in numerical optimization". In: *6th Annual Learning for Dynamics & Control Conference*. PMLR. 2024, pp. 541–553.
- <span id="page-6-24"></span>[20] Tim Salzmann et al. "Real-time neural MPC: Deep learning model predictive control for quadrotors and agile robotic platforms". In: *IEEE Robotics and Automation Letters* 8.4 (2023), pp. 2397–2404.
- <span id="page-6-6"></span>[21] Tohid Sardarmehni and Xingyong Song. "Path Planning and Energy Optimization in Optimal Control of Autonomous Wheel Loaders Using Reinforcement Learning". In: *IEEE Transactions on Vehicular Technology* 72.8 (2023), pp. 9821–9834.
- <span id="page-6-2"></span>[22] Junren Shi et al. "Planning the trajectory of an autonomous wheel loader and tracking its trajectory via adaptive model predictive control". In: *Robotics and Autonomous Systems* 131 (2020), p. 103570.
- <span id="page-6-3"></span>[23] Ruitao Song et al. "Autonomous wheel loader trajectory tracking control using lpv-mpc". In: *2022 American Control Conference (ACC)*. IEEE. 2022, pp. 2063–2069.
- <span id="page-6-22"></span>[24] Robin Verschueren et al. "acados—a modular opensource framework for fast embedded optimal control". In: *Mathematical Programming Computation* 14.1 (2022), pp. 147–183.
- <span id="page-6-26"></span>[25] Andreas Wachter and Lorenz T Biegler. "On the ¨ implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming". In: *Mathematical programming* 106 (2006), pp. 25–57.
- <span id="page-6-7"></span>[26] Shengjie Wang et al. "A Policy Optimization Method Towards Optimal-time Stability". In: *Conference on Robot Learning*. PMLR. 2023, pp. 1154–1182.
- <span id="page-6-12"></span>[27] Junlin Wu et al. "Neural lyapunov control for discretetime systems". In: *Advances in neural information processing systems* 36 (2023), pp. 2939–2955.
- <span id="page-6-13"></span>[28] Lujie Yang et al. *Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation*. 2024. arXiv: [2404.07956 \[cs.LG\]](https://arxiv.org/abs/2404.07956).
- <span id="page-6-27"></span>[29] Jun Zeng, Bike Zhang, and Koushil Sreenath. "Safetycritical model predictive control with discrete-time control barrier function". In: *2021 American Control Conference (ACC)*. IEEE. 2021, pp. 3882–3889.