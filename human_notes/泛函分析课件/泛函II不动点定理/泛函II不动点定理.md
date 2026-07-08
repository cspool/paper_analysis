# 不动点定理及应用

不动点理论与许多数学分支有紧密的联系, 特别是在研究各 类方程解的存在与唯一性问题时起着重要作用. 不动点理论在数 学之外的其它领域也有广泛而重要的应用.

定义 1: 设 X 是任一集合, 给定映射  $F: X \to X$ . 如果点  $x \in X$  满足 Fx = x, 那么称  $x \in F$  的一个不动点.

定义 1: 设 X 是任一集合, 给定映射  $F: X \to X$ . 如果点  $x \in X$  满足 Fx = x, 那么称  $x \in F$  的一个不动点.

定义2: 设 (X,d) 是一个距离空间, 映射  $F:X\to X$  满足

$$d(Fx, Fy) \le d(x, y), \quad \forall x, y \in X,$$
 (1)

那么称  $F \in X$  上的压缩映射.

定义 1: 设 X 是任一集合, 给定映射 F : X → X. 如果点 x ∈ X 满足 F x = x, 那么称 x 是 F 的一个不动点.

定义2: 设 (X, d) 是一个距离空间, 映射 F : X → X 满足

$$d(Fx, Fy) \le d(x, y), \quad \forall x, y \in X,$$
 (1)

那么称 F 是 X 上的压缩映射.

如果存在常数 µ < 1, 使得

$$d(Fx, Fy) \le \mu d(x, y), \quad \forall x, y \in X,$$
 (2)

那么称 F 是 X 上的严格压缩映射.

# 压缩映射 F 是<mark>连续映射</mark>:

### 压缩映射 F 是连续映射:

对于任意给定的 ε > 0, 取 δ = ε, 那么当 x, y ∈ X 满足 d(x, y) < δ 时, 就有

$$d(Fx, Fy) \le d(x, y) < \delta = \varepsilon.$$

# Banach不动点定理(压缩映象原理)

定理1: 完备距离空间上的严格压缩映射存在唯一不动点.

### Banach不动点定理(压缩映象原理)

定理1: 完备距离空间上的严格压缩映射存在唯一不动点.

**证明** 设 (X,d) 是完备的距离空间, F 是 (X,d) 到其自身的严格压缩映射.

### Banach不动点定理(压缩映象原理)

定理1: 完备距离空间上的严格压缩映射存在唯一不动点.

证明 设 (X, d) 是完备的距离空间, F 是 (X, d) 到其自身 的严格压缩映射.

任意取一点 x<sup>0</sup> ∈ X, 构造迭代序列如下,

$$x_1 = Fx_0, x_2 = Fx_1, \dots, x_n = Fx_{n-1}, \dots$$

### 对任何正整数n,

$$d(x_{n+1}, x_n) = d(Fx_n, Fx_{n-1})$$

$$\leq \mu d(x_n, x_{n-1})$$

$$= \mu d(Fx_{n-1}, Fx_{n-2})$$

#### 对任何正整数n,

$$d(x_{n+1}, x_n) = d(Fx_n, Fx_{n-1})$$

$$\leq \mu d(x_n, x_{n-1})$$

$$= \mu d(Fx_{n-1}, Fx_{n-2})$$

$$\leq \mu^2 d(x_{n-1}, x_{n-2})$$

### 对任何正整数n,

$$d(x_{n+1}, x_n) = d(Fx_n, Fx_{n-1})$$

$$\leq \mu d(x_n, x_{n-1})$$

$$= \mu d(Fx_{n-1}, Fx_{n-2})$$

$$\leq \mu^2 d(x_{n-1}, x_{n-2})$$

$$\cdots$$

$$\leq \mu^n d(x_1, x_0).$$

$$\begin{array}{ll} d(x_n,x_m) & \leq \sum\limits_{j=m+1}^n d(x_j,x_{j-1}) \ & \leq \mu^m (1-\mu)^{-1} d(x_1,x_0) \ & \to 0, \quad m \to \infty. \end{array}$$

$$d(x_n, x_m) \le \sum_{j=m+1}^n d(x_j, x_{j-1})$$
  
  $\le \mu^m (1-\mu)^{-1} d(x_1, x_0)$   
  $\to 0, \quad m \to \infty.$ 

所以  $\{x_n\}_{n=1}^{\infty}$  是 Cauchy 列.

$$\begin{array}{ll} d(x_n,x_m) & \leq \sum\limits_{j=m+1}^n d(x_j,x_{j-1}) \ & \leq \mu^m (1-\mu)^{-1} d(x_1,x_0) \ & \to 0, \quad m \to \infty. \end{array}$$

所以  $\{x_n\}_{n=1}^\infty$  是 Cauchy 列. 由 (X,d) 完备,知存在  $x\in X$ ,使得  $x_n\to x$ , $n\to\infty$ .

$$\begin{array}{ll} d(x_n,x_m) & \leq \sum\limits_{j=m+1}^n d(x_j,x_{j-1}) \ & \leq \mu^m (1-\mu)^{-1} d(x_1,x_0) \ & \to 0, \quad m \to \infty. \end{array}$$

所以  $\{x_n\}_{n=1}^{\infty}$  是 Cauchy 列.

由 (X,d) 完备, 知存在  $x \in X$ , 使得  $x_n \to x$ ,  $n \to \infty$ . 由 F 的连续性可得

$$Fx = \lim_{n \to \infty} Fx_n = \lim_{n \to \infty} x_{n+1} = x.$$

表明 x 是 F 的不动点.

### 不动点的唯一性.

若存在 x, y ∈ X 满足 F x = x, F y = y, 那么

$$d(x,y) = d(Fx, Fy) \le \mu d(x,y).$$

注意到 µ < 1 且 d(x, y) ≥ 0, 可见 d(x, y) = 0, 从而 x = y.

一般的压缩映射, 即使不等式严格成立, 也不一定有不动点.

一般的压缩映射, 即使不等式严格成立, 也不一定有不动点.

M 1: 设 $X=[0,+\infty)$ , 定义距离d(x,y)=|x-y|, 则(X,d)是完备的距离空间. 定义 $F:X\to X$ 为

$$Fx = x + \frac{1}{1+x}, \quad \forall x \in X.$$

容易验证 F 是压缩映射, 且

$$d(Fx, Fy) < d(x, y), \quad \forall x, y \in X.$$

一般的压缩映射, 即使不等式严格成立, 也不一定有不动点.

例 1: 设X = [0, +∞), 定义距离d(x, y) = |x − y|, 则(X, d)是完备的距离空间. 定义F : X → X为

$$Fx = x + \frac{1}{1+x}, \quad \forall x \in X.$$

容易验证 F 是压缩映射, 且

$$d(Fx, Fy) < d(x, y), \quad \forall x, y \in X.$$

但 F 在 X 中没有不动点.

**推论 1**: 在定理 1 的条件下, 对于任意的  $x_0 \in X$ , 迭代序列 都收敛到F的唯一不动点x, 且有<mark>先验误差估计</mark>

$$d(x_m, x) \le \mu^m (1 - \mu)^{-1} d(x_1, x_0). \tag{3}$$

#### 和后验误差估计

$$d(x_m, x) \le \mu (1 - \mu)^{-1} d(x_{m-1}, x_m). \tag{4}$$

推论 1: 在定理 1 的条件下, 对于任意的 x<sup>0</sup> ∈ X, 迭代序列 都收敛到F的唯一不动点x, 且有先验误差估计

$$d(x_m, x) \le \mu^m (1 - \mu)^{-1} d(x_1, x_0). \tag{3}$$

### 和后验误差估计

$$d(x_m, x) \le \mu (1 - \mu)^{-1} d(x_{m-1}, x_m). \tag{4}$$

先验误差界(3)可在计算之初根据给定的精度要求用来估计 计算的步数. 后验误差界(4)可用于中间步骤或计算结束时的估 计, 它至少有如(3)一样的精度(可能更好).

# 小结

- ▶ 压缩映射
- ▶ Banach不动点定理

# Banach不动点定理的应用

因为 Banach 不动点定理中距离空间的完备性条件和映射的严格压缩性条件比较容易验证, 所以其应用非常广泛, 并由之得到许多重要的结论.

因为 Banach 不动点定理中距离空间的完备性条件和映射 的严格压缩性条件比较容易验证, 所以其应用非常广泛, 并由之 得到许多重要的结论.

Banach 不动点定理的最有意义的应用是关于函数空间的, 利用这个定理可以给出微分方程与积分方程的解的存在性与唯一 性结果.

#### 在微分方程中的应用

例 2: 考虑如下常微分方程的 Cauchy 问题

$$\begin{cases} \frac{du(t)}{dt} = f(t, u(t)), & 0 \le t \le T, \\ u(0) = \alpha, \alpha \in \mathbb{R} \end{cases}$$
 (5)

### 在微分方程中的应用

例 2: 考虑如下常微分方程的 Cauchy 问题

$$\begin{cases} \frac{du(t)}{dt} = f(t, u(t)), & 0 \le t \le T, \\ u(0) = \alpha, \alpha \in \mathbb{R} \end{cases}$$
 (5)

函数  $f:[0,T]\times\mathbb{R}\to\mathbb{R}$  连续, 关于第二个分量满足 Lipschitz 条件, 即存在M, 使得

$$|f(t,u)-f(t,v)| \leq M|u-v|, \quad \forall t \in [0,T], \ \forall u,v \in \mathbb{R}. \ \ (6)$$

### 在微分方程中的应用

例 2: 考虑如下常微分方程的 Cauchy 问题

$$\begin{cases} \frac{du(t)}{dt} = f(t, u(t)), & 0 \le t \le T, \\ u(0) = \alpha, \alpha \in \mathbb{R} \end{cases}$$
 (5)

函数  $f:[0,T]\times\mathbb{R}\to\mathbb{R}$  连续, 关于第二个分量满足 Lipschitz 条件, 即存在M, 使得

$$|f(t,u)-f(t,v)| \leq M|u-v|, \quad \forall t \in [0,T], \ \forall u,v \in \mathbb{R}.$$
 (6)

则方程(5)在整个区间 [0,T] 上存在唯一解 $u \in C[0,T]$ .

#### 证明思路

(1) 转化: Cauchy 问题 等价于如下的积分方程

$$u(t) = \alpha + \int_0^t f(s,u(s)) ds \quad \forall t \in [0,T].$$

### 证明思路

(1) 转化:Cauchy 问题 等价于如下的积分方程

$$u(t) = \alpha + \int_0^t f(s,u(s)) ds \ \ \forall t \in [0,T].$$

(2) 确定完备空间: 赋以范数

$$\|w\|_e = \max_{t \in [0,T]} e^{-Mt} |w(t)|,$$

则(C[0, T ], ∥ · ∥e)为Banach 空间.

### (3) 建立自映射: 定义映射

$$(Fw)(t) = \alpha + \int_0^t f(s,w(s)) ds \quad \forall w \in C[0,T], \quad \forall t \in [0,T].$$

# (3) 建立自映射: 定义映射

$$(Fw)(t) = \alpha + \int_0^t f(s,w(s)) ds \quad \forall w \in C[0,T], \quad \forall t \in [0,T].$$

解的存在唯一性问题转化为找F 的不动点问题.

$$||Fu - Fw||_e \le (1 - e^{-MT})||u - w||_e.$$

$$||Fu - Fw||_e \le (1 - e^{-MT})||u - w||_e.$$

由于  $0 < 1 - e^{-MT} < 1$ , 所以 F 是严格压缩映射.

$$||Fu - Fw||_e \le (1 - e^{-MT})||u - w||_e.$$

由于  $0 < 1 - e^{-MT} < 1$ , 所以 F 是严格压缩映射.

$$\begin{array}{ll} \|Fu-Fw\|_e &=& \max_{0 \leq t \leq T} e^{-Mt} \left| \int_0^t (f(s,u(s)) - f(s,w(s))) ds \right| \ &\leq & \max_{0 \leq t \leq T} e^{-Mt} \int_0^t e^{Ms} e^{-Ms} M \left| u(s) - w(s) \right| ds \ &\leq & M \|u-w\|_e \max_{0 \leq t \leq T} e^{-Mt} \int_0^t e^{Ms} ds \ &\leq & M \|u-w\|_e \max_{0 \leq t \leq T} e^{-Mt} \frac{e^{Mt} - 1}{M} \ &\leq & (1-e^{-MT}) \|u-w\|_e. \end{array}$$

$$||Fu - Fw||_e \le (1 - e^{-MT})||u - w||_e.$$

由于  $0 < 1 - e^{-MT} < 1$ , 所以 F 是严格压缩映射.

$$\begin{array}{ll} \|Fu-Fw\|_e &=& \displaystyle\max_{0\leq t\leq T} e^{-Mt} \left| \int_0^t (f(s,u(s))-f(s,w(s)))ds \right| \ &\leq & \displaystyle\max_{0\leq t\leq T} e^{-Mt} \int_0^t e^{Ms} e^{-Ms} M \left| u(s)-w(s) \right| ds \ &\leq & M \|u-w\|_e \displaystyle\max_{0\leq t\leq T} e^{-Mt} \int_0^t e^{Ms} ds \ &\leq & M \|u-w\|_e \displaystyle\max_{0\leq t\leq T} e^{-Mt} \frac{e^{Mt}-1}{M} \ &\leq & (1-e^{-MT}) \|u-w\|_e. \end{array}$$

由 Banach 不动点定理可知 Cauchy 问题在整个区间[0,T]上存在唯一解  $u\in C[0,T]$ .

$$||Fu - Fw||_e \le (1 - e^{-MT})||u - w||_e.$$

由于  $0 < 1 - e^{-MT} < 1$ , 所以 F 是严格压缩映射.

$$\begin{array}{ll} \|Fu-Fw\|_e &=& \displaystyle\max_{0\leq t\leq T} e^{-Mt} \left| \int_0^t (f(s,u(s))-f(s,w(s)))ds \right| \ &\leq & \displaystyle\max_{0\leq t\leq T} e^{-Mt} \int_0^t e^{Ms} e^{-Ms} M \left| u(s)-w(s) \right| ds \ &\leq & M \|u-w\|_e \displaystyle\max_{0\leq t\leq T} e^{-Mt} \int_0^t e^{Ms} ds \ &\leq & M \|u-w\|_e \displaystyle\max_{0\leq t\leq T} e^{-Mt} \frac{e^{Mt}-1}{M} \ &\leq & (1-e^{-MT}) \|u-w\|_e. \end{array}$$

由 Banach 不动点定理可知 Cauchy 问题在整个区间[0,T]上存在唯一解  $u\in C[0,T]$ .

# (5) 求解迭代格式

$$u_{n+1}(t)=\alpha+\int_0^tf(s,u_n(s))ds, n=0,1,\cdots,$$
其中 $u_0(t)=\alpha.$ 

#### 在积分方程中的应用

例3. 考虑形如

<span id="page-40-0"></span>
$$x(t) - \lambda \int_{a}^{b} K(t, \tau) x(\tau) d\tau = v(t)$$
 (7)

的第二类Fredholm积分方程,其中 $x(t)\in C[a,b]$ 为未知函数, $\lambda$ 为一个参数,核函数K是定义在 $[a,b]\times [a,b]$ 上的已知连续函数, $v(t)\in C[a,b]$ 为已知函数.

### 在积分方程中的应用

例3. 考虑形如

$$x(t) - \lambda \int_{a}^{b} K(t, \tau) x(\tau) d\tau = v(t)$$
 (7)

的第二类Fredholm积分方程, 其中x(t) ∈ C[a, b]为未知函数, λ为一个参数, 核函数K是定义在[a, b] × [a, b]上的已知连续函 数, v(t) ∈ C[a, b]为已知函数.

(1) 确定完备空间: C[a, b]上定义距离使之完备

$$d(x,y) = \max_{a \le t \le b} |x(t) - y(t)|.$$

# (2) 建立自映射:

<span id="page-42-0"></span>
$$(Fx)(t) = v(t) + \lambda \int_{a}^{b} K(t,\tau)x(\tau)d\tau. \tag{8}$$

由v, K都是连续函数, 故[\(8\)](#page-42-0)式定义了空间C[a, b]上的自映射. 解的存在唯一性问题转化为找F 的不动点问题.

# (3) 验证F为压缩映射(限制参数 $\lambda$ )

$$\begin{array}{lll} d(Fx,Fy) &=& \displaystyle\max_{a\leq t\leq b} |(Fx)(t)-(Fy)(t)| \ &\leq& |\lambda| M d(x,y)(b-a). \end{array}$$

记
$$\mu = |\lambda| M(b-a)$$
,可以看出若 $|\lambda| < \frac{1}{M(b-a)}$ ,则有

$$\mu < 1, \ d(Fx, Fy) \le \mu d(x, y),$$

F是严格压缩映射. 方程(7)存在唯一的连续函数解x(t).

# (3) 验证F为压缩映射(限制参数λ)

$$\begin{array}{lll} d(Fx,Fy) &=& \displaystyle\max_{a\leq t\leq b} |(Fx)(t)-(Fy)(t)| \ &\leq& |\lambda| M d(x,y)(b-a). \end{array}$$

记
$$\mu = |\lambda| M(b-a)$$
,可以看出若 $|\lambda| < \frac{1}{M(b-a)}$ ,则有

$$\mu < 1, \ d(Fx, Fy) \le \mu d(x, y),$$

F是严格压缩映射. 方程[\(7\)](#page-40-0)存在唯一的连续函数解x(t).

# (4) 求解迭代格式

$$x_{n+1}(t) = v(t) + \lambda \int_a^b K(t, \tau) x_n(\tau) d\tau.$$

#### 在线性方程组方面的应用

Banach 不动点定理在用迭代法求解线性方程组方面有着重要应用,并为收敛性和误差界提供了充分条件.

### 在线性方程组方面的应用

Banach 不动点定理在用迭代法求解线性方程组方面有着重 要应用, 并为收敛性和误差界提供了充分条件.

求解线性方程组有各种直接法(如高斯消元法), 然而迭代法 或间接法对特殊的方程组可能更为有效. 例如, 振动问题、网络 问题和偏微分方程的差分逼近所对应的稀疏方程组(方程个数很 多, 但只有很少的非零系数).

**例4.** 设方程组  $x_i - \sum_{j=1}^n a_{ij} x_j = b_i, \ i = 1, 2, \cdots, n$ 满足

$$\sum_{i=1}^{n} \sum_{j=1}^{n} |a_{ij}|^2 < 1,$$

其中 $a_{ij} \in \mathbb{R}, b_i \in \mathbb{R}$ , 则方程组有唯一解.

**例4.** 设方程组  $x_i - \sum_{j=1}^n a_{ij} x_j = b_i, \ i = 1, 2, \cdots, n$ 满足

$$\sum_{i=1}^{n} \sum_{j=1}^{n} |a_{ij}|^2 < 1,$$

其中 $a_{ij} \in \mathbb{R}, b_i \in \mathbb{R}$ , 则方程组有唯一解.

(1) 确定完备空间:  $(\mathbb{R}^n, ||.||_2)$ 

**例4.** 设方程组  $x_i - \sum\limits_{j=1}^n a_{ij} x_j = b_i, \ i=1,2,\cdots,n$ 满足

$$\sum_{i=1}^{n} \sum_{j=1}^{n} |a_{ij}|^2 < 1,$$

其中 $a_{ij} \in \mathbb{R}, b_i \in \mathbb{R}$ , 则方程组有唯一解.

- (1) 确定完备空间:  $(\mathbb{R}^n, ||.||_2)$
- (2) 建立自映射:

$$(Fx)_i=\sum_{i=1}^n a_{ij}x_j+b_i, \ \ i=1,2,\cdots,n.$$

则方程组有唯一解转化为Fx = x 的不动点问题.

# (3) 验证F为压缩映射

$$\begin{array}{lcl} d(Fx,Fy) & = & \left(\sum_{i=1}^n |(Fx)_i - (Fy)_i|^2\right)^{\frac{1}{2}} \ & = & \left(\sum_{i=1}^n |\sum_{j=1}^n a_{ij}(x_j - y_j)|^2\right)^{\frac{1}{2}} \ & \leq & \left(\sum_{i=1}^n \sum_{j=1}^n a_{ij}^2\right)^{\frac{1}{2}} d(x,y) \end{array}$$

由题设条件知F为严格压缩映射, 故方程组有唯一解.

### (4) 求解迭代格式

$$x^{(m+1)} = Fx^{(m)}, m = 0, 1, 2, \cdots, \forall x^{(0)} \in \mathbb{R}^n.$$

### (4) 求解迭代格式

$$x^{(m+1)} = Fx^{(m)}, m = 0, 1, 2, \cdots, \forall x^{(0)} \in \mathbb{R}^n.$$

对一般矩阵方程AX = b, A是n阶方阵. 当 det A ̸= 0时, 把A写成A = B − G, 其中B为适当的非奇异矩阵, 即把AX = b变形为

$$X = B^{-1}GX + B^{-1}b$$

的形式, 进而利用相应条件判别解的情况.

#### 非线性方程数值解

挪威数学家Abel, 在1924年证明了5次方程没有解的精确通用表达式; 法国数学家伽罗瓦, 证明了任一个 $n(n \ge 5)$ 次多项式没有解的精确通用表达式. 因此对于某些高次方程, 只能利用数值方法求解!

### 非线性方程数值解

挪威数学家Abel, 在1924年证明了5次方程没有解的精确通用 表达式; 法国数学家伽罗瓦, 证明了任一个n(n ≥ 5)次多项式没 有解的精确通用表达式. 因此对于某些高次方程, 只能利用数值 方法求解!

对于任意方程 g(x) = 0, 定义映射 G(x) = x + Φ(g(x)), 其中映射 Φ 满足"Φ(s) = 0 当且仅当 s = 0". 于是求解方程 g(x) = 0 就等价于求解不动点方程 G(x) = x.

**例5.** 求 $f(x) = 4x^2 - sinx - 1 = 0$ 在[0,1]内的解. 解 原方程化为 $x = \frac{1}{2}\sqrt{sinx + 1}$ ,

**例5.** 求 $f(x) = 4x^2 - sinx - 1 = 0$ 在[0,1]内的解. 解 原方程化为 $x = \frac{1}{2}\sqrt{sinx + 1}$ ,

设 $F(x) = \frac{1}{2}\sqrt{sinx+1}$ ,则可证  $F: [0,1] \to [0,1]$ 为自映射.

例5. 求 $f(x) = 4x^2 - sinx - 1 = 0$ 在[0,1]内的解. 解原方程化为 $x = \frac{1}{2}\sqrt{sinx + 1}$ ,设 $F(x) = \frac{1}{2}\sqrt{sinx + 1}$ ,则可证  $F:[0,1] \rightarrow [0,1]$ 为自映射. 对 $\forall x,y \in [0,1]$ ,有  $|F(x) - F(y)| \leq \frac{1}{4}|x-y|$ ,知 F为严格压缩映射.

例5. 求 $f(x) = 4x^2 - sinx - 1 = 0$ 在[0,1]内的解. 解 原方程化为 $x = \frac{1}{2}\sqrt{sinx + 1}$ ,

设 $F(x) = \frac{1}{2}\sqrt{\sin x + 1}$ ,则可证  $F: [0,1] \to [0,1]$ 为自映射. 对 $\forall x,y \in [0,1]$ ,有  $|F(x) - F(y)| \leq \frac{1}{4}|x-y|$ ,知 F为严格压缩映射.

取 $x_0 = 0.5$ , 利用 $x_{n+1} = \frac{1}{2}\sqrt{sinx_n+1}$ , 得到: (保留10位)  $x_0 = 0.5, x_1 = 0.6081581905, \cdots, x_{13} = 0.6303648857, x_{14} = 0.6303648857.$ 

可见  $x_{13}$ 就是不动点, 也是方程的近似解.

例6. (Newton法) 设实值函数f ∈ C<sup>2</sup> [a, b], f(ξ) = 0, ξ ∈ (a, b), 且 f ′ (ξ) ̸= 0. 则存在 ξ 的邻域 U(ξ) ⊂ [a, b], 使得 任意x<sup>0</sup> ∈ U(ξ), 由

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}, \quad n = 0, 1, 2, \cdots.$$
 (9)

定义的迭代序列 {xn}<sup>∞</sup> <sup>n</sup>=0 都是收敛的, 并且 limn→∞ x<sup>n</sup> = ξ. 例6. (Newton法) 设实值函数f ∈ C<sup>2</sup> [a, b], f(ξ) = 0, ξ ∈ (a, b), 且 f ′ (ξ) ̸= 0. 则存在 ξ 的邻域 U(ξ) ⊂ [a, b], 使得 任意x<sup>0</sup> ∈ U(ξ), 由

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}, \quad n = 0, 1, 2, \cdots.$$
 (9)

定义的迭代序列 {xn}<sup>∞</sup> <sup>n</sup>=0 都是收敛的, 并且 limn→∞ x<sup>n</sup> = ξ. 证明: 由  $f'(\xi) \neq 0$ 及 f'(x)的连续性, 可知存在  $\delta_1 > 0$ , 使得  $[\xi - \delta_1, \xi + \delta_1] \subset [a, b]$ , 且

$$f'(x) \neq 0, \quad \forall x \in [\xi - \delta_1, \xi + \delta_1].$$

证明: 由 f ′ (ξ) ̸= 0及 f ′ (x)的连续性, 可知存在 δ<sup>1</sup> > 0, 使得 [ξ − δ1, ξ + δ1] ⊂ [a, b], 且

$$f'(x) \neq 0, \quad \forall x \in [\xi - \delta_1, \xi + \delta_1].$$

定义映射 F:

$$F(x) = x - \frac{f(x)}{f'(x)}, \quad x \in [\xi - \delta_1, \xi + \delta_1].$$

在区间 [ξ − δ1, ξ + δ1] 上方程 f(x) = 0 等价于不动点方程 F(x) = x. 显然有 F(ξ) = ξ, 即 ξ 是 F 的不动点.

此外

$$F'(x)=\frac{f(x)f''(x)}{(f'(x))^2}, \quad \forall x\in (\xi-\delta_1,\xi+\delta_1).$$

由此可知  $\lim_{x\to\xi}F'(x)=F'(\xi)=0$ . 而F'(x)连续,从而存在  $0<\delta<\delta_1$ ,使得

<span id="page-63-0"></span>
$$|F'(x)| \le \frac{1}{2}, \quad \forall x \in [\xi - \delta, \xi + \delta].$$
 (10)

另一方面, 对于任意的  $x \in [\xi - \delta, \xi + \delta]$ , 都有

$$F(x) - \xi = F(x) - F(\xi) = F'(\eta)(x - \xi),$$

其中 $\eta$ 位于x与 $\xi$ 之间.

另一方面, 对于任意的  $x \in [\xi - \delta, \xi + \delta]$ , 都有

$$F(x) - \xi = F(x) - F(\xi) = F'(\eta)(x - \xi),$$

其中 $\eta$ 位于x与 $\xi$ 之间.

利用 (10) 可得  $|F(x) - \xi| < \delta$ , 即  $F(x) \in [\xi - \delta, \xi + \delta]$  对于任意的  $x \in [\xi - \delta, \xi + \delta]$  都成立, 表明  $F \in [\xi - \delta, \xi + \delta]$  上的自映射.

另一方面, 对于任意的 x ∈ [ξ − δ, ξ + δ], 都有

$$F(x) - \xi = F(x) - F(\xi) = F'(\eta)(x - \xi),$$

其中 η 位于 x 与 ξ 之间.

利用 [\(10\)](#page-63-0) 可得 |F(x) − ξ| < δ, 即 F(x) ∈ [ξ − δ, ξ + δ] 对于 任意的 x ∈ [ξ − δ, ξ + δ] 都成立, 表明 F 是 [ξ − δ, ξ + δ] 上 的自映射.

结合 [\(10\)](#page-63-0) 还可知 F : [ξ − δ, ξ + δ] → [ξ − δ, ξ + δ] 是严格压 缩映射.

[ξ − δ, ξ + δ] 在通常的欧氏度量下是完备的距离空间. 记 U(ξ) = (ξ − δ, ξ + δ), 由 Banach 不动点定理可知, 对任意的 x<sup>0</sup> ∈ U(ξ), 迭代序列

$$x_n = F(x_{n-1}) = x_{n-1} - \frac{f(x_{n-1})}{f'(x_{n-1})}, \quad n \in \mathbb{N}$$

必收敛到 F 的不动点 ξ.

定理2. (Brouwer 不动点定理) 记 B¯ = {x ∈ R <sup>n</sup>|∥x∥ ≤ 1} 为 R <sup>n</sup> 中的闭单位球. 若映射 f : B¯ → B¯ 是连续的, 那么必定存 在点 x ∈ B¯, 使得 f(x) = x.

定理2. (Brouwer 不动点定理) 记 B¯ = {x ∈ R <sup>n</sup>|∥x∥ ≤ 1} 为 R <sup>n</sup> 中的闭单位球. 若映射 f : B¯ → B¯ 是连续的, 那么必定存 在点 x ∈ B¯, 使得 f(x) = x.

**推论1**如果  $K \in \mathbb{R}^n$  中的有界闭凸集, 映射  $f: K \to K$  连续, 则 f 在 K 上存在不动点.

**推论1**如果  $K \in \mathbb{R}^n$  中的有界闭凸集, 映射  $f: K \to K$  连续, 则 f 在 K 上存在不动点.

证明 由 K 同胚于  $\mathbb{R}^n$  中的闭单位球  $\bar{B}$ , 即知存在 映射  $\varphi: K \to \bar{B}$  连续可逆且  $\varphi^{-1}$  也连续.

推论1如果 K 是 R <sup>n</sup> 中的有界闭凸集, 映射 f : K → K 连续, 则 f 在 K 上存在不动点.

证明 由 K 同胚于 R <sup>n</sup> 中的闭单位球 B¯, 即知存在 映射 φ : K → B¯ 连续可逆且 φ−<sup>1</sup> 也连续.

现在考察映射

$$F \triangleq \varphi \circ f \circ \varphi^{-1}: \bar{B} \to \bar{B}.$$

显然 F 是连续的, 由 Brouwer 不动点定理可知存在 x ∈ B¯, 使得 F(x) = x. 于是点 y = φ−<sup>1</sup> (x) ∈ K 就是 f 在 K 上的不动 点.

# 推论2

设 X 是 n 维赋范线性空间,  $K \subset X$  是有界闭凸集, 映射  $f: K \to K$  连续, 则 f 在 K 上存在不动点.

# 推论2

设 X 是 n 维赋范线性空间, K ⊂ X 是有界闭凸集, 映射 f : K → K 连续, 则 f 在 K 上存在不动点.

Brouwer 不动点定理只保证不动点的存在性, 并不保证其唯一性.

**例7** 在  $l^2$  的闭单位球  $\bar{B}=\left\{x\in l^2\Big|\sum\limits_{n=1}^{\infty}|x_n|^2\leq 1\right\}$  上定义映射 f 如下,

<span id="page-75-0"></span>
$$f(x_1, x_2, \cdots, x_n, \cdots) = \left( (1 - \|x\|^2)^{\frac{1}{2}}, x_1, x_2, \cdots, x_{n-1}, \cdots \right),$$

显然 f 是连续的, 且  $f(\bar{B}) \subset \bar{B}$  ( $||f(x)|| = 1, \forall x \in \bar{B}$ ). 试证 明 f 在  $\bar{B} \subset l^2$  上没有不动点.

# 证明

假设存在 
$$x\in \bar{B}$$
 满足  $f(x)=x$ ,那么有 $x_n=x_{n+1},\quad n\in \mathbb{N},$ 

# 证明

假设存在 
$$x\in \bar{B}$$
 满足  $f(x)=x$ ,那么有 $x_n=x_{n+1},\quad n\in \mathbb{N},$ 

结合 P∞ n=1 |xn| <sup>2</sup> < ∞ 可知 x<sup>n</sup> = 0, 即 x = 0. 而由 [\(11\)](#page-75-0), 这时必 有 f(x) = (1, 0, · · · , 0, · · · ). 显然矛盾, 从而 f 在 l <sup>2</sup> 的单位球 B¯ 上不存在不动点.

# 证明

假设存在 
$$x\in \bar{B}$$
 满足  $f(x)=x$ ,那么有 $x_n=x_{n+1},\quad n\in \mathbb{N},$ 

结合 P∞ n=1 |xn| <sup>2</sup> < ∞ 可知 x<sup>n</sup> = 0, 即 x = 0. 而由 [\(11\)](#page-75-0), 这时必 有 f(x) = (1, 0, · · · , 0, · · · ). 显然矛盾, 从而 f 在 l <sup>2</sup> 的单位球 B¯ 上不存在不动点.

# 定理3

(Schauder 不动点定理) 设 X 是一个Banach空间, C 是 X 中的闭凸子集, 映射  $f:C\to C$  是连续的, 且 f(C) 是列紧的, 那么 f 在 C 上存在不动点.

# 定理3

(Schauder 不动点定理) 设 X 是一个Banach空间, C 是 X 中 的闭凸子集, 映射 f : C → C 是连续的, 且 f(C) 是列紧的, 那 么 f 在 C 上存在不动点.

推论3 设 X 是一个Banach空间, C 是 X 中的有界闭凸子集, 映 射 f : C → C 是全连续的, 那么 f 在 C 上存在不动点.

# 定理3

(Schauder 不动点定理) 设 X 是一个Banach空间, C 是 X 中 的闭凸子集, 映射 f : C → C 是连续的, 且 f(C) 是列紧的, 那 么 f 在 C 上存在不动点.

推论3 设 X 是一个Banach空间, C 是 X 中的有界闭凸子集, 映 射 f : C → C 是全连续的, 那么 f 在 C 上存在不动点.