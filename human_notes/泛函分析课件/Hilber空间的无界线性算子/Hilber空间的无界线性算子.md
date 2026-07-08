# Hilber空间的无界线性算子

# 希尔伯特伴随算子的定义及性质

̸ 设X为复Hilbert空间, 若D(T) = X, 我们把 T称为X上的算子 ; 若D(T) = X, 我们把 T称为 X中的算子.

## 定理1

设X为复Hilbert空间, T为X上的线性算子, 若对于所有 的x, y ∈ X均满足(T x, y) = (x, T y), 则T ∈ L(X).

## 证明

## 若结论不真,则必存在 $\{y_n\}_{n=1}^{\infty} \subset X$ 满足

$$||y_n|| = 1, \; \coprod ||Ty_n|| \to \infty.$$

#### 定义线性泛函

$$f_n(x) = (Tx, y_n) = (x, Ty_n), x \in X, n = 1, 2, \cdots$$

对每个固定的n,利用Schwarz不等式,有

$$|f_n(x)| = |(x, Ty_n)| \le ||Ty_n|| ||x||,$$

可见 $f_n$ 是有界线性泛函.

## 证明

#### 若结论不真, 则必存在{yn}<sup>∞</sup> <sup>n</sup>=1 ⊂ X满足

$$||y_n|| = 1, \; \mathbf{H} \; ||Ty_n|| \to \infty.$$

#### 定义线性泛函

$$f_n(x) = (Tx, y_n) = (x, Ty_n), \ x \in X, \ n = 1, 2, \cdots$$

对每个固定的n, 利用Schwarz不等式, 有

$$|f_n(x)| = |(x, Ty_n)| \le ||Ty_n|| ||x||,$$

可见fn是有界线性泛函.

对每个固定的x, 同样利用Schwarz不等式,

有|fn(x)| = |(T x, yn)| ≤ ∥T x∥, 可见{fn(x)}<sup>∞</sup> n=1是有界的. 利用一致有界定理可知{∥fn∥}<sup>∞</sup> <sup>n</sup>=1有界, 不妨设对所有的n, ∥fn∥ ≤ k. 则对每个x ∈ X, 有

$$|f_n(x)| \le ||f_n|| ||x|| \le k||x||,$$

取x = T yn, 得到

$$||Ty_n||^2 = (Ty_n, Ty_n) = |f_n(Ty_n)| \le k||Ty_n||,$$

因此, ∥T yn∥ ≤ k, 与假设∥T yn∥ → ∞ 矛盾, 从而证明 了T ∈ L(X).

设X为复Hilbert空间, T是X中的稠定线性算子, 记

D(T ∗ ) = {y|存在y <sup>∗</sup> ∈ X, 对所有的x ∈ D(T)满足(T x, y) = (x, y<sup>∗</sup> )},

对于每个y ∈ D(T ∗ ), 通过y <sup>∗</sup> = T <sup>∗</sup>y定义的算子 T <sup>∗</sup>称为 T的Hilbert伴随算子.

用记号  $T\subseteq S$  表示S是T的延拓, 即  $\mathbb{D}(T)\subseteq \mathbb{D}(S)$  且  $T=S|_{\mathbb{D}(T)}$ . 若 $\mathbb{D}(T)$ 是 $\mathbb{D}(S)$  的真子集,则称S是T的真延拓.

设X为复Hilbert空间, $S: \mathbb{D}(S) \to X$ 和  $T: \mathbb{D}(T) \to X$ 都是X中的稠定线性算子,则:

- (1) 若  $T \subseteq S$ , 则  $S^* \subseteq T^*$ ;
- <span id="page-8-1"></span><span id="page-8-0"></span>(2)  $\overline{A}\overline{D(T^*)} = X$ , M  $T \subseteq T^{**}$ , A T A T A A A A A A A A A A

设X为复Hilbert空间, $S: \mathbb{D}(S) \to X$ 和  $T: \mathbb{D}(T) \to X$ 都是X中的稠定线性算子,则:

- (1) 若  $T \subseteq S$ , 则  $S^* \subseteq T^*$ ;
- (2) 若 $\overline{\mathbb{D}(T^*)} = X$ , 则  $T \subseteq T^{**}$ , 其中  $T^{**} = (T^*)^*$ .

**证明** (1) 根据  $S^*$ 的定义, 对于所有的 $x \in \mathbb{D}(S)$ ,  $y \in \mathbb{D}(S^*)$ , 有 $(Sx,y) = (x,S^*y)$ .

由于  $T \subseteq S$ , 知对于所有的 $x \in \mathbb{D}(T)$ ,  $y \in \mathbb{D}(S^*)$ , 有

$$(Tx, y) = (x, S^*y).$$
 (2)

设X为复Hilbert空间, S : D(S) → X和 T : D(T) → X都是X中 的稠定线性算子, 则:

- (1) 若 T ⊆ S, 则 S <sup>∗</sup> ⊆ T ∗ ;
- (2) 若 D(T∗) = X, 则 T ⊆ T ∗∗ , 其中 T ∗∗ = (T ∗ ) ∗ .

证明 (1) 根据 S <sup>∗</sup>的定义, 对于所有的x ∈ D(S), y ∈ D(S ∗ ), 有(Sx, y) = (x, S∗y).

由于 T ⊆ S, 知对于所有的x ∈ D(T), y ∈ D(S ∗ ), 有

$$(Tx, y) = (x, S^*y).$$
 (2)

再根据 T <sup>∗</sup>的定义, 对于所有的x ∈ D(T), y ∈ D(T ∗ ), 有

$$(Tx, y) = (x, T^*y).$$
 (3)

由[\(2\)](#page-8-0)和[\(3\)](#page-8-1)可以推出D(S ∗ ) ⊆ D(T ∗ ) 且对所有y ∈ D(S ∗ ), 有T <sup>∗</sup>y = S <sup>∗</sup>y. 故知 S <sup>∗</sup> ⊆ T ∗ .

<span id="page-11-1"></span><span id="page-11-0"></span>(2) 对  $(Tx,y)=(x,T^*y)$ 取复共轭, 可知对所有的  $y\in\mathbb{D}(T^*)$ 和  $x\in\mathbb{D}(T)$ , 有  $(T^*y,x)=(y,Tx). \tag{4}$ 

(2) 对 (T x, y) = (x, T∗y)取复共轭, 可知对所有的 y ∈ D(T ∗ )和 x ∈ D(T), 有

$$(T^*y, x) = (y, Tx).$$
 (4)

由 D(T∗) = X 知T ∗∗存在, 并且根据定义, 对于所有的 y ∈ D(T ∗ )和x ∈ D(T ∗∗)有

$$(T^*y, x) = (y, T^{**}x).$$
 (5)

由[\(4\)](#page-11-0)和[\(5\)](#page-11-1)可知D(T) ⊆ D(T ∗∗) 且对所有x ∈ D(T), 有T ∗∗x = T x. 故知 T ⊆ T ∗∗ .

<span id="page-13-1"></span><span id="page-13-0"></span>设X为复Hilbert空间,T是X中的稠定线性算子,若 T是单射且有 $\overline{\mathbb{R}(T)}=X$ ,则 $T^*$ 是单射,并且 $(T^*)^{-1}=(T^{-1})^*$ .

设X为复Hilbert空间,T是X中的稠定线性算子,若 T是单射且有 $\overline{\mathbb{R}(T)}=X$ ,则 $T^*$ 是单射,并且 $(T^*)^{-1}=(T^{-1})^*$ .

证明 由 T是X中的稠定线性算子知 $T^*$ 存在. 由T是单射知 $T^{-1}$ 存在. 由 $\overline{\mathbb{D}(T^{-1})}=\overline{\mathbb{R}(T)}=X$ 知 $(T^{-1})^*$ 存在.

有 设X为复Hilbert空间, T是X中的稠定线性算子, 若 T是单射且 R(T) = X, 则T <sup>∗</sup>是单射, 并且(T ∗ ) <sup>−</sup><sup>1</sup> = (T −1 ) ∗ .

证明 由 T是X中的稠定线性算子知T <sup>∗</sup>存在. 由T是单射知T <sup>−</sup>1存 在. 由D(T <sup>−</sup>1) = R(T) = X知(T −1 ) <sup>∗</sup>存在. 设y ∈ D(T ∗ ), 则对于所有的x ∈ D(T −1 ), 有T <sup>−</sup>1x ∈ D(T), 并且

$$(T^{-1}x, T^*y) = (TT^{-1}x, y) = (x, y).$$
(6)

而根据(T −1 ) <sup>∗</sup>的定义, 对于所有的x ∈ D(T −1 )有

$$(T^{-1}x, T^*y) = (x, (T^{-1})^*T^*y),$$

表明T <sup>∗</sup>y ∈ D((T −1 ) ∗ ). 与式[\(6\)](#page-13-0)比较, 得到

$$(T^{-1})^*T^*y = y, \ y \in \mathbb{D}(T^*).$$
 (7)

<span id="page-16-1"></span><span id="page-16-0"></span>可见当 $T^*y=0$ 时, y=0, 因此 $(T^*)^{-1}:\mathbb{R}(T^*)\to\mathbb{D}(T^*)$ 存在. 此外, 由于 $(T^*)^{-1}T^*$ 是 $\mathbb{D}(T^*)$ 上的单位算子, 与(7)比较可知,  $(T^*)^{-1}\subseteq (T^{-1})^*$ .

可见当 $T^*y=0$ 时, y=0, 因此 $(T^*)^{-1}:\mathbb{R}(T^*)\to\mathbb{D}(T^*)$ 存在. 此外, 由于 $(T^*)^{-1}T^*$ 是 $\mathbb{D}(T^*)$ 上的单位算子, 与(7)比较可知,  $(T^*)^{-1}\subseteq (T^{-1})^*$ .

设 $x \in \mathbb{D}(T)$ , 则 $Tx \in \mathbb{R}(T) = \mathbb{D}(T^{-1})$ . 若  $y \in \mathbb{D}((T^{-1})^*)$ , 则有

$$(Tx, (T^{-1})^*y) = (T^{-1}Tx, y) = (x, y).$$
(8)

而根据 $T^*$ 的定义, 对于所有的 $x \in \mathbb{D}(T)$ 有

$$(Tx, (T^{-1})^*y) = (x, T^*(T^{-1})^*y).$$
(9)

表明 $(T^{-1})^*y \in \mathbb{D}(T^*)$ ,与式(9)比较,得到

$$T^*(T^{-1})^*y = y, \ y \in \mathbb{D}((T^{-1})^*). \tag{10}$$

可见当T <sup>∗</sup>y = 0时, y = 0, 因此(T ∗ ) −1 : R(T ∗ ) → D(T ∗ )存在. 此 外, 由于(T ∗ ) <sup>−</sup>1T <sup>∗</sup>是D(T ∗ )上的单位算子, 与[\(7\)](#page-13-1)比较可知, (T ∗ ) <sup>−</sup><sup>1</sup> ⊆ (T −1 ) ∗ .

设x ∈ D(T), 则T x ∈ R(T) = D(T −1 ). 若 y ∈ D((T −1 ) ∗ ), 则有

$$(Tx, (T^{-1})^*y) = (T^{-1}Tx, y) = (x, y).$$
(8)

而根据T <sup>∗</sup>的定义, 对于所有的x ∈ D(T)有

$$(Tx, (T^{-1})^*y) = (x, T^*(T^{-1})^*y).$$
(9)

表明(T −1 ) <sup>∗</sup>y ∈ D(T ∗ ), 与式[\(9\)](#page-16-0)比较, 得到

$$T^*(T^{-1})^*y = y, \ y \in \mathbb{D}((T^{-1})^*).$$
 (10)

利用 T ∗ (T ∗ ) <sup>−</sup>1是D((T ∗ ) −1 )上的单位算子 及(T ∗ ) −1 : R(T ∗ ) → D(T ∗ )是满射, 与[\(10\)](#page-16-1)比较可知, D((T ∗ ) −1 ) ⊇ D((T −1 ) ∗ ), (T ∗ ) <sup>−</sup><sup>1</sup> ⊇ (T −1 ) ∗ . 故 (T ∗ ) <sup>−</sup><sup>1</sup> = (T −1 ) ∗ .

# 对称和自伴算子

设X为复Hilbert空间, $T: \mathbb{D}(T) \to X$ 是X中的稠定线性算子. 如果对于所有的 $x,y \in \mathbb{D}(T)$ 有

<span id="page-20-0"></span>
$$(Tx, y) = (x, Ty),$$

<span id="page-20-1"></span>则称T是对称线性算子.

设X为复Hilbert空间, $T: \mathbb{D}(T) \to X$ 是X中的稠定线性算子. 如果对于所有的 $x,y \in \mathbb{D}(T)$ 有

$$(Tx, y) = (x, Ty),$$

则称T是对称线性算子.

#### 定理4

设X为复Hilbert空间, $T: \mathbb{D}(T) \to X$ 是X中的稠定线性算子,则T是对称线性算子当且仅当 $T \subseteq T^*$ .

设X为复Hilbert空间, T : D(T) → X是X中的稠定线性算子. 如 果对于所有的x, y ∈ D(T)有

$$(Tx, y) = (x, Ty),$$

则称T是对称线性算子.

## 定理4

设X为复Hilbert空间, T : D(T) → X是X中的稠定线性算子, 则T是对称线性算子当且仅当T ⊆ T ∗ .

证明 对所有的x ∈ D(T), y ∈ D(T ∗ ), 有

$$(Tx, y) = (x, T^*y).$$
 (11)

若T ⊆ T ∗ , 则对于y ∈ D(T), 有T <sup>∗</sup>x = T x, 所以对于所 有的x, y ∈ D(T), [式](#page-20-0)(11)变成

<span id="page-23-0"></span>
$$(Tx,y) = (x,Ty). (12)$$

可见T是对称线性算子.

若T是对称线性算子, 则对于所有的x, y ∈ D(T), 式[\(12\)](#page-23-0)成立, 与 式[\(11\)](#page-20-0)比较可得D(T) ⊆ D(T ∗ )且 T = T ∗ |D(T) ,可见T ⊆ T ∗ .

设X为复Hilbert空间, $T: \mathbb{D}(T) \to X$ 是X中的稠定线性算子. 如 果 $T=T^*$ ,则称T是自伴线性算子.

设X为复Hilbert空间, $T: \mathbb{D}(T) \to X$ 是X中的稠定线性算子. 如果 $T = T^*$ ,则称T是自伴线性算子.

注: 1. 自伴算子都是对称算子, 但对称算子未必是自伴算子.

设X为复Hilbert空间, T : D(T) → X是X中的稠定线性算子. 如 果T = T ∗ , 则称T是自伴线性算子.

注:1. 自伴算子都是对称算子, 但对称算子未必是自伴算子.

2.对于Hilbert空间上的线性算子来说, 对称性和自伴性是等同的 .

## 闭线性算子

设X为复Hilbert空间,  $\mathbb{D}(T)\subseteq X,\ T:\ \mathbb{D}(T)\to X$ 是线性算子. 若T的图像

$$G(T) = \{(x, Tx) \mid x \in \mathbb{D}(T)\}\$$

<span id="page-28-0"></span>是  $X \times X$  中的闭集, 则称 T 为闭线性算子, 其中  $X \times X$  上的范数定义为

$$||(x,y)|| = ||x|| + ||y||$$

设X为复Hilbert空间,  $\mathbb{D}(T)\subseteq X,\ T:\ \mathbb{D}(T)\to X$ 是线性算子. 若T的图像

$$G(T) = \{(x, Tx) \mid x \in \mathbb{D}(T)\}\$$

是  $X \times X$  中的闭集, 则称 T 为闭线性算子, 其中  $X \times X$  上的范数定义为

$$||(x,y)|| = ||x|| + ||y||$$

#### 定理5

设X为复Hilbert空间,  $\mathbb{D}(T)\subseteq X,\ T:\ \mathbb{D}(T)\to X$ 是线性算子, 则

- (1) T 为闭算子当且仅当对任意的  $\{x_n\}_{n=1}^{\infty}\subset \mathbb{D}(T)$ , 若
- $x_n \to x, Tx_n \to y, \text{ } y \text{ } y \text{ } x \in \mathbb{D}(T) \text{ } \exists y = Tx.$
- (2) 若 T 为闭算子且 $\mathbb{D}(T)$ 是闭集, 则 T 为有界算子 (闭图像定理).
- (3) 若 T 为有界算子, 则 T 为闭算子当且仅当 $\mathbb{D}(T)$ 是闭集.

<span id="page-30-0"></span>设X为复Hilbert空间, T是X中的稠定线性算子, 则T\*是闭算子.

设X为复Hilbert空间, T是X中的稠定线性算子, 则 $T^*$ 是闭算子. 证明 设  $\{y_n\}_{n=1}^\infty\subseteq\mathbb{D}(T^*)$ , 满足  $y_n\to y_0, T^*y_n\to z_0$ . 由 $T^*$ 的定义, 对于所有的 $y\in\mathbb{D}(T)$ , 有

$$(Ty, y_n) = (y, T^*y_n).$$

令  $n \to \infty$ 并利用内积的连续性, 可得

$$(Ty, y_0) = (y, z_0).$$

根据  $T^*$ 的定义, 知 $y_0 \in \mathbb{D}(T^*)$ ,  $T^*y_0 = z_0$ . 利用定理5可得 $T^*$ 是 闭算子.

若线性算子 T 有延拓  $T_1$ , 且 $T_1$ 是闭算子, 则称T是可闭算子, 称 $T_1$ 是T的闭线性延拓.

若线性算子 T 有延拓  $T_1$ , 且 $T_1$ 是闭算子, 则称T是可闭算子, 称 $T_1$ 是T的闭线性延拓.

若 $\bar{T}$  是可闭算子T的闭线性延拓, 并且T的每一个闭线性延拓  $T_1$ 也是 $\bar{T}$  的闭线性延拓, 则称 $\bar{T}$  是T的最小闭线性延拓.

若线性算子 T 有延拓 T1, 且T1是闭算子, 则称T是可闭算子, 称T1是T的闭线性延拓.

若 T¯ 是可闭算子T的闭线性延拓, 并且T的每一个闭线性延拓 T1也是T¯ 的闭线性延拓, 则称T¯ 是T的最小闭线性延拓.

若 T的最小闭线性延拓T¯存在, 则称T¯为T的闭包.

设X为复Hilbert空间,  $T: \mathbb{D}(T) \to X$ 是X中的稠定线性算子. 若T是对称算子, 则

- (1)
- $(2) (\bar{T})^* = T^*.$

# 自伴线性算子的谱

#### 引理1

<span id="page-37-0"></span>设X为复Hilbert空间, $T: \mathbb{D}(T) \to X$ 是X中的稠定自伴线性算子,则 $\lambda$ 属于T的预解集 $\rho(T)$ 的充分必要条件是:存在正数c使得对于每一个 $x \in \mathbb{D}(T)$ 有

$$||T_{\lambda}x|| \ge c||x||,\tag{13}$$

其中 $T_{\lambda} = T - \lambda I$ .

设X为复Hilbert空间, $T: \mathbb{D}(T) \to X$ 是X中的稠定自伴线性算子,则T的谱集 $\sigma(T)$ 是闭实数集.

设X为复Hilbert空间, T : D(T) → X是X中的稠定自伴线性算 子, 则T的谱集σ(T)是闭实数集.

证明 先证 σ(T)是实数集.

̸ 对每个0 = x ∈ D(T), 我们有

$$(T_{\lambda}x, x) = (Tx, x) - \lambda(x, x).$$

由于(T x, x),(x, x)都是实数, 所 以

$$\overline{(T_{\lambda}x,x)} = (Tx,x) - \overline{\lambda}(x,x).$$

记 λ = α + iβ, α, β ∈ R, 则λ = α − iβ.上面两式相减得到

$$-2i\operatorname{Im}(T_{\lambda}x,x) = \overline{(T_{\lambda}x,x)} - (T_{\lambda}x,x) = (\lambda - \overline{\lambda})(x,x) = 2i\beta ||x||^{2}.$$

#### 利用Schwarz不等式,有

$$\beta ||x||^2 \le |(T_\lambda x, x)| \le ||T_\lambda x|| ||x||.$$

由 $\|x\| \neq 0$ , 可知 $\beta \|x\| \leq \|T_{\lambda}x\|$ . 若 $\beta \neq 0$ , 根据引理1便有 $\lambda \in \rho(T)$ . 因此,  $\sigma(T)$ 是实数集.

## 利用Schwarz不等式, 有

$$\beta ||x||^2 \le |(T_\lambda x, x)| \le ||T_\lambda x|| ||x||.$$

̸ 由∥x∥ ≠ 0, 可知β∥x∥ ≤ ∥Tλx∥. 若β = 0, 根据引理1[便](#page-37-0) 有λ ∈ ρ(T). 因此, σ(T)是实数集. 再证 σ(T)是闭集.

设λ<sup>0</sup> ∈ ρ(T), 由

$$||Tx - \lambda_0 x|| = ||Tx - \lambda x + (\lambda - \lambda_0)x|| \le ||Tx - \lambda x|| + |\lambda - \lambda_0|||x||,$$

#### 得到

$$||Tx - \lambda x|| \ge ||Tx - \lambda_0 x|| - |\lambda - \lambda_0|||x||.$$

由于λ<sup>0</sup> ∈ ρ(T), 根据引理1[可](#page-37-0)知存在c > 0, 使得对于所有 的x ∈ D(T)有

$$||Tx - \lambda_0 x|| \ge c||x||.$$

若取|λ − λ0| ≤ <sup>c</sup> 2 , 则有

$$||Tx - \lambda x|| \ge \frac{c}{2} ||x||.$$

同样根据引理1[可](#page-37-0)知λ ∈ ρ(T), 得到ρ(T)是开集, σ(T) = C − ρ(T)是闭集.

# 乘法算子和微分算子

$$T: \mathbb{D}(T) \to L^2(-\infty, +\infty),$$
  
 $x \mapsto tx$  (14)

其中

$$\mathbb{D}(T) = \{x | \int_{-\infty}^{+\infty} t^2 |x(t)|^2 dt < \infty, x \in L^2(-\infty, +\infty) \}.$$

即算子T的定义域 $\mathbb{D}(T)$  是由满足  $Tx\in L^2(-\infty,+\infty)$ 的所有  $x\in L^2(-\infty,+\infty)$ 构成.

$$T: \mathbb{D}(T) \to L^2(-\infty, +\infty),$$

$$x \mapsto tx \tag{14}$$

其中

$$\mathbb{D}(T) = \{x | \int_{-\infty}^{+\infty} t^2 |x(t)|^2 dt < \infty, x \in L^2(-\infty, +\infty)\}.$$

即算子T的定义域 $\mathbb{D}(T)$  是由满足  $Tx\in L^2(-\infty,+\infty)$ 的所有  $x\in L^2(-\infty,+\infty)$ 构成.

显然,  $\mathbb{D}(T)$ 包含在紧区间外等于0的所有函数 $x \in L^2(-\infty, +\infty)$ , 而这种函数的集合在 $L^2(-\infty, +\infty)$ 中稠密. 容易证明, 由

$$x(t) = \begin{cases} \frac{1}{t}, & \mathbf{\Xi}t \ge 1, \\ 0, & \mathbf{\Xi}t < 1 \end{cases}$$

定义的函数 $x\in L^2(-\infty,+\infty)$ , 但 $x\notin \mathbb{D}(T)$ . 因此,  $\mathbb{D}(T)$ 在 $L^2(-\infty,+\infty)$ 中稠密, 但 $\mathbb{D}(T)\neq L^2(-\infty,+\infty)$ .

#### 引理2

乘法算子T不是有界算子.

## 证明 取函数

$$x_n(t) = \begin{cases} 1, & \mathsf{若} n \le t < n+1, \\ 0, & \mathsf{其它} \end{cases}$$

显然, ∥xn∥ = 1 且

$$||Tx_n||^2 = \int_n^{n+1} t^2 dt > n^2.$$

表明∥T xn∥/∥xn∥ > n, T不是有界算子.

乘法算子T是自伴算子.

乘法算子T是自伴算子.

证明 前面已经指出,  $T \in L^2(-\infty, +\infty)$ 中的稠定算子. 因为

$$(Tx,y) = \int_{-\infty}^{+\infty} tx(t)\overline{y(t)}dt = \int_{-\infty}^{+\infty} x(t)\overline{ty(t)}dt = (x,Ty).$$

所以T是对称算子. 根据定理4有 $T \subseteq T^*$ ,因此我们只需证明 $\mathbb{D}(T^*) \subseteq \mathbb{D}(T)$ 即可.

乘法算子T是自伴算子.

证明 前面已经指出, T是L 2 (−∞, +∞)中的稠定算子. 因为

$$(Tx,y) = \int_{-\infty}^{+\infty} tx(t)\overline{y(t)}dt = \int_{-\infty}^{+\infty} x(t)\overline{ty(t)}dt = (x,Ty).$$

明D(T ∗ ) ⊆ D(T)即可 所以T是对称算子. 根据定理[4](#page-20-1)有T ⊆ T ∗ , 因此我们只需证 . 设 y ∈ D(T ∗ ), 则对于所有的x ∈ D(T), 有

$$(Tx,y) = (x,y^*),$$

即

$$\int_{-\infty}^{+\infty} tx(t)\overline{y(t)}dt = \int_{-\infty}^{+\infty} x(t)\overline{y^*(t)}dt.$$

$$\int_{-\infty}^{+\infty} x(t) \left[t\overline{y(t)} - \overline{y^*(t)}\right] dt = 0.$$
 (15)

$$\int_{-\infty}^{+\infty} x(t) [t\overline{y(t)} - \overline{y^*(t)}] dt = 0.$$
 (15)

定义函数

$$x(t) = \left\{ \begin{array}{cc} ty(t) - y^*(t), & {\bf \Xi}a < t < b, \\ 0, & {\bf 其它} \end{array} \right.$$

其中(a,b)是任取的有界区间. 则式(15)变形为

$$\int_{a}^{b} ||ty(t) - y^{*}(t)||^{2} dt = 0.$$

$$\int_{-\infty}^{+\infty} x(t) \left[t\overline{y(t)} - \overline{y^*(t)}\right] dt = 0.$$
 (15)

定义函数

$$x(t) = \begin{cases} ty(t) - y^*(t), & \mathbf{\Xi}a < t < b, \\ 0, & \mathbf{J}\mathbf{\hat{c}} \end{cases}$$

其中(a, b)是任取的有界区间. 则式(15)变形为

$$\int_{a}^{b} ||ty(t) - y^{*}(t)||^{2} dt = 0.$$

这就推出了ty(t) − y ∗ (t) = 0,在 (a, b)上几乎处处成立. 即在区间 (a, b)上几乎处处有ty(t) = y ∗ (t). 由于区间 (a, b)的任意性, 便 知ty(t) = y ∗ (t) ∈ L 2 (−∞, +∞), 所以y ∈ D(T), T <sup>∗</sup>y = y <sup>∗</sup> = ty = T y. 由T <sup>∗</sup> = T及定理6[可](#page-30-0)知, 乘法算子T是闭算子

设 $\sigma(T)$ 是乘法算子T的谱,则

- (1) T没有特征值, 即T的点谱为空集.
- (2)  $\sigma(T) = \mathbb{R}$ .

设σ(T)是乘法算子T的谱, 则

- (1) T没有特征值, 即T的点谱为空集.
- (2) σ(T) = R.

证明 (1) 对任意的λ, 设x ∈ D(T)满足T x = λx, 则 有(T − λI)x = 0. 故根据T的定义有

$$0 = \|(T - \lambda I)x\|^2 = \int_{-\infty}^{+\infty} |t - \lambda|^2 \|x(t)\|^2 dt.$$

由于对所有的t ̸= λ有|t − λ| <sup>2</sup> > 0, 所有对几乎所有 的t ∈ R有x(t) = 0, 即x = 0. 这表明x不是特征向量, λ不是T的特 征值. 由于λ的任意性, 所以T没有特征值.

# (2) 由T是自伴算子知 $\sigma(T) \subseteq \mathbb{R}$ . 设 $\lambda \in \mathbb{R}$ , 定义

$$v_n(t) = \left\{ \begin{array}{ll} 1, & \lambda - \frac{1}{n} \leq t \leq \lambda + \frac{1}{n}, \\ 0, & 其它 \end{array} \right.$$

记 $x_n = ||v_n||^{-1}v_n$ , 则 $||x_n|| = 1$ . 由T及 $T_{\lambda}$ 的定义可得

$$||T_{\lambda}x_n||^2 = \int_{-\infty}^{+\infty} (t-\lambda)^2 ||x_n(t)||^2 dt \le \frac{1}{n^2} \int_{-\infty}^{+\infty} ||x_n(t)||^2 dt = \frac{1}{n^2}.$$

取平方根,得到

<span id="page-55-0"></span>
$$||T_{\lambda}x_n|| \le \frac{1}{n}.\tag{16}$$

由于T没有特征值, T −1 <sup>λ</sup> 存在, 并且因为x<sup>n</sup> ̸= 0, 所以有Tλx<sup>n</sup> ̸= 0,

$$y_n = ||T_{\lambda}x_n||^{-1}T_{\lambda}x_n \in \mathbb{R}(T_{\lambda}), ||y_n|| = 1.$$

用T −1 <sup>λ</sup> 作用到yn上, 并利用式[\(16\)](#page-55-0)便得到

$$||T_{\lambda}^{-1}y_n|| = ||T_{\lambda}x_n||^{-1}||x_n|| \ge n.$$

这就证明了T −1 <sup>λ</sup> 是无界的, 因此<sup>λ</sup> <sup>∈</sup> <sup>σ</sup>(T), <sup>σ</sup>(T) = <sup>R</sup>. 微分算子:

$$D: \mathbb{D}(D) \to L^2(-\infty, +\infty),$$

$$x \mapsto ix',$$
(17)

其中x'=dx/dt, 加i是为了使D成为自伴算子. 根据定义, D的定义域 $\mathbb{D}(D)$ 由 $L^2(-\infty,+\infty)$  中的在 $\mathbb{R}$ 中每个紧区间上绝对连续且其导数 $x'\in L^2(-\infty,+\infty)$ 的所有 x构成.

微分算子:

$$D: \mathbb{D}(D) \to L^{2}(-\infty, +\infty),$$

$$x \mapsto ix',$$
(17)

其中x ′ = dx/dt, 加i是为了使D成为自伴算子. 根据定义, D的定 义域D(D)由L 2 (−∞, +∞) 中的在R中每个紧区间上绝对连续且 其导数x ′ ∈ L 2 (−∞, +∞)的所有 x构成.

## 注

于[a, b]上的任意有限个互不相交的开区间(a1, b1),(a2, b2), · · · 设f(x)是[a, b]上的函数, 若对任意的 ε > 0, 存在 δ > 0, 使得对 , (an, bn), 当

$$\sum_{j=1}^{n} (b_j - a_j) < \delta$$

<sup>时</sup>, 就有 <sup>n</sup>

$$\sum_{j=1}^{n} |f(b_j) - f(a_j)| < \varepsilon,$$

则称f(x)是[a, b]上的绝对连续函数 .  $\mathbb{D}(D)$ 包含规范正交序列 $\{e_n\}_{n=1}^{\infty}$ ,其中

$$e_n(t) = \frac{1}{(2^n n! \sqrt{\pi})^{\frac{1}{2}}} e^{-\frac{t^2}{2}} H_n(t),$$

这里 $H_n(t)$ 为n阶埃尔米特多项式, 具体形式为

$$H_0(t) = 1, H_n(t) = (-1)^n e^{t^2} \frac{d^n}{dt^n} (e^{-t^2}), n = 1, 2, \cdots$$

由 $\{e_n\}_{n=1}^{\infty}$ 在 $L^2(-\infty,+\infty)$ 中的完全性结果,可知 $\mathbb{D}(D)$ 在 $L^2(-\infty,+\infty)$ 中稠密.

## 引理3

微分算子D是无界算子.

#### 引理3

微分算子D是无界算子.

证明 把 $L^2[0,1]$ 看成 $L^2(-\infty,+\infty)$ 的子空间,记 $Y=\mathbb{D}(D)\cap L^2[0,1]$ ,则D是算子 $D_0=D|_Y$ 的一个延拓.若 $D_0$ 无界,则D无界.设函数

$$x_n(t) = \begin{cases} 1 - nt, & 0 \le t \le \frac{1}{n}, \\ 0, & \frac{1}{n} < t < 1. \end{cases}$$

### 其导函数为

$$x_n'(t) = \left\{ \begin{array}{ll} -n, & 0 \leq t < \frac{1}{n}, \\ 0, & \frac{1}{n} < t < 1. \end{array} \right.$$

#### 计算可得

$$||x_n||^2 = \int_0^1 |x_n(t)|^2 dt = \frac{1}{3n},$$
$$||D_0 x_n||^2 = \int_0^1 |x_n'(t)|^2 dt = n,$$

可见 $D_0$ 无界, 进而知D无界.

- (1) 微分算子是自伴算子,
- (2) D没有特征值, σ(D) = R.

# 量子力学中的位置算子与动量算 子

我们考虑任意固定时刻的系统, 也就是把时间视为保持不变的参数. 在经典力学中, 我们用一对数(粒子的特定位置和速度)来描述系统的瞬时状态. 在量子力学中, 我们用函数 $\psi$ 来描述系统的状态, 其中 $\psi$ 是定义在 $\mathbb{R}$ 上关于自变量q的复值函数.  $\psi$ , q都是物理学中的标准符号, 分别代替了常用的函数记号x和时间记号t.

在经典力学中, 我们用一对数(粒子的特定位置和速度)来描述系统的瞬时状态. 在量子力学中, 我们用函数 $\psi$ 来描述系统的状态, 其中 $\psi$ 是定义在 $\mathbb{R}$ 上关于自变量q的复值函数.  $\psi$ , q都是物理学中的标准符号, 分别代替了常用的函数记号x和时间记号t.

假设 $\psi$ 是希尔伯特空间 $L^2(-\infty,+\infty)$ 中的元素,即 $\psi \in L^2(-\infty,+\infty)$ . 则粒子在给定子集 $J \subseteq \mathbb{R}$ 中出现的概率为

$$\int_{J} |\psi(q)|^2 dq. \tag{18}$$

在经典力学中, 我们用一对数(粒子的特定位置和速度)来描述系 统的瞬时状态. 在量子力学中, 我们用函数ψ来描述系统的状态, 其中ψ是定义在R上关于自变量q的复值函数. ψ, q都是物理学中 的标准符号, 分别代替了常用的函数记号x和时间记号t.

假设ψ是希尔伯特空间L 2 (−∞, +∞)中的元素, 即ψ ∈L 2 (−∞, + ∞). 则粒子在给定子集J ⊆ R中出现的概率为

$$\int_{J} |\psi(q)|^2 dq. \tag{18}$$

对应于整个一维空间R的概率为1, 也就是说, 粒子必处于实直线 上的某一点, 这就施加了一个正规化条件

$$\|\psi\|^2 = \int_{-\infty}^{+\infty} |\psi(q)|^2 dq = 1.$$
 (19)

#### 把物理系统在某一时刻的**状态**定义为元素

$$\psi \in L^2(-\infty, +\infty), \ \|\psi\| = 1.$$
 (20)

## 把物理系统在某一时刻的状态定义为元素

$$\psi \in L^2(-\infty, +\infty), \ \|\psi\| = 1.$$
 (20)

从式(18)可以看出, |ψ(q)| <sup>2</sup> 起着R上概率分布密度的作用. 根据定 义, 相应的均值(期望值)是

$$\mu_{\psi} = \int_{-\infty}^{+\infty} q |\psi(q)|^2 dq, \qquad (21)$$

分布的方差是

$$\operatorname{var}_{\psi} = \int_{-\infty}^{+\infty} (q - \mu_{\psi})^2 |\psi(q)|^2 dq, \tag{22}$$

标准差是sd<sup>ψ</sup> = <sup>√</sup>varψ. 直观上, <sup>µ</sup>ψ测量了平均值或中心位置, varψ测量了分布的范围.

#### 把式(21)改写成

$$\mu_{\psi}(Q) = (Q\psi, \psi) = \int_{-\infty}^{+\infty} Q\psi(q) \overline{\psi(q)} dq, \qquad (23)$$

其中算子Q为

$$Q: \mathbb{D}(Q) \to L^2(-\infty, +\infty),$$

$$\psi \mapsto q\psi$$
(24)

其中  $\mathbb{D}(Q)$ 是由 $L^2(-\infty,+\infty)$ 中满足 $Q\psi\in L^2(-\infty,+\infty)$ 的所有 $\psi$ 构成. 由于 $\mu_\psi(Q)$ 表征了粒子的平均位置, 所以我们称算子Q为位置算子. 由上一小节对乘法算子的讨论可知, Q是 $L^2(-\infty,+\infty)$ 中稠定的无界线性算子.

### 利用算子Q, 可把式(22)改写为

$$\operatorname{var}_{\psi}(Q) = ((Q - \mu I)^{2} \psi, \psi) = \int_{-\infty}^{+\infty} (Q - \mu I)^{2} \psi(q) \overline{\psi(q)} dq, \ \mu = \mu_{\psi}(Q).$$
(25)

## 利用算子Q, 可把式(22)改写为

$$\operatorname{var}_{\psi}(Q) = ((Q - \mu I)^{2} \psi, \psi) = \int_{-\infty}^{+\infty} (Q - \mu I)^{2} \psi(q) \overline{\psi(q)} dq, \ \mu = \mu_{\psi}(Q).$$
(25)

我们把物理系统在某一时刻的观察量定义为自伴线性算 子T : D(T) → L 2 (−∞, +∞), 其中D(T)在L 2 (−∞, +∞)中稠

密.

#### 类似于(23)和(25), 我们把均值 $\mu_{\psi}(T)$ 定义为

$$\mu_{\psi}(T) = (T\psi, \psi) = \int_{-\infty}^{+\infty} T\psi(q) \overline{\psi(q)} dq, \tag{26}$$

### 类似于(23)和(25), 我们把均值 $\mu_{\psi}(T)$ 定义为

$$\mu_{\psi}(T) = (T\psi, \psi) = \int_{-\infty}^{+\infty} T\psi(q) \overline{\psi(q)} dq, \tag{26}$$

#### 方差 $var_{\psi}(T)$ 定义为

$$\operatorname{var}_{\psi}(T) = ((T - \mu I)^{2} \psi, \psi) = \int_{-\infty}^{+\infty} (T - \mu I)^{2} \psi(q) \overline{\psi(q)} dq, \ \mu = \mu_{\psi}(T).$$
(27)

## 类似于(23)和(25), 我们把均值µψ(T) 定义为

$$\mu_{\psi}(T) = (T\psi, \psi) = \int_{-\infty}^{+\infty} T\psi(q) \overline{\psi(q)} dq, \qquad (26)$$

## 方差varψ(T)定义为

$$\operatorname{var}_{\psi}(T) = ((T - \mu I)^{2} \psi, \psi) = \int_{-\infty}^{+\infty} (T - \mu I)^{2} \psi(q) \overline{\psi(q)} dq, \ \mu = \mu_{\psi}(T).$$
(27)

## 标准差定义为

$$\operatorname{sd}_{\psi}(T) = \sqrt{\operatorname{var}_{\psi}(T)}.$$

如果系统处于状态ψ, 则µψ(T)刻画了在实验中我们能够期望得到 的观察量T的平均值, 方差varψ(T)刻画了观察量在均值周围的变 化范围.

#### 另一个重要的观察量是动量p,相应的动量算子是

$$D: \mathbb{D}(D) \to L^2(-\infty, +\infty),$$

$$\psi \mapsto \frac{h}{2\pi i} \frac{d\psi}{dq},$$
(28)

其中 $h=6.62607015\times 10^{-34}J\cdot s$ 是普朗克常数. D的定义域 $\mathbb{D}(D)$ 由 $L^2(-\infty,+\infty)$  中的在 $\mathbb{R}$ 中每个紧区间上绝对连续且 $D\psi\in L^2(-\infty,+\infty)$ 的所有函数  $\psi$ 构成.

### 另一个重要的观察量是动量p, 相应的动量算子是

$$D: \mathbb{D}(D) \to L^{2}(-\infty, +\infty), \psi \mapsto \frac{h}{2\pi i} \frac{d\psi}{dq},$$
 (28)

其中h = 6.62607015 × 10−34J · s是普朗克常数. D的定义 域D(D)由L 2 (−∞, +∞) 中的在R中每个紧区间上绝对连续 且Dψ ∈ L 2 (−∞, +∞)的所有函数 ψ构成.

对于观察量D, 当然也有相应的均值µψ(D)及方差 varψ(D)结果.

设S和T是同一复的希尔伯特空间中的两个自伴线性算子,则称算子

$$C = ST - TS$$

为S和T的换位子,它的定义域是 $\mathbb{D}(C) = \mathbb{D}(ST) \cap \mathbb{D}(TS)$ .

设S和T是同一复的希尔伯特空间中的两个自伴线性算子, 则称算 子

$$C = ST - TS$$

为S和T的换位子, 它的定义域是D(C) = D(ST) ∩ D(T S).

在量子力学中, 位置算子和动量算子的换位子具有基本的重要性. 通过直接微分, 有

$$DQ\psi(q) = D(q\psi(q)) = \frac{h}{2\pi i} [\psi(q) + q\psi'(q)] = \frac{h}{2\pi i} \psi(q) + QD\psi(q).$$

由此得到重要的海森堡交换关系

$$DQ - QD = \frac{h}{2\pi i}\tilde{I},$$

其中˜I是定义域D(DQ) ∩ D(QD)上的恒等算子. 这个定义域在空 间L 2 (−∞, +∞)中稠密.

设S和T是定义域和值域都落在 $L^2(-\infty, +\infty)$ 中的自伴线性算子,则算子C=ST-TS对于 $\mathbb{D}(C)$ 中的每一个 $\psi$ 满足

$$|\mu_{\psi}(C)| \le 2\operatorname{sd}_{\psi}(S)\operatorname{sd}_{\psi}(T). \tag{29}$$

设S和T是定义域和值域都落在L 2 (−∞, +∞)中的自伴线性算子, 则算子C = ST − T S对于D(C)中的每一个ψ满 足

$$|\mu_{\psi}(C)| \le 2\operatorname{sd}_{\psi}(S)\operatorname{sd}_{\psi}(T). \tag{29}$$

证明 记µ<sup>1</sup> = µψ(S), µ<sup>2</sup> = µψ(T), A = S − µ1I, B = T − µ2I, 则 直接验算可知

$$C = ST - TS = AB - BA.$$

由于S和T 都是自伴算子, µ1, µ2都是形如(26)的内积, 都是实数, 所以A和B也都是自伴算子. 因而从均值的定义可得

$$\mu_{\psi}(C) = ((AB - BA)\psi, \psi) = (AB\psi, \psi) - (BA\psi, \psi)$$
$$= (B\psi, A\psi) - (A\psi, B\psi).$$

## 利用三角不等式及Schwarz不等式有

$$|\mu_{\psi}(C)| \le |(B\psi, A\psi)| + |(A\psi, B\psi)| \le 2||B\psi|| ||A\psi||.$$

因为B是自伴算子, 所以由式((27))可得

$$||B\psi|| = ((T - \mu_2 I)^2 \psi, \psi)^{\frac{1}{2}} = \sqrt{\operatorname{var}_{\psi}(T)} = \operatorname{sd}_{\psi}(T).$$

再通过类似的方法得到∥Aψ∥ = sdψ(S), 从而证明了式(29).

利用海森堡交换关系可以看出,位置算子和动量算子的换位子是 $\frac{h}{2\pi i}\tilde{I}$ ,因此 $|\mu_{\psi}(C)|=\frac{h}{2\pi}$ . 再结合式(29),得到

#### 定理13

对于位置算子Q和动量算子D,有

$$\mathrm{sd}_{\psi}(D)\mathrm{sd}_{\psi}(Q) \ge \frac{h}{4\pi}.\tag{30}$$

利用海森堡交换关系可以看出, 位置算子和动量算子的换位子 是 <sup>h</sup> 2πi ˜I, 因此|µψ(C)| = h 2π . 再结合式(29), 得到

## 定理13

对于位置算子Q和动量算子D, 有

$$\mathrm{sd}_{\psi}(D)\mathrm{sd}_{\psi}(Q) \ge \frac{h}{4\pi}.$$
 (30)

此定理称为海森堡不确定定理. 由于标准差sdψ(D)和sdψ(Q)分别 表征了动量和位置测量的精度, 所以式(30)意味着我们不能无限 精确地同时测量粒子的位置和动量.