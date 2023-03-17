# Machine learning

gradient descent algorithm：
$$
\theta 1:=\theta 1-\alpha \frac{dJ(\theta1)}{d\theta_1}
$$
$$
J(\theta)=\frac{1}{2m}\sum_{j=0}^m(h_\theta(x_j)-y_j)^2
$$

$$
\theta1:=\theta1-\alpha
\frac{1}{m}(h_\theta(x_j)-y_j)\frac{dh_{\theta1}}{d\theta1}
$$



the α should be adjust mentally,if the α is very small although it can converge, it regrate very slow.
if the α is very large,the θ will diverge and can not converge.

- bow-shaped function：

  always has a global optimum；

  it's called ***convex function***;

## matrix:

associative：结合律
commutative：交换律
inverse：
$$
AA^{-1}=A^{-1}A=I
$$
only square matrix has its inverses 

singular matrix（奇异矩阵）、degenerate matrix（退化矩阵）have no inverse；

transpose（转置）:
$$
A=\begin{bmatrix}
1,2,0\\3,4,9
\end{bmatrix},A^{T}=\begin{bmatrix}1,3\\2,4\\0,9\end{bmatrix}
$$

$$
A_{ij}=A^{T}_{ji}
$$

## logistic regression：

$$
sigmod:g(y)=\frac{1}{1+e^{-y}}=\frac{e^y}{1+e^y}
$$

$$
g'(y)=(1-g(y))g(y)
$$

把线性回归函数的结果y，放到sigmod函数中去，就构造了**逻辑回归函数**

##  最大似然估计：

$$
L(\theta|x)=P(x|\theta)
$$

- 求概率$$P(x|\theta)$$是确定了$$\theta$$，问当X生成x时概率多大；
- 而$$L(\theta|x)$$则是求当x确定时，$$\theta$$怎么取得到$$P(x|\theta)$$最大；