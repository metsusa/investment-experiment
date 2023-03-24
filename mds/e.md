#  方法：

$$\alpha\ \beta\ \gamma$$分别为3个Q矩阵的权重，即：

将$$Q(\alpha,\beta,\gamma)$$分为三个矩阵

由于我们的各个投入项不能简单地归类到哪一个具体意图中，所以需要把它们正交分解到三个意图分组中.

1. subagent: 学院—>agent: 投入项—>subrole:产出细项->role: 产出大项

2. $$L_s[j]$$:学院j的可投入能力.	
3. $$W_s[j]$$:投入项的重要性，跟三元组无关.
4. $$Q_s$$:学院投入的意向，跟三元组相关.

5. L[j]:投入项j的可投入能力	
6. W[k]:产出项的重要性，跟三元组相关.

7. Q:投入产出相关度，evaluation生成，与三元组无关.	
8. T:最终投入矩阵.	

$$\alpha、\beta、\gamma$$跟且仅跟$$Q_{s\alpha}、Q_{s\beta}、Q_{s\gamma}$$相关.

通过指派，得到$$T_s$$而T则是基于$$T_s$$的分配
$$
L[j]={\sum_i^m{T[i][j]}}
$$
通过evaluation得到每个学院对某项院内投入指标的投入与学校整体的某项发展指标的相关度，每个投入有不同的偏向性，比如因此有对应的W向量进行表示，其中：$$Q_\alpha\ Q_\beta\ Q_\gamma$$分别为三个group的Q矩阵；
$$
max \ \sigma =\sum_i^m{\sum_j^n{\alpha*Q_\alpha[i][j]*T_\alpha[i][j]
+\beta*Q_\beta[i][j]*T_\beta[i][j]
+\gamma*Q_\gamma[i][j]*T_\gamma[i][j]}}
$$


$$T_\alpha\ T_\beta\ T_\gamma$$分别为3个group的指派矩阵；

要求得:
		$$Max\{\sum_{k=\alpha}^\gamma\sum_{i=1}^m\sum_{j=1}^nk*Q_{[k,i,j]}\}$$

并以此作为基础得到热力图，通过gradient decent不断改变$$\alpha\ \beta\ \gamma$$的值得到最符合当前结果的$$\alpha\ \beta\ \gamma$$值，推得当前发展意图；

由新的$$\alpha\ \beta\ \gamma$$值来得到最优的投入热力图；


$$

$$
