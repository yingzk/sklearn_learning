# Python ML Library --- sklearn

## 线性模型--Par 2

### Lasso

Lasso是估计系数系数的线性模型，在一些情况下非常有用，因为它倾向于具有较少参数的解决方案，有效地减少给定解决方案所依赖的的变量的数量，为此，Lasso及其变体是compressed sensing(压缩感测领域)的基础，在某些条件下，它可以回复精确的非零权重集

在数学上，它有一个使用$l_{1}$先验作为正则化的线性模型组成，最小化目标函数是:

$\underset{\omega}{min}\frac{1}{2n_{samples}}||X_{\omega}-y||_{2}^{2} + \alpha||\omega||_{1}$

因此lasso estimate解决了加上罚项$\alpha||\omega||_{1}$的最小二乘法的最小化，其中, $\alpha$是常数， $||\omega||_{1}$是参数向量$l_{1}$范数

Lasso类中的实现使用坐标下降作为算法来拟合系数，查看最小角度回归用于另一个实现:

```python
>>> from sklearn import linear_model
>>> reg = linear_model.Lasso(alpha = 0.1)
>>> print reg.fit([[0, 0], [1, 1]], [0, 1])
Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
>>> print reg.predict([[1, 1]])
array([ 0.8])
```

对于较低级别的任务也很有用的是函数 **lasso_path** 来计算可能值的完整路径上的系数。



#### 设置正则化参数

alpha参数控制估计的系数的稀疏度

##### 使用交叉验证

sklearn 通过交叉验证来设置Lasso alpha对象[**LassoCV**](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV)和 [**LassoLarsCV** ](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV)。  **LassoLarsCV **是基于下面解释的 ** Least Angle Regression**  ( 最小角度回归 ) 算法。
对于具有许多共线回归的高维数据集， [**LassoCV** ](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV)最常见。然而， [**LassoLarsCV** ](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV)具有探索更相关的 alpha 参数值的优点，并且如果样本数量与观察次数相比非常小，则通常比 [**LassoCV** ](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV)快。



<img src="C:\Users\yzk13\Desktop\sklearn\page3\1.png" style="zoom:50%"><img src="C:\Users\yzk13\Desktop\sklearn\page3\2.png" style="zoom:50%">



#### Information-criteria based model selection ( 基于信息标准的模型选择 )

估计器 [**LassoLarsIC** ](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC)建议使用 **Akaike** 信息准则（**AIC**）和贝叶斯信息准则（**BIC**）。当使用** k-fold** 交叉验证时，正则化路径只计算一次而不是** k + 1** 次，所以找到 α 的最优值是一个计算上更便宜的替代方法。然而，这样的标准需要对解决方案的自由度进行适当的估计，为大样本（渐近结果）导出，并假设模型是正确的，即数据实际上是由该模型生成的。当问题条件差时，它们也倾向于打破（比样本更多的特征）。

<img src="C:\Users\yzk13\Desktop\sklearn\page3\3.png" style="zoom:50%">