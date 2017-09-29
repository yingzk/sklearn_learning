# 决策树(Decision Trees)

**决策树（DTS）**是一种非参数监督学习用于方法分类和回归，目标是创建一个模型，通过学习，从数据特征推断的简单决策规划来预测目标变量的值。

例如在下面的示例中，决策树从数据中学习，以一组if-then-else决策规划近似正弦曲线，树越深决策规划和训练模型越复杂。

<img src="C:\Users\yzk13\Desktop\sklearn\page4\1.png" style="zoom:50%">

### 决策树的优点

- 简单的理解和解释，决策树的可视化
- 需要很少量的数据准备，其它技术通常需要数据归一化，需要创建虚拟变量，并删除空值，**注意**此模块不支持缺失值
- 使用树的成本（即，预测数据）在用于训练树的数据点的数量上是对数的
- 能够处理数字和分类数据，其他技术通常专门用于分析只有一种变量类型的数据集
- 能够处理多输出的问题
- 使用白盒模型，如果给定的情况在模型中可以观察到，那么条件的解释很容易用布尔逻辑来解释，相比之下，在黑盒子模型（例如，在人造神经网络中），结果可能更难解释
- 可以使用统计测试验证模型，这样可以说明模型的可靠性
- 即使其假设被数据生成的真实模型有些违反，表现良好



### 决策树的缺点

- 决策学习者可以创建不能很好的推广数据的过于复杂的树，这被称为过拟合，修剪（不支持当前）的机制，设置叶系欸但那所需的最小样本数或设置树的最大深度是避免此问题的必要条件
- 决策树可能不稳定，因为数据的小变化可能会导致完全不同的树生成，通过使用合奏中的决策树来减轻这个问题
- 在最优性的几个方面甚至更简单的概念中，学习最优决策树的问题已知是NP完整的，因此，实际的决策树学习算法基于启发式算法，例如在每个节点进行局部最优决策的贪心算法，这样的算法不能保证返回全局最优决策树，这可以通过在综合学习者中训练多个树木来缓解，其中特征和样本随机抽样取代。
- 有一些难以学习的概念，因为决策树不能很容易地表达它们，例如XOR，奇偶校验或复用器问题。
- 如果某些类占主导地位，决策树学习者会创造有偏见的树木。因此，建议在拟合之前平衡数据集与决策树



## 分类

DecisionTreeClassifier是能够对数据集执行多类分类的类。

与其他分类器一样，DecisionTreeClassifier将两个数组：数组X， 稀疏或密集，大小为$[n\_samples, n\_features]$保存为训练样本，以及整数值的数组Y， 大小为$[n\_samples]$， 持有类标签，训练样本：

```python
from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

#预测分类
print (clf.predict([[2, 2]]))

# 预测被为某类的概率
print (clf.predict_proba([[2., 2.]]))

​````````````````````````````````````````
[1]
[[ 0.  1.]]
```



DecisionTreeClassifier能同时应用于二分类（标签为[-1, 1]）和多分类[0, 1, k-1]

用数据及iris，可以构造下面的树

```python
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
```

训练完成后，可以导出树为```Graphviz```格式，下面是导出iris数据集训练树的例子

```python

>>> with open("iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
```

然后我们可以使用的Graphviz的`dot`工具来创建一个PDF文件（或任何其他支持的文件类型）：

 `dot -Tpdf iris.dot -o iris.pdf`

```python
>>> import os
>>> os.unlink('iris.dot')
```

或者，如果我们安装了Python模块`pydotplus`，我们可以直接在Python中生成PDF文件（或任何其他支持的文件类型）：

```python
>>> import pydotplus
>>> dot_data = tree.export_graphviz(clf, out_file=None)
>>> graph = pydotplus.graph_from_dot_data(dot_data)
>>> graph.write_pdf("iris.pdf")
```

[export_graphviz](http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz)出口也支持多种美学选项，包括可以通过类着色节点（或值回归）和如果需要的话使用显式的变量和类名称。IPython笔记本还可以使用Image（）函数内联渲染这些图

```python
>>> from IPython.display import Image 
>>> dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names, 
                         class_names=iris.target_names, 
                         filled=True, rounded=True, 
                         special_characters=True) 
>>> graph = pydotplus.graph_from_dot_data(dot_data) 
>>> Image(graph.create_png())
```

<img src="C:\Users\yzk13\Desktop\sklearn\page4\2.png" style="zoom:50%">

安装后，可以使用该模型来预测样本类别：

```python
>>> clf.predict(iris.data[:1, :])
array([0])
```

或者，可以预测每个类的概率，这是叶中相同类的训练样本的分数：

```python
>>> clf.predict_proba(iris.data[:1, :])
array([[ 1.,  0.,  0.]])
```

例子：

- [在虹膜数据集上绘制决策树的决策面](http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html#sphx-glr-auto-examples-tree-plot-iris-py)

  对于每对虹膜特征，决策树由训练样本推断和简单的阈值组合制定的决策边界。


```python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:00:48 2017

@author: YingJoy
"""

# import some package and dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# parameters    n_classes 分为几类     bry: 蓝，红， 黄   step: 步长
n_classes = 3
plot_colors = 'bry'
plot_step = 0.02

# Load data
iris = load_iris()

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], \
                                [1, 2], [1, 3], [2, 3]]):
    # 仅取2个相应特征
    x = iris.data[:, pair]
    y = iris.target
    
    #  训练
    clf = DecisionTreeClassifier().fit(x, y)
    
    # 画决策边界   一幅图2行3列 选择第  pairidx + 1 个
    plt.subplot(2, 3, pairidx + 1)
    
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), \
                         np.arange(y_min, y_max, plot_step))
    
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, z, cmap = plt.cm.Paired)
    
    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")
    
    # plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(x[idx, 0], x[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired)
        plt.axis("tight")
        
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()
```

