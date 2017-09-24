# Python ML Library --- sklearn

## 线性模型--Par 2

### 岭回归（Ride Regression）

岭回归分析是一种专用于共线性数据分析的有偏估计回归方法，实质上是一种改良的最小二乘估计法，通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价获得回归系数更为符合实际、更可靠的回归方法，对病态数据的耐受性远远强于最小二乘法。岭回归分析主要解决两类问题：数据点少于变量个数；变量间存在共线性。



岭回归引入了一种对系数大小进行惩罚的措施，来解决普通最小二乘法可能遇到的某些问题，岭回归最小化带有惩罚的残差平方和

​											$\underset{\omega}{min}||X_{\omega} - y||_{2}^2 + \alpha||\omega||_2^2$

这里，$\alpha > 0$是一个复杂的参数，用于控制系数的缩减量。$\alpha$的值越大，系数缩减的越多，因而会对共线性更加鲁棒（稳定性）

<img src="C:\Users\yzk13\Desktop\sklearn\page2\1.png", style="zoom:50%">

与其他线性模型一样。Ridge类的成员函数fit以数组x, y作为输入，并将线性模型的系数$\omega$存储在其变量$coef\_$中：

```python
#导入线性模型
>>> from sklearn import linear_model
#创建线性模型中的岭回归，设置alpha为0.5
>>> reg = linear_model.Ridge(alpha = .5)
#训练			0:1步长为0.1
>>> print reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
>>> print reg.coef_
[ 0.34545455  0.34545455]
>>> print reg.intercept_
0.136363636364
```

### 例子:

- 岭回归系数图像：一种正则化的方法

  ### 绘制岭回归图像

  本例使用了岭回归

  本例还显示了将岭回归应用与高度病态矩阵的有效性，病态矩阵，目标变量的稍微变化可能导致最终计算权重有巨大差异，在这种情况下一般设置一定的正则化(alpha)，来减少这种变化

  当alpha非常大的时候，正则化效果使得函数的系数趋近于0，在最后，由于alpha趋于0，解决方案趋向与普通最小二乘法，所以系数振荡变大，在现实例子中，可以调整alpha，保持平衡

  ```python
  # -*- coding: utf-8 -*-
  """
  Created on Sun Sep 24 20:50:44 2017

  @author: YingJoy
  """

  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn import linear_model

  #X是10x10的希尔伯特矩阵
  X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
  y = np.ones(10)

  #########################计算###########################
  # 定义alpha的数量
  n_alphas= 200
  # 建立alphas向量    
  # 对数等分 log以10为底  在-10 到-2之间分为n_alphas份   这里的alpha非常的小
  alphas = np.logspace(-10, -2, n_alphas)

  # 系数向量
  coefs = []
  # 循环alphas，一共n_alpha个岭回归模型，
  # 分别以不同的alpha，训练, 最终保存所有模型计算出的系数到coefs向量中
  for a in alphas:
      ridge = linear_model.Ridge(alpha = a, fit_intercept = False)
      ridge.fit(X, y)
      coefs.append(ridge.coef_)
  ########################结束计算#########################

  # 画图，显示结果   
  # plt.gca()获取当前轴的对象ax，然后通过ax.plot()画图
  ax = plt.gca()
  ax.plot(alphas, coefs)
  #将x轴取对数
  ax.set_xscale('log')
  #翻转x轴
  ax.set_xlim(ax.get_xlim()[::-1])
  plt.xlabel('alpha')
  plt.ylabel('weights')
  plt.title('Ridge coefficients as a function of the regularization')
  plt.axis('tight')
  plt.show()
  ```


- 使用稀疏特征进行进行文本文档分类

  来自[官方文档](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py)

  ```python
  # Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
  #         Olivier Grisel <olivier.grisel@ensta.org>
  #         Mathieu Blondel <mathieu@mblondel.org>
  #         Lars Buitinck
  # License: BSD 3 clause

  from __future__ import print_function

  import logging
  import numpy as np
  from optparse import OptionParser
  import sys
  from time import time
  import matplotlib.pyplot as plt

  from sklearn.datasets import fetch_20newsgroups
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.feature_extraction.text import HashingVectorizer
  from sklearn.feature_selection import SelectFromModel
  from sklearn.feature_selection import SelectKBest, chi2
  from sklearn.linear_model import RidgeClassifier
  from sklearn.pipeline import Pipeline
  from sklearn.svm import LinearSVC
  from sklearn.linear_model import SGDClassifier
  from sklearn.linear_model import Perceptron
  from sklearn.linear_model import PassiveAggressiveClassifier
  from sklearn.naive_bayes import BernoulliNB, MultinomialNB
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.neighbors import NearestCentroid
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.utils.extmath import density
  from sklearn import metrics


  # Display progress logs on stdout
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(levelname)s %(message)s')


  # parse commandline arguments
  op = OptionParser()
  op.add_option("--report",
                action="store_true", dest="print_report",
                help="Print a detailed classification report.")
  op.add_option("--chi2_select",
                action="store", type="int", dest="select_chi2",
                help="Select some number of features using a chi-squared test")
  op.add_option("--confusion_matrix",
                action="store_true", dest="print_cm",
                help="Print the confusion matrix.")
  op.add_option("--top10",
                action="store_true", dest="print_top10",
                help="Print ten most discriminative terms per class"
                     " for every classifier.")
  op.add_option("--all_categories",
                action="store_true", dest="all_categories",
                help="Whether to use all categories or not.")
  op.add_option("--use_hashing",
                action="store_true",
                help="Use a hashing vectorizer.")
  op.add_option("--n_features",
                action="store", type=int, default=2 ** 16,
                help="n_features when using the hashing vectorizer.")
  op.add_option("--filtered",
                action="store_true",
                help="Remove newsgroup information that is easily overfit: "
                     "headers, signatures, and quoting.")


  def is_interactive():
      return not hasattr(sys.modules['__main__'], '__file__')

  # work-around for Jupyter notebook and IPython console
  argv = [] if is_interactive() else sys.argv[1:]
  (opts, args) = op.parse_args(argv)
  if len(args) > 0:
      op.error("this script takes no arguments.")
      sys.exit(1)

  print(__doc__)
  op.print_help()
  print()


  # #############################################################################
  # Load some categories from the training set
  if opts.all_categories:
      categories = None
  else:
      categories = [
          'alt.atheism',
          'talk.religion.misc',
          'comp.graphics',
          'sci.space',
      ]

  if opts.filtered:
      remove = ('headers', 'footers', 'quotes')
  else:
      remove = ()

  print("Loading 20 newsgroups dataset for categories:")
  print(categories if categories else "all")

  data_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=42,
                                  remove=remove)

  data_test = fetch_20newsgroups(subset='test', categories=categories,
                                 shuffle=True, random_state=42,
                                 remove=remove)
  print('data loaded')

  # order of labels in `target_names` can be different from `categories`
  target_names = data_train.target_names


  def size_mb(docs):
      return sum(len(s.encode('utf-8')) for s in docs) / 1e6

  data_train_size_mb = size_mb(data_train.data)
  data_test_size_mb = size_mb(data_test.data)

  print("%d documents - %0.3fMB (training set)" % (
      len(data_train.data), data_train_size_mb))
  print("%d documents - %0.3fMB (test set)" % (
      len(data_test.data), data_test_size_mb))
  print("%d categories" % len(categories))
  print()

  # split a training set and a test set
  y_train, y_test = data_train.target, data_test.target

  print("Extracting features from the training data using a sparse vectorizer")
  t0 = time()
  if opts.use_hashing:
      vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                     n_features=opts.n_features)
      X_train = vectorizer.transform(data_train.data)
  else:
      vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                   stop_words='english')
      X_train = vectorizer.fit_transform(data_train.data)
  duration = time() - t0
  print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
  print("n_samples: %d, n_features: %d" % X_train.shape)
  print()

  print("Extracting features from the test data using the same vectorizer")
  t0 = time()
  X_test = vectorizer.transform(data_test.data)
  duration = time() - t0
  print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
  print("n_samples: %d, n_features: %d" % X_test.shape)
  print()

  # mapping from integer feature name to original token string
  if opts.use_hashing:
      feature_names = None
  else:
      feature_names = vectorizer.get_feature_names()

  if opts.select_chi2:
      print("Extracting %d best features by a chi-squared test" %
            opts.select_chi2)
      t0 = time()
      ch2 = SelectKBest(chi2, k=opts.select_chi2)
      X_train = ch2.fit_transform(X_train, y_train)
      X_test = ch2.transform(X_test)
      if feature_names:
          # keep selected feature names
          feature_names = [feature_names[i] for i
                           in ch2.get_support(indices=True)]
      print("done in %fs" % (time() - t0))
      print()

  if feature_names:
      feature_names = np.asarray(feature_names)


  def trim(s):
      """Trim string to fit on terminal (assuming 80-column display)"""
      return s if len(s) <= 80 else s[:77] + "..."


  # #############################################################################
  # Benchmark classifiers
  def benchmark(clf):
      print('_' * 80)
      print("Training: ")
      print(clf)
      t0 = time()
      clf.fit(X_train, y_train)
      train_time = time() - t0
      print("train time: %0.3fs" % train_time)

      t0 = time()
      pred = clf.predict(X_test)
      test_time = time() - t0
      print("test time:  %0.3fs" % test_time)

      score = metrics.accuracy_score(y_test, pred)
      print("accuracy:   %0.3f" % score)

      if hasattr(clf, 'coef_'):
          print("dimensionality: %d" % clf.coef_.shape[1])
          print("density: %f" % density(clf.coef_))

          if opts.print_top10 and feature_names is not None:
              print("top 10 keywords per class:")
              for i, label in enumerate(target_names):
                  top10 = np.argsort(clf.coef_[i])[-10:]
                  print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
          print()

      if opts.print_report:
          print("classification report:")
          print(metrics.classification_report(y_test, pred,
                                              target_names=target_names))

      if opts.print_cm:
          print("confusion matrix:")
          print(metrics.confusion_matrix(y_test, pred))

      print()
      clf_descr = str(clf).split('(')[0]
      return clf_descr, score, train_time, test_time


  results = []
  for clf, name in (
          (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
          (Perceptron(n_iter=50), "Perceptron"),
          (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
          (KNeighborsClassifier(n_neighbors=10), "kNN"),
          (RandomForestClassifier(n_estimators=100), "Random forest")):
      print('=' * 80)
      print(name)
      results.append(benchmark(clf))

  for penalty in ["l2", "l1"]:
      print('=' * 80)
      print("%s penalty" % penalty.upper())
      # Train Liblinear model
      results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                         tol=1e-3)))

      # Train SGD model
      results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                             penalty=penalty)))

  # Train SGD with Elastic Net penalty
  print('=' * 80)
  print("Elastic-Net penalty")
  results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                         penalty="elasticnet")))

  # Train NearestCentroid without threshold
  print('=' * 80)
  print("NearestCentroid (aka Rocchio classifier)")
  results.append(benchmark(NearestCentroid()))

  # Train sparse Naive Bayes classifiers
  print('=' * 80)
  print("Naive Bayes")
  results.append(benchmark(MultinomialNB(alpha=.01)))
  results.append(benchmark(BernoulliNB(alpha=.01)))

  print('=' * 80)
  print("LinearSVC with L1-based feature selection")
  # The smaller C, the stronger the regularization.
  # The more regularization, the more sparsity.
  results.append(benchmark(Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                    tol=1e-3))),
    ('classification', LinearSVC(penalty="l2"))])))

  # make some plots

  indices = np.arange(len(results))

  results = [[x[i] for x in results] for i in range(4)]

  clf_names, score, training_time, test_time = results
  training_time = np.array(training_time) / np.max(training_time)
  test_time = np.array(test_time) / np.max(test_time)

  plt.figure(figsize=(12, 8))
  plt.title("Score")
  plt.barh(indices, score, .2, label="score", color='navy')
  plt.barh(indices + .3, training_time, .2, label="training time",
           color='c')
  plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
  plt.yticks(())
  plt.legend(loc='best')
  plt.subplots_adjust(left=.25)
  plt.subplots_adjust(top=.95)
  plt.subplots_adjust(bottom=.05)

  for i, c in zip(indices, clf_names):
      plt.text(-.3, i, c)
      
  plt.show()
  ```

  ​

### 岭回归的时间复杂度

岭回归的复杂度与普通最小二乘法一样

#### 使用广义交叉验证设置正则化参数

RidgeCV使用内置的交叉验证方法选择参数$\alpha$，进而实现了岭回归，该对象和GridSearchCV的原理类似，只不过RidgeCV默认使用广义交叉验证方法（留一交叉验证的一种高效形式）:

```python
>>> from slearn import linear_model
>>> clf = learn_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
>>> print clf.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, gcv_mode=None,
    normalize=False, scoring=None, store_cv_values=False)
>>> print clf.alpha
0.1
```
