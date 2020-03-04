A `Decision Tree` is a machine learning algorithm

Entropy: $H(x) = -\sum p(x)log_2 p(x)$

用entropy来确定一个model的`惊讶度`

  * High Entropy - 比如掷骰子，uniform distribution，每一种情况出现的可能都相似，所以难以预测
  * Low Entropy - distribution 有很多peak和valley，某些情况可能性很高，就更好预测。

Entropy也能计算joint distribution，eg.

![joint distribution](./images/decision_tree_01.png)

`Entropy`: $H(x,y) = -\sum\sum p(x,y)log_2 p(x,y)$

   $H(x,y)=-\frac{24}{100}log_2\frac{24}{100}-\frac{25}{100}log_2\frac{25}{100}-\frac{1}{100}log_2\frac{1}{100}-\frac{50}{100}log_2\frac{50}{100}\approx 1.56~bit$
   $H(Y|X)= \sum p(x)H(Y|X=x)$

`Information Gain` 表示一个dataset给了我们多少信息

$IG(Y |X) = H(Y ) − H(Y |X)$