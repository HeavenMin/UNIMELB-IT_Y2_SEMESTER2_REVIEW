# COMP90051 Statistical Machine Learning
#Review/ML
---
## L1 - Introduction Probability Theory
### ML system consist of
* __Instance__ : measurements about individual entities
* __Attribute__ (feature) : component of the instance
* __Label__ : an outcome that is categorical, numeric
* __Examples__ :  instance coupled with label
* __Models__ : discovered relationship between attributes, label

### Learning
* __supervised learning__
	* Labelled
	* predict labels on new instances
* __unsupervised learning__
	* Unlabelled
	* cluster related instances
	* understand attribute relationships

### Evaluation
* Pick an __evaluation metric__ comparing label vs prediction
* Procure an independent, labelled test set
* “Average” the evaluation metric over the test set
### Evaluation Metrics
* Accuracy
* Contingency table
* Precision-Recall
* ROC curves
> When data poor, using __cross-validation__

### Random Variables (r.v.’s)
[r.v.](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F) :
A random variable X is a numeric function.

### Expectation and Variance
$E[X]$ is the r.v. X’s “average” value
$$
\text{Discrete: } E[X] = \sum_{x}xP(X=x)\\
\text{Continuous: } E[X] = \int_{x}xp(x)dx\\
E[aX+b] = aE[X]+b \hspace{2em} E[X+Y] = E[X] + E[Y]\\
X \geq Y \Rightarrow E[X] \geq E[Y]\\
\text{Variance: } Var(X) = E[(X - E[X])^2]
$$

### Bayes’s Theorem
* $P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$
* $P(A|B) = \dfrac {P(B|A)P(A)}{P(B)}$
	* Marginals: probabilities of individual variables
	* Marginalisation: summing away all but r.v.’s of interest
####Bayes's rule applies to Bayesian modelling
$P(parameters|data) = \dfrac {P(data|parameters)P(parameters)}{P(data)}$
* posterior = whole expression
* likelihood = $P(data|parameters)$
* prior = $P(parameters)$
* marginal likelihood = $P(data)$
---
## L2 - Statistical schools
### MLE- Maximun-likelihood Estimation
$\widehat {\theta }\left( x_{1},\ldots ,x_{n}\right) =argmax_{\theta \in \Theta }\prod ^{n}_{i=1}p_{\theta }\left( x_{i}\right)$  
__$\hat{\theta} = argmax_\theta P(X=x|\theta)$__ `MLE`
### [MAP](http://www.cnblogs.com/sylvanas2012/p/5058065.html)
$\hat{\theta}=P(\theta|X = x)$  
$= argmax_\theta \dfrac {P(X = x|\theta)P(\theta)}{P(X = x)}$  
__$= argmax_\theta P(X = x|\theta)P(\theta)$__ `MAP`

### [MLE ‘algorithm’](https://www.zhihu.com/question/19725590/answer/217025594)
* given data $x_1,...,x_n$ __define__ probability distribution, $p_\theta$, assumed to have generated the data
* express likelihood of data, $\prod^{n}_{i=1}p_\theta(x_i)$
* optimise to find best parameters $\hat{\theta}$

### Parametetric vs Non-parametric Models
Parametric | Non-parametric
---------- | --------------
Determined by fixed, finite number of parameters | Number of parameters grows with data, infinite
Limited flexibility | More flexible
Efficient statistically and computationally | Less Efficient

### Generative vs. Discriminative Models
* For G: Model full joint $P(X,Y)$
* For D: Model conditional $P(Y|X)$ only
---
## L3 - Linear Regression & Regularisation


































---
