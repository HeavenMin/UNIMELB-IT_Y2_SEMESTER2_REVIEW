---
typora-copy-images-to: ./image
---

# COMP90051 Statistical Machine Learning
#### Review/ML

`Author: Min Gao`

> `important aspects`
>
> Basis expansion, representation, optimisation, loss functions, regularisation, overfitting

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
[r.v.](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F) : A random variable X is a numeric function.

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

  ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot.png)
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
| Parametric                               | Non-parametric                           |
| ---------------------------------------- | ---------------------------------------- |
| Determined by fixed, finite number of parameters | Number of parameters grows with data, infinite |
| Limited flexibility                      | More flexible                            |
| Efficient statistically and computationally | Less Efficient                           |

### Generative vs. Discriminative Models
* For G: Model full joint $P(X,Y)$
* For D: Model conditional $P(Y|X)$ only
---
## L3 - Linear Regression & Regularisation

### Loss function

* 0-1 loss function
  * $L(Y, f(X)) = \{ ^{1,Y\neq f\left( x\right) }_{0,Y=f\left( x\right) }$
* quadratic loss function (squared loss)
  * $L(Y,f(X)) = (Y - f(X))^2$
* absolute loss function
  * $L(Y,f(X)) = |Y - f(X)|$
* logrithmic loss function
  * $L(Y, P(Y|X)) = -logP(Y|X)$

### Linear Regression

> it's simple, easier to understand, computationally efficient

__Example__: $H = a +bT$, find parameter values

* To find $a,b$ that minimise $L=\sum ^{10}_{i=1}\left( H_{i}-\left( a+bT_{i}\right) \right) ^{2}$
  * write derivative
  * set to zero
  * solve for model
* using np.linalg.solve to solve ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/linearEquations.png)
* __Coordinate descent__: guess a, solve for b, solve for a; repeat

__Linear Regression model__: A simple model that tends to require less data and be easier to interpret.

* $y \approx w_0 + \sum^{m}_{i=1}{x_iw_i}$

_Trick_: add a dummy feature $x_0 = 1$ and use vector notation

* $y \approx \sum^{m}_{i=0}{x_iw_i} = \boldsymbol{x'w} = \boldsymbol{x^Tw}$

### Regression as a probabilistic model

$y =  \boldsymbol{x^Tw} + \epsilon$

$\epsilon$ is noise, using "Log trick":

$\sum^{n}_{i=1}logp(y_i|\boldsymbol{x_i}) = -\dfrac {1}{2\sigma^2}\sum^{n}_{i=1}(y_i - (\boldsymbol{(x_i)^Tw})^2) + K$

Therefore, under this model, maximising log-likelihood as a function of w is equivalent to minimsing the sum of squared errors.

### Method of least squares

The model assumes $\boldsymbol{y \approx Xw}$

To find $\boldsymbol{w}$, minimise the __sum of squared errors__

$L = \sum^{n}_{i=1}(y_i - \sum^{m}_{j=0}X_{ij}w_j)^2 = ||\boldsymbol{y - Xw}||^2$

Setting derivative to zero and solving for $\boldsymbol{w}$ yields

$\boldsymbol{\hat{w}= (X'X)^{-1}X'y}$     `normal equations`: this system is defined only if the __inverse__ exists

### Regularisation

> Process of introducing additional information in order to solve an ill-posed problem or to prevent overfitting ---- (wiki)  (not just for linear methods)

* avoid ill-conditioning
  * irrelevant features
    * Feature $\boldsymbol{X}_{.j}$ is irrelevant if $\boldsymbol{X}_{.j}$ is  linear combination of other columns
      * $\boldsymbol{X}_{.j} = \sum_{l \neq j} \alpha_l \boldsymbol{X}_{.l}$
    * The normal equations solution: $\boldsymbol{\hat{w}= (X'X)^{-1}X'y}$. __With irrelevant features, $\boldsymbol{X'X}$ has no inverse__
  * lack of data
* Introduce prior knowledge
* Constrain modelling (vary the model complexity) — to deal with __underfitting__ and __overfitting__
  * Explicit model selection
    * Try different classes of models
    * use held out validation to select the model (split training data into training set and validation set)
  * Regularisation
    * Augment the problem: $\boldsymbol{\theta} = argmin_{\boldsymbol{\theta} \in \Theta}(L(data, \boldsymbol{\theta}) + \lambda R(\boldsymbol(\theta))$
    * E.g. ridge regression (L2): $||\boldsymbol{y - Xw}||^2_2 + \lambda ||\boldsymbol{w}||^2_2$
    * E.g. Lasso (L1): $||\boldsymbol{y - Xw}||^2_2 + \lambda ||\boldsymbol{w}||_1$
    * Use held out validation/cross validation to choose $\lambda$

#### L1 and L2 norms

> Intuitively, norms measure lengths of vectors in some sense

* L2 (Euclidean distance): $||\boldsymbol{a}||=||\boldsymbol{a}||_2 \equiv \sqrt {a^2_1+…+a^2_n}$
* L1 (Manhattan distance): $||\boldsymbol{a}||_1 \equiv |a_1|+…+|a_n|$

#### Re-conditioning the problem

Using regularisation to introduce an additional condition into the system

* the original problem is to minimise $||\boldsymbol{y - Xw}||^2_2$
* The regularisation problem is to minimise $||\boldsymbol{y - Xw}||^2_2 + \lambda ||\boldsymbol{w}||^2_2$ for $\lambda > 0$
* So the solution is now $\boldsymbol{\hat{w}= (X'X} + \lambda \boldsymbol{I)^{-1}X'y}$  __`ridge regression`__

#### Regulariser as a prior

> TODO `question`

---

## L4 - Logistic Regression & Basis Expansion

> logistic regerssion model is a linear method for binary classification

### Methods for binary classification

* Logistic regression
* perceptron
* SVM (support vector machines)

### Logistic regression model

Problem: the probability needs to be between 0 and 1. Need to squash the functio

__logistic funciton__: $f(s) = \dfrac {1}{1 + exp(-s)}$ 

__model__: $P(y = 1|\boldsymbol{X}) = \dfrac {1}{1 + exp(-\boldsymbol{x'w})}$

__Logistic regression is a linear classifier__

__Classification rule__: if $P(y=1|\boldsymbol{x} > \dfrac {1}{2})$ then class "1", else class "0"

__Decision boundary__: $ \dfrac {1}{1 + exp(-\boldsymbol{x'w})} = \dfrac {1}{2} \therefore \boldsymbol{x'w} = 0$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8831913.png)

__decision boundary__ is the line where $P(y=1|\boldsymbol{x}) = 0.5$

* 就是能够将所有数据点进行很好地分类的边界


* in higher dimensional problems, the decision boundary is a plane or hyperplane
* vector $\boldsymbol{w}$ is perpendicular(垂直) (normal) to the decision boundary

### Linear and logistic probabilistic model

__Linear regression__ assumes a _Normal distribution_ with a fixed variance and mean given by linear model:

$p(y|\boldsymbol{x} = Normal(y|\boldsymbol{x'w}, \sigma^2))$

__Logistic regression__ assumes a Bernoulli distribution with parameter given by logistic transform of linear model:

$p(y|\boldsymbol{x}) = Bernolli(y|\theta(\boldsymbol{x}) = \dfrac {1}{1 - exp(-\boldsymbol{x'w})}$

Bernoulli distribution is defined as $p(y) = \theta^y(1-\theta)^{(1-y)}$ for $y \in \{0,1\}$

#### Training as maximising likelihood estimation

$p(y_1,….y_n|\boldsymbol{x_1,…,x_n}) = \prod^{n}_{i=1}p(y_i|\boldsymbol{x_i})$

$\because p(y_i|\boldsymbol{x_i}) = \theta(\boldsymbol{x_i})^{y_i}(1-\theta (\boldsymbol{x_i}))^{(1-y_i)}$ and $\theta (\boldsymbol{x_i}) = \dfrac {1}{1 + exp(\boldsymbol{-x_i'w})}$

using log trick: $\log (\prod^{n}_{i=1}p(y_i|\boldsymbol{x_i})) = \sum^n_{i=1} \log p(y_i|\boldsymbol{x_i})$

$= \sum^n_{i=1} \log (\theta(\boldsymbol{x_i})^{y_i}(1-\theta (\boldsymbol{x_i}))^{(1-y_i)})$

$= \sum^n_{i=1}(y_i \log (\theta(\boldsymbol{x_i})) + (1-y_i)\log (1 - \theta(\boldsymbol{x_i})))$

$= \sum^n_{i=1}((y_i -1)\boldsymbol{x_i'w} - log(1+ exp(-\boldsymbol{x_i'w})))$

#### Cross entropy 

> is a method for comparing two distributions

$H(g_{ref}, g_{est}) = -\sum_{a \in A}g_{ref}(a)\log g_{est}(a)$

Logistic regression aims to estimate this distribution as

$g_{est}(1) = \theta(\boldsymbol{x_i})$ and $g_{est}(0) = 1 - \theta (\boldsymbol{x_i})$

#### optimisation for logistic regression

Training for logistic regression is amounts to finding $\boldsymbol{w}$ that maximise log-likelihood

​	same as finding $\boldsymbol{w}$ that minimise the sum of cross entropies for each training point

__no closed form solution__ — `stochastic gradient descent` is used

### Basis Expansion

> Extending the utility of models via data transformation, can be applied for both regression and classification

#### Transform the data

> Map the data onto another features space, such that the data is linear in that space

* Denote this transformation $\varphi :\mathbb{R} ^{m}\rightarrow \mathbb{R} ^{k}$. 
  * if $\boldsymbol{x}$ is the original set of features 
  * $\varphi(\boldsymbol{x})$ denotes the new set of features

example for polynomial regression: $y = w_0 + w_1x + w_2x^2$

```
example:
Consider a 2-dimensional dataset, where each point is represented by two features and the label (x1, x2, y). The features are binary, the label is the result of XOR function, and so the data consists of four points (0, 0, 0), (0, 1, 1), (1, 0, 1) and (1, 1, 0). Design a feature space transformation that would make the data linearly separable:
Answer: new feature space (x3), where x3 = (x1 − x2)^2
```

#### RBF (Radial basis funcitons)

* is a function of form $\varphi(\boldsymbol{x}) = \psi (||\boldsymbol{x-z}||)$
* Examples: $\varphi(\boldsymbol{x}) = exp(-\dfrac {1}{\sigma}||\boldsymbol{x-z}||^2)$
* one limitation is that the transformation needs to be defined beforehand

---

## L5 - Optimisation & Regularisation

### Iterative Optimisation

#### frequentist supervised learning

* Assume a model
  * Denote parameters of the model as $\theta$
  * Model predictions are $\hat{f}(\boldsymbol{x,\theta})$
* Choose a way to measure discrepancy between predictions and training label
  * E.g. sum of squared residuals $||\boldsymbol{y - Xw}||^2$
* Traing = parameter estimation = optimisation
  * $\hat{\theta} = argmin_{\theta \in \Theta}L(data, \boldsymbol{\theta})$

### Loss functions

> using to measure discrepancy prediction and label

Examples:

* squared loss $l_{sq} = (y - \hat{f}(\boldsymbol{x,\theta}))^2$
* absolute loss $l_{abs} = |y - \hat{f}(\boldsymbol{x,\theta})|$
* preceptron loss
* Hinge loss

### Solving optimisation problems

* Analytic solution
  * known only in limited number of cases
  * Use necessary condition: $\dfrac {\partial L}{\partial \theta_1} = …= \dfrac {\partial L}{\partial \theta_p} = 0$
* Approximate iterative solution
  1. Init: choose starting guess $\boldsymbol{\theta^{(1)}}$, set $i = 1$
  2. Update: $\boldsymbol{\theta^{(i+1)}} \leftarrow SomeRule [\boldsymbol{\theta^{(i)}}]$, set $i \leftarrow i + 1$
  3. Termination: decide whether to __Stop__
  4. Go to step 2
  5. __Stop__: return $\boldsymbol{\hat{\theta} \approx \theta^{(i)}}$

#### Coordinate descent

* Suppose $\boldsymbol{\theta} = [\theta_1,…,\theta_K]'$
  1. Choose $\boldsymbol{\theta^{(1)}}$ and some T
  2. For i from 1 to T*
     1. $\boldsymbol{\theta^{(i+1)}} \leftarrow \boldsymbol{\theta^{(i)}}$
     2. For j from 1 to K
        1. Fix components of $\boldsymbol{\theta^{(i+1)}}$, expect j-th component
        2. FInd $\hat{\theta}^{i+1}_j$ that minimises $L(\theta^{(i+1)}_j)$
        3. Update j-th component of $\boldsymbol{\theta^{(i+1)}}$
  3. Return $\boldsymbol{\hat{\theta} \approx \theta^{(i)}}$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8843665.png)

#### Gradient

$\nabla L = \left[\dfrac {\partial L}{\partial \theta_1},…,\dfrac {\partial L}{\partial \theta_p}\right]'$ computed at point $\boldsymbol{\theta}$

$\nabla$ is nabla symbol

#### Gradient descent [wiki](https://baike.baidu.com/item/梯度下降/4864937?fr=aladdin)

1. Choose $\boldsymbol{\theta^{(1)}}$ and some T
2. For i from 1 to T*
   1. $\boldsymbol{\theta^{(i+1)}} = \boldsymbol{\theta^{(i)}} - \eta\nabla L(\boldsymbol{\theta^{(i)}})$
3. Return $\boldsymbol{\hat{\theta} \approx \theta^{(i)}}$

> $\eta$ is dynamically updated in each step

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8845064.png)

#### Stochastic gradient descent

* split all training data in B batches
* choose init $\boldsymbol{\theta^{(1)}}$
* For i from 1 to T (epochs is the iterations over the entire dataset)
  * For j from 1 to B
    * Do gradient descent update using data from batch j

__Pros__: computational feasibility for large datasets 

### Bias-variance trade-off

> Generalisation capacity of the model is an important consideraition

Training the model is to minimisation of training error, generalisation capacity is captured by the __test error__. Also, __Model complexity__ is a major factor that influences the ability of the model to generalise.

Lemma: $test Error For x_0 = (bias)^2 + variance + irreducible Error$

simple model $\rightarrow$ low variance, high bias

complex model  $\rightarrow$ high variance, low 

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8845714.png)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8845766.png)

---

## L6 - Linear Algebra & Perceptron

### Dot product

$\boldsymbol{u\cdot v \equiv u'v} \equiv \sum^m_{i=1}u_iv_i \equiv ||\boldsymbol{u}||||\boldsymbol{v}||cos\theta$

* if two vectors are orthogonal then $\boldsymbol{u'v}$ = 0

### HyperPlane and normal vectors

> a hyperplane defined by parameters $\boldsymbol{w}$ and $b$, is a set of points $\boldsymbol{x}$ that satisfy $\boldsymbol{x'w} +b=0$

normal vector (法线) for a hyperplane is a vector perpendicular to that hyperplane

Besides, vector $\boldsymbol{w}$ is a normal vector to the hyperplane

### Preceptron model

> also a linear binary classifier, but a building block for artificial neural network

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8899898.png)

$w_0$ is bias weight and $f$ is activation function `question` : compare this model to logistic regression

Predict class A if $s \geq 0$ and Predict class B if $s < 0$ where $s = \sum^m_{i=0}x_iw_i$

#### loss function for perceptron

* $L(s,y)=0$ if both $s,y$ have the same sign
* $L(s,y) = |s|$ if both $s,y$ have different sign
* $\therefore L(s,y) = \max (0,-sy)$

#### Perceptron training algorithm

* Choose initial guess $\boldsymbol{w}^{(0)}, k=0$
* For i from 1 to T (epochs)
  * For j from 1 to N (training data)
    * data {$\boldsymbol{x_j},y_j$}
    * Update*: $\boldsymbol{w}^{(k++)} = \boldsymbol{w}^{(k)} - \eta \nabla L(\boldsymbol{w}^{(k)})$

Therefore, when classified correctly, weights are unchanged

When misclassified: $\boldsymbol{w}^{(k+1)} = -\eta(\pm\boldsymbol{x})$

| if $y=1$ but $s < 0$            | if $y=-1$ but $s \geq 0$        |
| ------------------------------- | ------------------------------- |
| $w_i \leftarrow w_i + \eta x_i$ | $w_i \leftarrow w_i - \eta x_i$ |
| $w_0 \leftarrow w_0 + \eta$     | $w_0 \leftarrow w_0 - \eta$     |

__Pros__: If the data is linearly separable, the perceptron training algorithm will converge to a correct solution.

__Cons__: If the data is not linearly separable, the perceptron will fial completely rather than give some approximate solution.

`note`: will have a Calculation problem.

---

## L7 - Multilayer Perceptron & Backpropagation

### Multilayer Perceptron

> modelling non-linearity via function composition

Nodes in ANN can have various activation functions

* Step function: $f(s) = \{^{1,  s \geq 0}_{0,  s < 0}$
* Sign functions: $f(s) = \{^{1,  s \geq 0}_{-1,  s < 0}$
* logistic function (Sigmoid): $f(s) = \dfrac {1}{1 + e^{-s}}$
* ReLu: $f(s) = \max (0, x)$
* Tanh:  $tanh(s) = 2sigmoid(2x) - 1 = \dfrac {e^x - e^{-x}}{e^x + e^{-x}}$

### ANN

 __Artifical neural network__(ANN) is a network of procesisng elements

Output using a activation function of a weighted sum of inputs

When using ANN, we need to

- design network topology
- adjust weights to given data

#### Feed-forward ANN

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8903753.png)

$g,h$ are activation functions.

$r_j = v_{0j}+\sum^m_{i=1}x_iv_{ij} = \sum^m_{i=0}x_iv_{ij}$ if add bias node $x_0 =1$

$u_j = g(r_j)$

$s_k = w_{0k} + \sum^p_{j=1}u_jw_{jk} = \sum^p_{j=0}u_jw_{jk}$ if add bias node $u_0 = 1$

$z_k = h(s_k)$

ANN is supervised learning

* can do univariate regression
* multivariate regression
* binary classification
* multivariate classification

__How to train__: Define the loss function and find parameters that minimise the loss on training data

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8906076.png)

$r_j = \sum^m_{i=0}x_iv_{ij}$

$z = s = \sum^p_{j=0}u_jw_j$

How many parameters does this ANN have? (Include bias nodes $x_0$ and $u_0$)

(m+1)p + (p+1) = __(m+2)p + 1__

#### Loss function for ANN training

Online training: use stochastic gradient descent with a batch size of one

For regression: $L = \dfrac {1}{2}(\hat f(\boldsymbol{x,\theta}) -y)^2 = \dfrac {1}{2}(z-y)^2$  (the constant is for convenience)

#### Stochastic gradient descent(batch size is 1) for ANN

* Choose initial guess $\boldsymbol{\theta}^{(0)}$, $k=0$ (Here $\boldsymbol{\theta}$ is a set of all weights from all layers)
* For i from i to T (epochs)
  * For j from 1 to N (training examples)
    * consider example {$\boldsymbol{x_j},y_j$}
    * Update: $\boldsymbol{\theta}^{(i++)} = \boldsymbol{\theta}^{(i)} - \eta \nabla L(\boldsymbol{\theta}^{(i)})$

$L = \dfrac {1}{2} (z_j - y_j)^2$  and need to compute partial derivatives $\dfrac {\partial L}{\partial v_{ij}}$ and $\dfrac {\partial L}{\partial w_{j}}$

#### BackPropagation

> Using chain rule

$\dfrac {\partial L}{\partial w_{j}} = \dfrac {\partial L}{\partial z} \dfrac {\partial z}{\partial s} \dfrac {\partial s}{\partial w_j}$

$\dfrac {\partial L}{\partial v_{ij}} = \dfrac {\partial L}{\partial z} \dfrac {\partial z}{\partial s}\dfrac {\partial s}{\partial u_{j}} \dfrac {\partial u_j}{\partial r_j}\dfrac {\partial r_j}{\partial v_{ij}}$

define $\delta \equiv \dfrac {\partial L}{\partial s} = \dfrac {\partial L}{\partial z} \dfrac {\partial z}{\partial s}$ and $\epsilon_j = \dfrac {\partial L}{\partial r_j} = \dfrac {\partial L}{\partial z} \dfrac {\partial z}{\partial s}\dfrac {\partial s}{\partial u_{j}} \dfrac {\partial u_j}{\partial r_j}$

$\because L = \dfrac {1}{2} (z - y)^2$ and $z = s$

$\therefore \delta = (z-y)$ 

$\because s = \sum^p_{j=0}u_jw_j$ and $u_j = h(r_j)$

$\therefore \epsilon_j = \delta w_j h'(r_j)$

$\dfrac {\partial s}{\partial w_j} = u_j$ and $\dfrac {\partial r_j}{\partial v_{ij}} = x_i$

Therefore, $\dfrac {\partial L}{\partial w_{j}} = (z-y)u_j$ and $\dfrac {\partial L}{\partial v_{ij}} = (z-y)w_jh'(r_j)x_i$ `backpropagation`

ANN is a flexible model, but the cons of it is over-parameterisation, hence tendency to __overfitting__

Starting weights are usually small random values distributed around zero

---

## L8 - Deep Learning CNN & Autoencoders

> Hidden layers viewed as feature space transformation

A hidden layer can be thought of as the transformaed feature space. e.g. $\boldsymbol{u} = \varphi(\boldsymbol{x})$

Parameters of such a transformation are learned from data

### Depth vs Width

a single infinitely wide layer used in theory gives a __universal approximator__

However, depth tends to give more accurate models

### CNN

> based on repeated application of small filters to patches of a 2D image or range of a 1D input

__Convolution__

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8909126.png)

























---
