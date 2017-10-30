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

> MLE is a frequentist principle that suggests that given a dataset, the "best" parameters to use are the ones that maximise the probability of the data

$\widehat {\theta }\left( x_{1},\ldots ,x_{n}\right) =argmax_{\theta \in \Theta }\prod ^{n}_{i=1}p_{\theta }\left( x_{i}\right)$  
__$\hat{\theta} = argmax_\theta P(X=x|\theta)$__ `MLE`

Methods that can be used to find MLE of parameters of a probabilistic model:

* gradient descent of the negative log likelihood
* closed form, such taht gradient of NLL = 0
* EM (expectation maximisation)

### [MAP](http://www.cnblogs.com/sylvanas2012/p/5058065.html)
$\hat{\theta}=P(\theta|X = x)$  
$= argmax_\theta \dfrac {P(X = x|\theta)P(\theta)}{P(X = x)}$  
__$= argmax_\theta P(X = x|\theta)P(\theta)$__ `MAP`

> MAP 和 MLE 区别在于MAP多了一个先验概率

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
>
> Method1:  Analytically — Ridge regression, Lasso
>
> Method2: Algorithmically

* avoid ill-conditioning
  * irrelevant features
    * Feature $\boldsymbol{X}_{.j}$ is irrelevant if $\boldsymbol{X}_{.j}$ is  linear combination of other columns
      * $\boldsymbol{X}_{.j} = \sum_{l \neq j} \alpha_l \boldsymbol{X}_{.l}$
    * The normal equations solution: $\boldsymbol{\hat{w}= (X'X)^{-1}X'y}$. __With irrelevant features, $\boldsymbol{X'X}$ has no inverse__
  * lack of data
* Introduce prior knowledge
* Constrain modelling (control the model complexity) — to deal with __underfitting__ and __overfitting__
  * Explicit model selection
    * Try different classes of models
    * use held out validation to select the model (split training data into training set and validation set)
  * Regularisation (Analytically, by adding a data-independent term to the objective function)
    * Augment the problem: $\boldsymbol{\theta} = argmin_{\boldsymbol{\theta} \in \Theta}(L(data, \boldsymbol{\theta}) + \lambda R(\boldsymbol(\theta))$
    * E.g. __ridge regression__ (L2): $||\boldsymbol{y - Xw}||^2_2 + \lambda ||\boldsymbol{w}||^2_2$
    * E.g. __Lasso__ (L1): $||\boldsymbol{y - Xw}||^2_2 + \lambda ||\boldsymbol{w}||_1$
    * Use held out validation/cross validation to choose $\lambda$
  * Regularisation (Algorithmically)
    * Early sopping in ANN
    * Weights sharing in CNN
    * Restricting tree depth in Random Forests

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

Problem: the probability needs to be between 0 and 1. Need to squash the function

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

$p(y|\boldsymbol{x}) = Normal(y|\boldsymbol{x'w}, \sigma^2)$

__Logistic regression__ assumes a Bernoulli distribution with parameter given by logistic transform of linear model:

$p(y|\boldsymbol{x}) = Bernolli \left(y|\theta(\boldsymbol{x}) = \dfrac {1}{1 - exp(-\boldsymbol{x'w})} \right)$

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

```
Why does the algorithm for training logistic regression involve gradient descent, while linear regression does not? 
There’s no closed form solution to the logistic optimisation problem, whereas linear can be solved for a stationary point (the global optimum) using linear alebra. 
```

### Basis Expansion

> Extending the utility of models via data transformation, can be applied for both regression and classification
>
> e.g. polynomial basis, RBF basis, also earlier layers of ANN can be viewed as transformation

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

simple model $\rightarrow$ underfitting, low variance, high bias

complex model  $\rightarrow$ overfitting, high variance, low bias 

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
* $L(s,y) = |s|​$ if both $s,y​$ have different sign
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

__Convolution__ on 1D

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8909126.png)

__Convolution__ on 2D

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8909353.png)

#### Components of CNN

__Convolutional layers__

* Complex input representations based on convolution operation
* Filter weights are learned from training data

__Max Pooling__ usually used to downsampling, [pooling](http://www.techweb.com.cn/network/system/2017-07-13/2556494.shtml)

* Re-scaling to smaller resolution, limits parameter explosion 

__FC__ and output layer or softmax

* Merges representations together, get final results

Convolution can also use as a __regulariser__

### Autoencoder

> An ANN training setup that can be used for unsupervised learning or efficient coding

Supervised learning:

* Univariate regression: predict $y$ from $\boldsymbol{x}$
* Multivariate regression: predict $\boldsymbol{y}$ from $\boldsymbol{x}$

Unsupervised learning: explore data $\boldsymbol{x_1,…,x_n}$

* No response variable

For each $\boldsymbol{x_i}$ set $\boldsymbol{y_i \equiv x_i}$, Train an ANN to predict $\boldsymbol{y_i}$ from $\boldsymbol{x_i}$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8910745.png)

### Dimensionality reduction

* Autoencoders can be used for compression and dimensionality reduction via a non-linear transformation 
* `TODO`

---

## L9 - Support Vector Machine (SVM) hard margin

### Maximum Margin classifier (SVM)

> For binary classification problem

#### linear hard margin SVM

> SVM is a linear binary classifier

Predict class A if $s \geq 0$ and Predict class B if $s < 0$ where $s = b + \sum^m_{i=1}x_iw_i$

the different SVMs from preceptron is that the way the parameters are learned. They have different loss. If any good boundaries for the dataset is linearly separable, the perceptron loss is zero.

#### choosing separation boundary (SVM)

SVM aiming for the safest boundary.  To find the separation boundary that __maximises the margin__ between the classes.

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8912865.png)

The points on margin boundaries from either side are called __support vectors__

Distance is $||\boldsymbol{r}||=\pm \dfrac {\boldsymbol{w'x} + b}{\boldsymbol{||w||}}$

Therefore, the distance from the i-th point to a  perfect boundary can be encoded as $||\boldsymbol{r_i}||= \dfrac {y_i(\boldsymbol{w'x_i} + b)}{\boldsymbol{||w||}}$

Therefore, the SVMs aim to maximise $\left( min_{i=1,…,n} \dfrac {y_i(\boldsymbol{w'x_i} + b)}{\boldsymbol{||w||}}\right)$

SVM has non-unique representation, because the same boundary can be expressed with infinitely many parameter combinations.

__Resolving ambiguity__: rescale parameters such that $\dfrac {y_{i^*}(\boldsymbol{w'x_{i^*}} + b)}{\boldsymbol{||w||}} = \dfrac {1}{\boldsymbol{||w||}}$, $i^*$ is the distance to the closest point

Therefore, we now have that SVMs aim to find $argmin_\boldsymbol{w}||\boldsymbol{w}||$ , s.t. $y_i(\boldsymbol{w'x_i} + b) \geq 1$ for $i = 1,…,n$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8915037.png)

### SVM Objective as Regularised Loss

$l_\infty = \{ ^{0, 1- y_i(\boldsymbol{w'x_i} + b) \leq 0}_{\infty, 1 - y_i(\boldsymbol{w'x_i} + b) > 0}$

$l_\infty = \{ ^{0, 1- y(\boldsymbol{w'x} + b) \leq 0}_{\infty, otherwise}$

`question`

---

## L10 - SVM soft margin

Real data is unlikely to be linearly separabel, SVMs offer 3 approaches to address this problem:

* Still use hard margin SVM, but transform the data
* Relax the constraints
* Combination of 1 and 2

### Soft Margin SVMs to deal with Non-linearity data

> we relax the constraints to allow points to be inside the margin or even on the wrong side of the boundary

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8931234.png)

#### hinge loss: soft margin SVM loss

$l_h = \{ ^{0, 1- y(\boldsymbol{w'x} + b) \leq 0}_{1 - y(\boldsymbol{w'x} + b), otherwise}$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8929998.png)

With ridge regression, soft margin SVM objective can be defined as

$argmin_{\boldsymbol{w}} \left( \sum^n_{i=1}l_h(\boldsymbol{x_i},y_i,\boldsymbol{w}) + \lambda||\boldsymbol{w}||^2 \right)$

#### Re-formulating soft margin objective

* soft margin SVM loss (hinge loss)
  * $l_h = \max (0,1 - y_i(\boldsymbol{w'x_i} + b))$
* define slack variables as an upper bound on loss
  * $\xi_i \geq l_h = \max (0,1 - y_i(\boldsymbol{w'x_i} + b))$
* Re-write the soft margin SVM objective as:
  * $argmin_{\boldsymbol{w},\xi} \left( \dfrac {1}{2} ||\boldsymbol{w}||^2 + C\sum^n_{i=1} \xi_i \right)$
  * s.t. $y_i(\boldsymbol{w'x_i} + b) \geq 1 - \xi_i$ for $i = 1,…,n$  and  $\xi_i \geq 0$ for $i = 1,…n$

Therefore, for __hard margin SVM__ objective:

$argmin_\boldsymbol{w} \dfrac {1}{2} ||\boldsymbol{w}||^2$  `hard margin SVM`

s.t. $y_i(\boldsymbol{w'x_i} + b) \geq 1$ for $i = 1,…,n$

And for __soft margin SVM__ objective:

$argmin_{\boldsymbol{w},\xi} \left( \dfrac {1}{2} ||\boldsymbol{w}||^2 + C\sum^n_{i=1} \xi_i \right)$  `soft margin SVM`

s.t. $y_i(\boldsymbol{w'x_i} + b) \geq 1 - \xi_i$ for $i = 1,…,n$  and  $\xi_i \geq 0$ for $i = 1,…n$

the constraints are relaxed ("softened") by allowing violations by $\xi_i$. 

### SVM training preliminaries

> is to solving the corresponding optimisation problem

#### Constrained optimisation

Method of Lagrange/KKT multipliers

* Transform the original problem into an unconstrained optimisation problem
* Analyse/relate necessary and sufficient conditions for solutions for both problems

KKT objective:

$L(\boldsymbol{x, \lambda, v}) = f(\boldsymbol{x}) + \sum^n_{i=1}\lambda_ig_i(\boldsymbol{x}) + \sum^m_{j=1}v_jh_j(\boldsymbol{x})$

#### Lagrangian for hard margin SVM

Define the lagrangian/KKT objective for hard margin SVM first

$L_{KKT}(\boldsymbol{w}, b , \boldsymbol{\lambda}) = \dfrac {1}{2} ||\boldsymbol{w}||^2 - \sum^n_{i=1} \lambda_i(y_i(\boldsymbol{w'x_i} + b) - 1)$

​			(primal objective)			(constraints)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8932266.png)

Given these, in order to solve the primal problem, we pose a new optimisation problem, called __Lagrangian dual problem__ also called __(quadratic optimisation problem)__ `question`

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8932391.png)

For __hard margin SVM__ training: finding $\boldsymbol{\lambda}$ that solve above formula

__Marking predictions__: classify new instance x based on the sign of 

$s = b^* + \sum^n_{i=1}\lambda_i^*y_i\boldsymbol{x_i'x}$

$b^*$ can be found in $y_j(b^* + \sum^n_{i=1}\lambda_i^*y_i\boldsymbol{x_i'x}) = 1$

The difference from __soft__ to __hard__ margin SVM is that the constaints in __soft__ is

s.t. $C \geq \lambda_i \geq 0$ which in __hard__ is s.t. $\lambda_i \geq 0$ `difference between soft and hard`

p.s. $1- y_i(\boldsymbol{w'x_i} + b) \leq 0$ means that $\boldsymbol{x_i}$ is outside the margin

THerefore, for KKT, $\lambda_i^*$ must =0 when points outside the margin and the points with non-zero $\lambda s$ are __support vectors__

---

## L11 - Kernel Methods

### Kernel trick

> efficient computation of a dot product in transformed feature space
>
> Pros: The kernel trick is replacing explicit expression of a dot product with a function called kernel. The kernel equals to dot product in some feature space, but does not evaluate it explicitly. The advantage is faster computation, and the ability to implicitly use intractably large (even infinite) feature spaces.

P.s. 升维和降维都可能将本来不能二分的问题进行二分

__Kernel__ is a function that can be expressed as _dot product_ in some feature space $K(\boldsymbol{u,v})=\varphi(\boldsymbol{u})' \varphi(\boldsymbol{v})$

Choosing a kernel implies some transformation $\varphi (\boldsymbol{x})$, the pros of using kernels is that __we don't need to actually compute components of $\varphi (\boldsymbol{x})$__. This is beneficial when the transformed space is multidimensional. In addition, it makes it possible to transform the data into an infinite- dimensional space.

#### Kernels

* Linear kernel
  * $K(\boldsymbol{u,v}) = \boldsymbol{u'v}$
* Polynomial kernel
  * $K(\boldsymbol{u,v}) = (\boldsymbol{u'v} + c)^d$
* Radial basis function (rbf) kernel (also called Gaussian kernel)
  * $K(\boldsymbol{u,v}) = exp(-\gamma ||\boldsymbol{u-v}||^2)$

#### Kernel as a similarity measure

we can extend kernel methods to objects that are not vectors

#### Identifying new kernels

__using identities__ : Let $K_1(\boldsymbol{u,v})$, $K_2(\boldsymbol{u,v})$ be kernels and $c>0$ an be constant, following is also kernel

* $K(\boldsymbol{u,v}) = K_1(\boldsymbol{u,v}) + K_2(\boldsymbol{u,v})$
* $K(\boldsymbol{u,v}) = c K_1(\boldsymbol{u,v})$
* $K(\boldsymbol{u,v}) = f(\boldsymbol{u}) K_1(\boldsymbol{u,v}) f(\boldsymbol{v})$

Or using __Mercer's theorem__

* Construct nxn matrix of pairwise values $K(\boldsymbol{x_i,x_j})$
* $K(\boldsymbol{x_i,x_j})$ is kernel if this matrix is positive-semidefinite (半正定). `question`

### Modular Learning

* choose learning method (model)
* choose feature space mapping

---

## L12 - Ensemble methods & Interim Revision

 ### Combining models ([Ensemble methods](https://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning)) (集成学习)

__model combination__ (ensemble learning) constructs a set of base models (learners) from a given set of training data and aggregates the ouputs into a single meta-model

* Classification via (weighted) majority vote
* Regression via (weighted) averaging
* More generally: meta-model = f(base models)

Because Test error = (bias)^2 + variance + irreducible error, Averaging k independent and identically distributed predictions reduces variance: $Var[\hat f_{avg}] = \dfrac {1}{k}Var[\hat f]$

> All three are so-called "meta-algorithms": approaches to combine several machine learning techniques into one predictive model in order to decrease the variance (bagging), bias (boosting) or improving the predictive force (stacking alias ensemble)

#### bagging (bootstrap aggregating)

> construct "novel" datasets via sampling with replacement
>
> (stands for Bootstrap Aggregation) is the way decrease the variance of your prediction by generating additional data for training from your original dataset using combinations with repetitions to produce multisets of the same cardinality/size as your original data. By increasing the size of your training set you can't improve the model predictive force, but just decrease the variance, narrowly tuning the prediction to expected outcome.

* Generate k datasets, each size n sampled from training data with replacement
* Build base classifier on each constructed dataset
* Combine predictions via voting/averaging

Example:

```
Original training dataset: {0,1,2,3}
Bootstrap samples:
{1,2,2,3} out of sample 0
{0,0,3,3} out of sample 1, 2
```

#### decision trees

Model complexity is defined by the depth of the tree

Deep trees: high variance, low bias

shallow trees: low variance, high bias

#### bagging example: Random forest

> just bagged trees

Algorithm

* init forest as empty
* For $c = 1…k$
  * Create new bootstrap sample of training data
  * select random subset of l of the m features
  * Train decision tree on bootstrap sample using the l features
  * Add tree to forest
* Making predictions via majority vote or averaging

Reflections: simple method based on sampling and voting, possibility to parallelies computation of individual base classifiers, highly effective over noisy datasets, performance is better, can improves unstable calssifiers by reducing variance

### Boosting

> focus attention of base classifiers on examples "hard to classify"
>
> is a two-step approach, where one first uses subsets of the original data to produce a series of averagely performing models and then "boosts" their performance by combining them together using a particular cost function (=majority vote). Unlike bagging, in the classical boosting the subset creation is not random and depends upon the performance of the previous models: every new subsets contains the elements that were (likely to be) misclassified by previous models.

Method: iteratively change the distribution on examples to reflect performance of classifier on the previous iteration

Example:

```
original training dataset: {0,1,2,3,4,5,6,7,8,9}
Boosting samples:
iteration 1: {7,2,6,7,5,4,8,8,1,0} 	1 2
assume example 2 was misclassified
iteration 2: {1,3,8,2,3,5,2,0,1,9}	2 2s
assume example 2 was still misclassified
iteration 3: {2,9,2,2,7,9,3,2,1,0}	4 2s
```

#### AdaBoost

* Init example distribution $P_1(i) = 1/n, i = 1,…,n$
* For $c = 1…k$
  * Train base classifier $A_c$ on sample with replacement from $P_c$
  * set confidence $\alpha_c = \dfrac {1}{2} ln \dfrac {1 - \epsilon _c}{\epsilon _c}$ for classifier's error rate $\epsilon _c$
  * Update example distribution to be normalised of:
    * $P_{c+1}(i) \propto P_c(i) \times \{ ^{exp(-\alpha _c), A_c(i) = y_i} _{exp(\alpha _c), otherwise}$
* Classify as majority vote weighted by confidences $argmax_y \sum^k_{c=1} \alpha _t \delta (A_c(\boldsymbol{x}) = y)$

Reflections: boosting based on iterative sampling and weighted voting, more computationally expensive than bagging. In practical applications, boosting can overfit

| Bagging                      | Boosting                     |
| ---------------------------- | ---------------------------- |
| Parallel sampling            | Iterative sampling           |
| __Minimise variance__        | __Target "hard" instances__  |
| Simple voting                | Weighted voting              |
| Classification or regression | Classification or regression |
| Not prone to overfitting     | Prone to overfitting         |

### Stacking

> "smooth" errors over a range of algorithms with different biases
>
>  is a similar to boosting: you also apply several models to your original data. The difference here is, however, that you don't have just an empirical formula for your weight function, rather you introduce a meta-level and use another model/approach to estimate the input together with outputs of every model to estimate the weights or, in other words, to determine what models perform well and what badly given these input data.

Method: train a meta-model over the outputs of the base learners

* train base- and meta-learners using cross-validation
* Simple meta-classifier: logistic regression

`question` : difference between bagging and boosting

Reflections: Mathematically simple but computationally expansive method.

### Supervised Learning Interim Summary

[Supervised Learning](#frequentist supervised learning)

#### supervised learning methods

* [Linear Regression](#Linear Regression)
  * Model: $y =  \boldsymbol{x'w} + \epsilon$, where $\epsilon \sim \mathbb{N} (0,\sigma ^2)$
  * Loss function: [Squared loss](#Method of least squares)
  * Optimisation: [Analytic solution](#Solving optimisation problems)
  * can also be optimised iteratively
* [Logistic Regression](#Logistic regression model)
  * Model: $P(y|\boldsymbol{x}) = Bernoulli \left(y|\theta(\boldsymbol{x}) = \dfrac {1}{1 + exp(-\boldsymbol{x'w})} \right)$
  * Loss function: [Cross-entropy](#Cross entropy) (aka log loss)
  * Optimisation: [stochastic gradient descent](#optimisation for logistic regression)
* [Perceptron](#Preceptron model)
  * Model: label is baed on sign of $s = \sum^m_{i=0}x_iw_i$ or $w_0 + \boldsymbol{w'x}$
  * Loss function: [Perceptron loss](#loss function for perceptron)
  * Optimisation: [Stochastic gradient descent](#Perceptron training algorithm)
* [ANN](#ANN)
  * Model: defined by network topology
  * Loss function: varies ,can use cross-entropy , [loss function](#Loss function for ANN training)
  * [Optimisation](#Stochastic gradient descent(batch size is 1) for ANN): Variations of gradient descent, Ada
  * Notes: [backpropagation](#BackPropagation) used to compute partial derivatives.
  * [CNN](#CNN)
* [SVM](#L9 - Support Vector Machine (SVM) hard margin)
  * Model: label is based on sign of $s = b + \sum^m_{i=1}x_iw_i = b + \boldsymbol{w'x}$
  * Loss funtion: [hard margin SMV loss](#SVM Objective as Regularised Loss); [hinge loss](#hinge loss: soft margin SVM loss)
  * Optimisation: [Quadratic Programming](#SVM training preliminaries)
  * Notes: Specialised optimisation algorithms `question`
* [Random Forest](#bagging example: Random forest)
  * Model: average of decision trees
  * Loss function: Cross-entropy; or squared loss
  * Optimisation: Greed growth of each tree
  * Notes: an example of model averaging


---

## L13 - Clustering & Gaussian Mixture model (GMM)

### Unsupervised Learning

> concerns with learning the structure of the data in the absence of labels

* Clustering
* Dimensionality reduction
* Learning parameters of probabilistic models

### Clustering

> automatic grouping of objects such that the objects within each group (cluster) are more similar to each other than objects from different groups

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8993441.png)

#### Measuring dissimilarity

__Euclidean distance__ : 

$d_{ij} = ||\boldsymbol{x_i - x_j}|| = \sqrt {\sum^m_{l=1} \left( \boldsymbol{(x_i)_l - (x_j)_l} \right)^2}$

#### K-means clustering

* Init: choose k cluster centroids randomly
* Update
  * Assign points to the nearest centroid
  * compute centroids under the current assignment
* Termination: if no change then stop
* Goto Step 2: Update

#### Determine number of clusters

* Cross-validation-like strategies for determining k
* Try several possible k and look at the trend
* Information-theoretic results

#### Kink method and gap statistics

`question`

Manual inspection of minimised within cluster variation as function of k

### Guassian Mixture model (GMM)

> a probabilistic view of clustering
>
> GMM clustering is a generalisation of k-means
>
> example application of expectation Maximisation (EM) algorithm

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8994751.png)

#### Normal (Gaussian) distribution

For 1D Gaussian: $\mathbb{N}(x|\mu, \sigma) \equiv \dfrac {1}{\sqrt {2\pi\sigma ^2}}exp \left( - \dfrac {(x-\mu)^2}{2 \sigma ^2} \right)$

For m-D Gaussian: $\mathbb{N}(\boldsymbol{x|\mu, \Sigma}) \equiv (2\pi)^{-\dfrac {m}{2}} (det\boldsymbol{\Sigma})^{- \dfrac {1}{2}} exp \left(- \dfrac {1}{2} (\boldsymbol{(x - \mu)' \Sigma^{-1}(x - \mu)}) \right)$

$\boldsymbol{\Sigma}$ is a symmetric $m \times m$ matrix that is assumed to be positive definite

$det\boldsymbol{\Sigma}$ denotes matrix determinant

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8995908.png)

#### GMM

__Gaussian mixture distribution__ (for one data point):

$p(\boldsymbol{x}) \equiv \sum^k_{c=1} w_c \mathbb{N}(\boldsymbol{x| \mu_c, \Sigma_c})$

Here $w_c \geq 0$ and $\sum^k_{c=1}w_c = 1$ 

$w_c$ is a probability distribution over components, all $w_c$ add up to 1

Parameters of model are $w_c, \boldsymbol{\mu_c, \Sigma_c}, c = 1,…k$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-8996673.png)

`question`

```
example for parameter: 
Consider a GMM with 5 components for 3D data. How many independent parameters does this model have?
```

For 3D data and 5 components: $w = 4, \boldsymbol{\mu} = 3, \boldsymbol{\Sigma} = 6$  Therefore, the answer is $6 \times 5 + 3 \times 5 + 4$

__MLE__ is used for Clustering GMM as an optimisation problem

#### Fitting a GMM model to data (optimisation)

Assuming that data points are independent, our aim is to find $w_c, \boldsymbol{\mu_c, \Sigma_c}, c = 1,…k$ that maximise

$p(\boldsymbol{x_1,…,x_n}) = \prod^n_{i=1} \sum^k_{c=1} w_c \mathbb{N}(\boldsymbol{x_i| \mu_c, \Sigma_c})$

After using log trick

$\log p(\boldsymbol{x_1,…,x_n}) = \sum^n_{i=1} \log \left( \sum^k_{c=1} w_c \mathbb{N}(\boldsymbol{x_i| \mu_c, \Sigma_c} \right)$

The log cannot be pushed inside the sum, so taking the derivative of this expression is challenging, need use [__EM__](#Estimating Parameters of GMM) to solve this optimisation. 

#### Using Expectation Maximisation (EM)

> is a common way to find parameters of GMM

__EM__ is an _algorithm_ to solve the problem posed by MLE

---

## L14 - Expectation Maximisation Algorithm (EM)

[EM](http://blog.csdn.net/zouxy09/article/details/8537620) 期望最大算法

> key idea for EM is to introduce latent variables
>
> In GMM, if we let $z_i$ denote true cluster membership for each point $\boldsymbol{x_i}$. computing the likelihood with known values $\boldsymbol{z}$ is simplified

### Jensen's Inequality

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9008967.png)

Compares effect of averaging before and after applying a convex function:$f(Average(\boldsymbol{x})) \leq Average(f(\boldsymbol{x}))$

Therefore, If $\boldsymbol{X}$ random variable, f is a convex function: $f(\mathbb{E}[\boldsymbol{X}]) \leq \mathbb{E}[f(\boldsymbol{X})]$

Because we want to maximise $\log p(\boldsymbol{X|\theta})$ . We don't know $\boldsymbol{Z}$ , but consider an arbitary non-zero distribution $p(\boldsymbol{Z})$ 
$$
\log p(\boldsymbol{X|\theta}) = log \sum_z p(\boldsymbol{X,Z|\theta}) \\
= \log \sum_z \left( p(\boldsymbol{X,Z|\theta}) \dfrac {p(\boldsymbol{Z})}{p(\boldsymbol{Z})} \right) \\
= \log \sum_z \left( p(\boldsymbol{Z}) \dfrac {p(\boldsymbol{X,Z|\theta})}{p(\boldsymbol{Z})} \right) \\
= \log \mathbb{E}_{\boldsymbol{Z}} \left[ \dfrac {p(\boldsymbol{X,Z|\theta})}{p(\boldsymbol{Z})} \right] \\
\geq \mathbb{E}_{\boldsymbol{Z}} \left[ \log  \dfrac {p(\boldsymbol{X,Z|\theta})}{p(\boldsymbol{Z})} \right] \\
= \mathbb{E}_{\boldsymbol{Z}}[\log p(\boldsymbol{X,Z|\theta})] - \mathbb{E}_{\boldsymbol{Z}}[\log p(\boldsymbol{Z})]
$$
Then maximising the lower bound:

$\log p(\boldsymbol{X|\theta}) \geq \mathbb{E}_{\boldsymbol{Z}}[\log p(\boldsymbol{X,Z|\theta})] - \mathbb{E}_{\boldsymbol{Z}}[\log p(\boldsymbol{Z})]$    `EM`

$\mathbb{E}_{\boldsymbol{Z}}[\log p(\boldsymbol{X,Z|\theta})] - \mathbb{E}_{\boldsymbol{Z}}[\log p(\boldsymbol{Z})] \equiv G(\boldsymbol{\theta},p(\boldsymbol{Z}))$    `lower bound`

The right hand side (RHS) is a lower bound on the original log likelihood. We want to maximise the RHS as a function of "variables" $\boldsymbol{\theta}$ and $p(\boldsymbol{Z})$ .

It is hard to optimise with respect to both at the same time, so __EM__ resorts to an iterative procedure

__EM__ using coordinate descent:

* Fix $\boldsymbol{\theta}$ and optimise the lower bound for $p(\boldsymbol{Z})$ 
* Fix $p(\boldsymbol{Z})$ and optimise for $\boldsymbol{\theta}$

For any point $\boldsymbol{\theta}^*$, it can be shown that setting $p(\boldsymbol{Z}) = p(\boldsymbol{Z|X, \theta^*})$ makes the lower bound tight

For any $p(\boldsymbol{Z})$ , the second term does not depend on $\boldsymbol{\theta}$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9010554.png)

#### EM as iterative optimisation

* Init: choose initial values of $\boldsymbol{\theta}^{(1)}$
* Update:
  * __E-step__: compute $Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)}) \equiv \mathbb{E}_{\boldsymbol{Z|X, \theta^{(t)}}}[\log p(\boldsymbol{X,Z|\theta})]$    根据参数初始值或上一次迭代的模型参数来计算出隐性变量的后验概率，其实就是隐性变量的期望。作为隐藏变量的现估计值
  * __M-step__: $ \boldsymbol{\theta}^{(t+1)} = argmax_\theta Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})$.  将似然函数最大化以获得新的参数值
* Termination: if no change then stop
* Go to step 2

> The resulting estimate can be only a local maximum

#### Setting a tight lower bound

$$
\log p(\boldsymbol{X|\theta}) \geq \mathbb{E}_{\boldsymbol{Z}} \left[ \log  \dfrac {p(\boldsymbol{X,Z|\theta})}{p(\boldsymbol{Z})} \right] \\
=  \mathbb{E}_{\boldsymbol{Z}} \left[ \log  \dfrac {p(\boldsymbol{Z|X,\theta})p(\boldsymbol{X|\theta})}{p(\boldsymbol{Z})} \right] \\
= \mathbb{E}_{\boldsymbol{Z}} \left[ \log  \dfrac {p(\boldsymbol{Z|X,\theta})}{p(\boldsymbol{Z})} \right] + \log p(\boldsymbol{X|\theta}) \\
\therefore \log p(\boldsymbol{X|\theta}) \geq \mathbb{E}_{\boldsymbol{Z}} \left[ \log  \dfrac {p(\boldsymbol{Z|X,\theta})}{p(\boldsymbol{Z})} \right] + \log p(\boldsymbol{X|\theta}) \\
\because \mathbb{E}_{\boldsymbol{Z}} \left[ \log  \dfrac {p(\boldsymbol{Z|X,\theta})}{p(\boldsymbol{Z})} \right] \leq 0
$$

And if $p(\boldsymbol{Z}) = p(\boldsymbol{Z|X,\theta})$ , then first term of RHS = 0

For any $\boldsymbol{\theta}^*$ , setting $p(\boldsymbol{Z}) = p(\boldsymbol{Z|X,\theta^*})$ maximises the lower bound on $\log p(\boldsymbol{X|\theta^*})$ and makes it tight

### Estimating Parameters of GMM

> using EM algorithm

* Let $z_1, …,z_n$ denote true origins of the corresponding points $\boldsymbol{x_1,…,x_n}$ . Each $z_i$ is a discrete variable that takes values in $1,..,k$ where k is a number of clusters
* Them ,if we knew $\boldsymbol{z}$ , the complete data log likelihood is 
  * $\log p(\boldsymbol{x_1,…,x_n, z}) = \sum^n_{i=1} \log \left( \sum^k_{c=1} w_{zi} \mathbb{N}(\boldsymbol{x_i| \mu_{zi}, \Sigma_{zi}} \right)$

EM algorithm handles this uncertainty replacing $\log p(\boldsymbol{X|\theta})$ with expectation $\mathbb{E}_{\boldsymbol{Z|X, \theta^{(t)}}}[\log p(\boldsymbol{X,Z|\theta})]$ . This in turn requires the distribution of $p(\boldsymbol{Z|X, \theta^{(t)}})$ given current parameter estimates

Assuming that $z_i$ are pairwise independent, we nee to define $P(z_i = c| \boldsymbol{x_i,\theta ^{(t)}})$

We can use  $P(z_i = c| \boldsymbol{x_i,\theta ^{(t)}}) = \dfrac {w_c \mathbb{N}(\boldsymbol{x_i| \mu_c, \Sigma_c})}{\sum^k_{l=1} w_l \mathbb{N}(\boldsymbol{x_i| \mu_l, \Sigma_l})}$ `reponsibility`

This probability is called __responsibility__ that cluster c takes for data point i

$r_{ic} \equiv P(z_i = c| \boldsymbol{x_i,\theta ^{(t)}})$

#### Expectation step for GMM 

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9014018.png)

#### Maximisation step for GMM

> In the maximisation step, take partial derivatives $Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})$, with respect to each of the parameters and set the derivatives to zero to obtain new parameter estimates

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9014192.png)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9014238.png)

Also, __k-means__ algorithm is a EM algorithm for the restricted GMM model.

---

## L15 - Dimensionality Reduction

### Dimensionality reduction

> DR refers to representing the data using a smaller number of variables (dimensions) while preserving the "interesting" structure of the data

Several purpose

* visualisation (e.g. mao on 2D or 3D)
* Computational efficiency
* Data compression

Although DR in general results in loss of some information, it need to ensure that most of the "interesting" information (signal) is preserved.

### Principal Component Analysis (PCA)

> Finding a rotation of data that minimises covariance between variables

在降维的同时保持数据之间最大差异性

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9015062.png)

The "new coordinate system" is a set of vectors $\boldsymbol{p_1,…,p_m}$, where each $||\boldsymbol{p_i}|| = 1$

Therefore, the corresponding $i^{th}$ coordinate for all point after the transformation is $\boldsymbol{X'p_i}$

The variance along this axis is $\dfrac {1}{n-1}(\boldsymbol{X'p_i})'(\boldsymbol{X'p_i}) = \dfrac {1}{n-1}(\boldsymbol{p_i'X} \boldsymbol{X'p_i}) = \boldsymbol{p_i' \Sigma_x p_i}$

here $\boldsymbol{\Sigma_x}$ is the covariance matrix (协方差矩阵) of the original data $\boldsymbol{\Sigma_x} \equiv \dfrac {1}{n-1} \boldsymbol{XX'}$ 

__PCA__ aims to find $\boldsymbol{p_i}$ that maximises $\boldsymbol{p_i' \Sigma_x p_i}$

#### solving the optimisation

Using Lagrange to transform the original problem into an unconstrained optimisation problem. Introduce a Lagrange multiplier $\lambda_1$ , and set derivatives of the Lagrangian to zero
$$
L = \boldsymbol{p_1' \Sigma_X p_1} - \lambda_1 (\boldsymbol{p_1'p_1} - 1) \\
\dfrac {\partial L}{\partial \boldsymbol{p_1}} = 2 \boldsymbol{\Sigma_x p_1} -2 \lambda_1\boldsymbol{p_1} = 0 \\
\boldsymbol{\Sigma_Xp_1} = \lambda_1 \boldsymbol{p_1}
$$
eigenvector 特征向量， 得到的新矢量仍然与原来的 保持在同一条直线上，但其长度或方向也许会改变 

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9017292.png)

> Given a square matrix A , a column vector e is called an eigenvector if Ae = $\lambda e$ . Here $\lambda$ is the corresponding eigenvalue 

`question`

Therefore, in order to maximise variance along the first principal axis, the axis should be chosen such that $\boldsymbol{\Sigma_Xp_1} = \lambda_1 \boldsymbol{p_1}$

$\boldsymbol{p_1}$ has to be an __eigenvector__ of centered data covariance matrix $\boldsymbol{\Sigma_X}$

Becasue $\lambda_1 = \boldsymbol{p_1' \Sigma_x p_1}$  , thus we need to choose $\boldsymbol{p_1}$ that corresponds to the largest eigenvalue of centered data covariance matrix $\boldsymbol{\Sigma_X}$

Also, $||\boldsymbol{p_i}|| = 1$ is a important constraint

> Lemma: a real symmetric m x m matrix has m real eigenvalues and the corresponding eigenvectors are orthogonal

Step:

* sort eigenvalues from largest to smallest
* Set $\boldsymbol{p_1,…,p_m}$ as corresponding eigenvectors
* Project data $\boldsymbol{X}$ onto these new axes to get coordinates of the transformed data
* Keep only the first s coordinates to reduce dimensionality

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9073809.png)

### Multidimensional Scaling

MDS

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9073941.png)

Two "parameters" of MDS approach

* how to measure dissimilarity
* how to measure preservation of dissimilarity

One way is using Euclidean distance  $d(\boldsymbol{z_i,z_j}) = ||\boldsymbol{z_i-z_j}||$

The preservation can be measured suing a function such as

$S(\boldsymbol{z_1,…,z_n}) = \dfrac {\sum_{i,j}(d(\boldsymbol{x_i,x_j}) - d(\boldsymbol{z_i,z_j}))^2}{\sum_{i,j} d(\boldsymbol{z_i,z_j})^2}$   `stress function`

The aim of such MDS is to find $\boldsymbol{z_1,…,z_n}$ that minimise $S_M(\boldsymbol{z_1,…z_n})$, can using gradient descent to deal with it 

```
Suppose that there are several clusters in high dimensional data, points within clusters are close to each other, points from different clusters are far away. MDS attemps to preserve this distance structure, so that clusters are preserved in low dimensional map.
```

### Data representation

Switching between data representations

* Coordinates for each point (High dimensional coordinates — to DO dimensionality reduction using MDS)
  * Compute distances
* Matrix of pairwise distances
  * Do MDS
* (Reconstructed) coordinates for each point (or Approximate low dimensional coordinates)

Example:

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9075270.png)

---

## L16 - Manifold Learning

流形可以简单的理解为空间 [Manifold](http://blog.csdn.net/xuexiyanjiusheng/article/details/46928771)

> 所谓 Machine Learning 里的 Learning ，就是在建立一个模型之后，通过给定数据来求解模型参数。而 Manifold Learning 就是在模型里包含了对数据的流形假设

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9075839.png)

> Isomap 将这个数据集从 4096 维映射到 3 维空间中，并显示了其中 2 维的结果

__K-means__ can find spherical clusters and __GMM__ can find elliptical clusters. But they will struggle in the case of handel the manifold problem.

### Geodesic Distances and Isomap

> A non-linear dimensionality reduction algorithm that preserves locality information using geodesic distances 

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9077814.png)

We are more interested in preserving distances along the manifold (geodesic distances)

### Geodesic distances

Geodesic distances can be approximated using a graph in which vertices represent data points

__Similarity graph__

* define some local radius $\epsilon$ . Connect vertices $i$ and $j$ with an edge if $d(i,j) \leq \epsilon$
* define nearest neighbor threshold $k$. Connect vertices $i$ and $j$. if $i$ is among the k nearest neighbors of $j$ or $j$ is among the $k$ nearest neighbors of $i$ , set weight for each edge to $d(i,j)$

Given the similarity graph, compute shortest paths between each pair of points

Set geodesic distance between vertices $i$ and $j$ to the length (sum of weights) of the shorest path between them.

Define a new similarity matrix based on geodesic distances

### Isomap

* Construct the similarity graph
* Compute shortest paths
* Geodesic distances are the lengths of the shortest paths
* Construct similarity matrix using geodesic distanes
* Apply MDS

### Spectral Clustering

> An spectral graph theory approach to non-linear dimensionality reduction

In contrast to Isomap, spectral clustering uses a different non-linear mapping technique called Laplacian eigenmap

* Construct __similarity graph__ , use the corresponding adjacency matrix as a new similarity matrix
  * Unlike Isomap, the adjacency matrix is used "as is", shortest paths are not used
* Map data to a lower dimensional space using __Laplacian eigenmaps__ on the adjacency matrix
* Apply __k-means__ clustering to the mapped points

#### Graph Laplacian

`TODO`

#### Laplacian eigenmaps `Todo` (拉普拉斯特征映射) 

> is a non-linear dimensionality reduction method

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9079989.png)

`Todo`

---

## L17 - Bayesian inference & Bayesian regression `question` 

### Uncertainty

> From small training sets, we rarely have complete confidence in any models learned.

贝叶斯回归比较适合数据集少的情况，并且可以防止过拟合

Single prediction is of limited use due to uncertainty

* single number uninformative - may be wildly off
* might want to formulate decision from prediction

### The Bayesian View

> Retain and model all unknows (e.g., uncertainty over parameters) and use this information when making inferences.

### Frequentist vs Bayesian divide

__Frequentist__

learning using point estimates, regularisation, p-values ...

__Bayesian__

maintain uncertainty, marginalise (sum) out unknowns during inference

### Bayesian Regression

> Application of Bayesian inference to linear regression, using Normal prior over w

[Bayesian Regression](https://www.zhihu.com/question/22007264)

Recall probabilistic formulation of linear regression

$y \sim Normal(\boldsymbol{x'w,} \sigma^2)$ 

Motivated by Bayes rule

$ \boldsymbol{w} \sim Normal (\boldsymbol(0, \gamma^2 \boldsymbol{I}_D))$ 

$p(\boldsymbol{w|X,y}) = \dfrac {p(\boldsymbol{y|X,w})p(\boldsymbol{w})}{p(\boldsymbol{y|X})}$

$max_\boldsymbol{w} p(\boldsymbol{w|X,y}) = max_{\boldsymbol{w}} p(\boldsymbol{y|X,w})p(\boldsymbol{w})$

Rewind one step, consider full posterior

$p(\boldsymbol{w|X,y}, \sigma^2) = \dfrac {p(\boldsymbol{y|X,w}, \sigma^2)p(\boldsymbol{w})}{p(\boldsymbol{y|X})}$

$ = \dfrac {p(\boldsymbol{y|X,w}, \sigma^2)p(\boldsymbol{w})}{\int p(\boldsymbol{y|X,w}, \sigma^2)p(\boldsymbol{w})d\boldsymbol{w}}$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9098674.png)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9098724.png)

#### Bayesian Linear Regression example

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9098802.png)

__Sequential Bayesian Updating__

can repeat from step 2 when we see more and more data

* Start from prior
* See new labelled datapoint
* Compute posterior $p(\boldsymbol{w|X, y}, \sigma^2)$
* The posterior now takes rols of prior
  * repeat from step 2

> Uncertainty not captured by MEL and MAP

---

## L18 - Bayesian classification & Model selection

`question` `TODO` 

### Stages of Training

1. Decide on model formulation & prior
2. Compute posterior over parameters, $p(\boldsymbol{w|X,y})$

| MAP                               | approx.Bayes                             | exact Bayes                              |
| --------------------------------- | ---------------------------------------- | ---------------------------------------- |
| 3. Find mode for w                | 3. Sample many w                         | 3. Use all w to make expected predicton on test |
| 4. Use to make prediction on test | 4. Use to make ensemble average prediction on test |                                          |

### Bayesian Classification

#### Beta-Binomial (二项式分布)

Binomial dist : $p(k|n,q) = (^n_k)q^k(1-q)^{n-k}$

 And its conjugate prior[共轭先验](http://blog.csdn.net/baimafujinji/article/details/51374202), Beta dist : $p(q) = Beta(q;\alpha, \beta) = \dfrac {\gamma (\alpha + \beta)}{\gamma(\alpha)\gamma(\beta)} q^{\alpha -1}(1 - q)^{\beta -1}$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9100590.png)

#### Suite of useful conjugate priors

| likelihood  | conjugate prior                          |
| ----------- | ---------------------------------------- |
| Normal      | Normal (for mean)                        |
| Normal      | Inverse Gamma (for variance) or Inverse Wishart(convariance) |
| Binomial    | Beta                                     |
| Multinomial | Dirichlet                                |
| Poisson     | Gamma                                    |

### Bayesian Model Selection

* Choosing the best model
  * linear, polynomial order, RBF basis/kernel
  * setting model hyperparameters
  * optimiser settings
  * type of model

Model selection using Bayes rule, to select between competing model classes

$ p(M_i|D) = \dfrac {p(D|M_i)p(M_i)}{p(D)}$

with $M_i$ as model i and D the dataset

The evidence : $p(D|M) = p(\boldsymbol{y|X}, M)$

---

## L20 - PGM Representation

### Probabilistic Graphical Models

> PGM aka "Bayes Nets" 贝叶斯网络

### Joint Distribution

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9101978.png)

M boolean r.v.'s  require $2^M -1 $ rows

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9102323.png)

Cons:Tables are too large

Therefore, assume independence

### Factoring Joint Distributions

Chain Rule: For any ordering of r.v's can always factor:

$Pr(X_1,X_2,…,X_k) = \prod^k_{i=1}Pr(X_i|X_{i+1},…,X_k)$

Model's independence assumptions correspond to

* Dropping conditioning r.v.'s in the factors
* Example unconditional independence : $Pr(X_1|X_2) = Pr(X_1)$
* Example conditional independence : $ Pr(X_1|X_2,X_3) = Pr(X_1|X_2)$
* Example independent r.v.'s $Pr(X_1,…,X_k) = \prod^k_{i=1}Pr(X_i)$

### Directed PGM

Conditional dependence:

* Node table: $Pr(child|parents)$

* Child directly depends on parents

* Joint factorisation

  * $Pr(X_1,X_2,…,X_k) = \prod^k_{i=1}Pr(X_i|X_j \in parents(Xi))$

   ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9103784.png)

Example:

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9103926.png)

### Naive Bayes

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9104227.png)

#### short hand for repeats: Plate notation

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9104281.png)

---

## L21 - Independence in PGMs

### Marginal and Conditional Probability distributions

* Marginal distribution: assigns a probability to configurations of a set of res, $p(A=a, B=b)$
* Conditional Distribution: includes conditioning context in the form of a disjoint set of rvs taking on specific values, $p(A=a, B=b| C=c, D=d)$ 

### Independence relations (D-separation)

* Marginal independence $P(X,Y) = P(X)P(Y)$
* Conditional independence $P(X,Y|Z) = P(X|Z)P(Y|Z)$

Notation $A\bot B|C$:

* RVs in set A are independent of RVs in set B, when given the values of RVs in C
* Symmetric: can swap roles of A and B
* $A \bot B$ denotes marginal independence, $C = \emptyset$

#### Marginal independence in Graph

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9152765.png)

$X \bot Y \leftarrow P(X,Y) = P(X)P(Y)$

When $X \bot Z$ where Z connected to Y

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9152797.png)

For the example:

Head-to-head ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9152975.png)

Marginal independence denoted $X \bot Y$

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9153061.png)

### Conditional independence

 Tail-to-tail![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9153767.png)
$$
\because P(AB) = P(A|B)P(B) \therefore P(A|B) = \dfrac {P(AB)}{P(B)} \\
P(X,Y|Z) = \dfrac {P(X,Y,Z)}{P(Z)} = \dfrac {P(Z)P(X|Z)P(Y|Z)}{P(Z)} = P(X|Z)P(Y|Z) \\
\therefore X \bot Y|Z
$$
Head-to-tail ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9154009.png)
$$
P(X,Y|Z) = \dfrac {P(X)P(Z|X)P(Y|Z)}{P(Z)} = \dfrac {P(Z)P(X|Z)P(Y|Z)}{P(Z)} = P(X|Z)P(Y|Z)
$$

> Marginal dependence ≠ conditional independence

#### How to apply to larger graphs?

* based on paths separating nodes, do they contain nodes with head-to-head, head-to-tail,or tail-to tail links?
* can all [undirected!] paths connecting two nodes be blocked by an independence relation?

### Markov Blanket

Solve using d-separation rules from graph

### Undirected PGMs

> key difference between undirected and directed is the normalisation

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9161790.png)

Based onn notion of

* Clique: a set of fully connected nodes (e.g., A-D,C-D, C-D-F)
* Maximal clique: largest cliques in graph (not C-D, due to C-D-F)

Joint probability defined as 

$P(a,b,c,d,e,f) = \dfrac {1}{Z} \psi_1(a,b) \psi_2(b,c)\psi_3(a,d)\psi_4(d,c,f)\psi_5(d,e)$

For example: $ B \bot D | \{A,C\} $

where $\psi$ is a positive function and Z is the normalising 'partition' function

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9162225.png)

---

## L22 - PGM Probabilistic Inference

### Probabilistic inference on PGMs

> Computing other distributions from joint
>
> Joint + Bayes rule + marginalisation -> anything

Example

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9163416.png)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9163929.png)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9164333.png)

__The elimination algorithm__ is used for efficient marginalisation on probabilistic graphical models (PGMs)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9164676.png)

---

## L23 - PGM Statistical inference

> learn parameters from (missing) data

### Fully-observed case is "easy"

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9166625.png)

When presence of unobserved variables trickier

* Maximise likelihood of observed data only
* Marginalise full joint to get to desired "partial" joint
* $ argmax_{\theta \in \Theta} \prod^n_{i=1} \sum_{latentj} \prod_j p(X^j = x_i^j|X^{parents(j)} = x_i^{parents(j)})$
* This won't decouple
* Partially-observed case:
  * Init seed, then do until "convergence", fill missing as expectation. And MLE on fully-observed

### Expectation-Maximisation Algorithm

* Seed parameters randomly
* E-step: Complete unobserved data as posterior distributions (prob inference)
* M-step: Update parameters with MLE on the fully-observed data

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9166452.png)

---

## L24 - Hidden Markov Models & message passing

### Hidden Markov Models

> Model of choice for sequential data. A form of clustering (or dimensionality reduction) for discrete time series.

#### The HMM (and KAlman Filter)

* Sequential observed outputs from hidden state
  * states take discrete values (i.e. clusters)
  * assumes discrete time steps 1, 2, .., T
* The Kalman filter same with continuous Gaussian r.v.'s 

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9167403.png)

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/image/screenshot-9167503.png)



---

