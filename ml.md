# COMP90051 Statistical Machine Learning
#Review/ML

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
[r.v.](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F)
> A random variable X is a numeric function.  

### Expectation and Variance
E[X] is the r.v. X’s “average” value
![](ml/ml/450F947D-9D70-4E53-91A3-2055715BB7CB.png)

### Bayes’s Theorem
* P(A,B) = P(A|B)P(B) = P(B|A)P(A)
* __P(A|B) = P(B|A)P(A)/P(B)__
	* Marginals: probabilities of individual variables
	* Marginalisation: summing away all but r.v.’s of interest

## L2 - Statistical schools
### MLE- Maximun-likelihood Estimation
![](ml/ml/myscript_mathpad.png)

### MLE ‘algorithm’
* 




