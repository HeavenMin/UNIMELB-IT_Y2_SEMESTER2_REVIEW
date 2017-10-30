# COMP90056 Stream Computing and Applications

#### Review/Stream

`Author: Min Gao`

> `important aspects`
>
> Hash functions, fundamental data structures, probability, communication complexity, algorithms on graphs

`important formula summary`
$$
1 - x < e^{-x} \\
1 + 2 + ... + n = \dfrac {n(n+1)}{2} \\
1 + 1/2 + 1/3 + ...+ 1/n \text{ is close to } \ln (n) \\
\text{Markov's Inequality: } Pr[X \geq \alpha \mu] \leq \dfrac {1}{\alpha} \\
\text{Variance: } Var(X) = E[(X - \mu)^2] = E(X^2) - (E[X])^2 \\
\text{Chebyshev's inequality: } Pr(|X- \mu| \geq b) \leq \dfrac {Var(X)}{b^2} \\
\text{or } Pr(|x-a| \geq b) \leq \dfrac {(x-a)^2}{b^2} \\
 (^n_k) = \dfrac {n!}{(n-k)!k!}
$$

## L1

> Several computational problems are going to solved in this subject

* Data arrives at a rapid rate
* do not have enough space to store it all
* might need to provide an answer at any point in the stream
* should provide quick updates to our data structures

Using a mathematical & probabilistic approach

### Birthday paradox

Example: The probability of all 25 people having a different birthday is:

$ (^{365}_{25})25! / 365^{25}$

A more manageable approach:

Image there are n possible birthdays and m people, then the probability they are all different is

$ 1(1 - \dfrac {1}{n})(1 - \dfrac {2}{n})( 1 - \dfrac {3}{n})…(1 - \dfrac {m - 1}{n})$ 

$ \because 1 - x \leq e^{-x}$

$\therefore  1(1 - \dfrac {1}{n})(1 - \dfrac {2}{n})( 1 - \dfrac {3}{n})…(1 - \dfrac {m - 1}{n}) \leq exp(-[\dfrac {1}{n} + \dfrac {2}{n} + … + \dfrac {m-1}{n}]) $

$ \leq exp(-\dfrac {m(m-1)}{2n})$

### Collecting coupons

When the probability to roll a dice of 'six' is 1/6, then the expected number to roll a 'six' is 6.

__expectation__ is linear, regardless of independence

So the expectation of the total number of boxes bought is 

$ n/n +n/(n-1)+…+ n/1$

$\because 1 + 1/2 + 1/3 + …+ 1/n$ is close to $\ln (n)$ 

The average number of boxes purchased is $n \ln (n)$

---

## L2

### Randomized Algorithms and Probability

#### Indicator random variables

Indicator variable
$$
X = {
\begin{cases}
1, &\text{some event}\\
0, &\text{otherwise}\\
\end{cases}
}
$$
The expected value of X is $E[X] = 1 \times Pr[some event] + 0 \times Pr[otherwise]$, which is 1/2 if the probability of event is fair.

#### Expectations

When a random variable Y takes on discrete values, such as 0,1,2,3,...

$E[Y] = 0 \times Pr[Y=0] + 1 \times Pr[Y=1] + 2\times Pr[Y=2] + …$

Another definition is:

$ E[Y] = Pr[Y \geq 1] + Pr[Y \geq 2] + Pr[Y \geq 3] + …$

also, $E[Y+Z] = E[Y] + E[Z]$  and $E[aY] = a E[Y]$

If two events are independent: $ Pr[Y=a \text{ and }Z = b] = Pr[Y=a]Pr[Z=b]$

And $E[YZ] = E[Y]E[Z]$

There is a __union bound__ $Pr[A_1 \text{ or } A_2 \text{ or } A_3 \text{ or } …] \leq Pr[A_1] + Pr[A_2] + ...$

```
Example:
suppose roll two fire six-sided dice
the expectation of the sum of values of the dice? 7
the expectation of maximum of the values?
```

$$
Pr[max_2 \leq 4] =Pr[dice_1 \leq 4 \text{ AND } dice_2 \leq 4] \\
=2/3 * 2/3 = 4^2/36 \\
Pr[max_2 \leq 3] = 3^2/36\\
\therefore Pr[max_2 \leq i] = i^2/36 \\
Pr[max_2 \geq i+1] = 1 - i^2/36 \\
E[max_2] = Pr[max_2 \geq 0] + Pr[max_2 \geq 1] +...+ Pr[max_2 \geq 6] \\
E[max_2] = 1 + (1- 1^2/36) + ...+ (1- 6^2/36) \\
$$

### Permutations

 $ (^n_k) = \dfrac {n!}{(n-k)!k!}$

A useful equation : $ (^n_k) \leq \dfrac {n^k}{k!} \leq \left( \dfrac {en}{k} \right)^k$

`quick puzzle` : $1+2+…+n = \dfrac {(n+1)n}{2} = (^{n+1}_{2})$

### Variance

> variance is defined to be the expected value of the square of the deviation from the mean

$Var[X] = E[X-E[X]]^2 = E[X^2] - E[X]^2$

$\therefore Var[aX] = a^2Var[X]$

And if X and Y are independent, then $Var[X+Y] = Var[X] + Var[Y]$

And also the __standard deviation__ is $\sqrt{Var[X]}$

### Binomial Distribution (二项分布)

$Pr[X=k] = (^n_k) p^k(1-p)^{(n-k)}$

mean of it is : $np$

Variance is $np(1-p)$

 ![screenshot](/Users/heaven/Library/Group Containers/Q79WDW8YH9.com.evernote.Evernote/Evernote/quick-note/1023142-personal-app.yinxiang.com/quick-note-9zNhGa/attachment--bOeNnc/screenshot.png)

```
Consider tossing a fair coin n times, what's the expected length of the longest streak of heads?
```

$\log_2 n$ and the logest streak is at least $\dfrac {\log_2 n}{2}$ and at most $ 2\log_2n$ 

### Balls in bins

`TODO`

## L3

### Hash function

__pros__:

* It spreads items evenly
* It is fast
* a given key must be consistently hashed to the same value

__cons__:

* the hash function might be slow
* several items might be hashed to the same position in the table (collision)

Hash functions are the backbone of many streaming techniques.

### Bloom Filter

> have a family of k hash functions
>
> 通过N个hash function将一个元素映射在hash表中的N个值中

 ![screenshot](/Users/heaven/Library/Group Containers/Q79WDW8YH9.com.evernote.Evernote/Evernote/quick-note/1023142-personal-app.yinxiang.com/quick-note-9zNhGa/attachment--JsBXdK/screenshot.png)

When item x arrives, set all of $h_1(x), h_2(x), h_3(x)…$ to true

Better k is $k = \ln(2)n/m$

$k$ is the num of hash functions

$n$ is the number of distinct elements in the stream

### Counters

> estimating counts

__Morris counter__ : only use $loglogn$ bits space

```
Let z be 0
When a new item arrives:
	Flip a coin with success probability 1/2^z
	if success
		increment z
Return 2^z - 1 as the estimated count
```

if there are n items, The expected value of $Y_n = 2^{Z_n}​$ is n + 1
$$
\text{The relationship between } Z_{n-1} \text{ and } Z_n \\
Pr[Z_n=j] = Pr[Z_{n-1} =j](1- \dfrac {1}{2^j}) + Pr[Z_{n-1}=j-1]\dfrac {1}{2^{j-1}} \\
E[2^{Z_n}] = \sum_jPr[Z_n=j]2^j = \sum_jPr[Z_{n-1}=j]2^j + \sum_j(2Pr[Z_{n-1}=j-1] - Pr[Z_{n-1}=j]) \\
\text{The first term on the right hand side is simply } E[2^{Z_{n-1}}] = E[Y_{n-1}] \\
\text{And the second term collapses to } \sum_jPr[Z_{n-1}=j] = 1 \\
\therefore E[Y_n] = E[Y_{n-1}] + 1 \\
\text{Since the expected value of } Y_0 is 1. \text{The expected value of returned solution is } n
$$

### Sampling

#### Reservoir sampling

When a stream of data arrives, and we can't store it all, one simple idea is to take a uniform random sample of it.

```
Let S[1..k] be an empty array	#k个可以存储的位置
Let m be 0
For each item x
	increment m
	if m <= k
		put x in S[m]	 #先将位置全部填满
	Else
		Let r be chosen uniformly in [1..m]	#之后进来的有k/m的几率替换进列表
		If r <= k, S[r] becomes x
Output S
```

When k = 1, for every item in stream, the chance that it's the sampled item is 1/m

---

## L4

### Frequent Items

> Finding frequent items in a stream

#### Misra-Gries algorithm

> key: track __k-1__ items, only item more than __m/k__ times can be record finally 

```
# we track of k-1 items, with a counter for each
# only item more than m/k times can be record finally

While the stream is not empty
    let the new item be x
    if x is a tracked item
        increment its counter
    if not and if fewer than k-1 items are tracked
        Add x to tracked items, with a count of 1
    Else
        Decrement the count of every tracked item
        evict every tracked item that has count zero
return the tracked items
```

For all items, the frequency estimate is at most the true frequency and at least the true frequency minus m/k 

####Distinct items (AMS)

> determining the number of distinct items in a stream

If we record the largest power of two that divides into a hash value we’ve seen... ◦ That’s a good estimate of the number of distinct items in the stream 

```
# AMS algorithm
# choose a hash h function uniformly at random from a 2-universal family mapping {1,2…n} to {1,2…n}
Let z be 0
For each item x
    z gets max{z, zeros(h(x))}
return 2^(z+1/2)
```

#### BJKST

```
#BJKST 1
#choose a hash h function uniformly at random from a 2-universal family mapping {1,2…n} to {1,2…n^3}
Let t be 96/epsilon^2
Let Q record the t smallest hash values seen so far [initialize with t values > n^3]
For each item x
	Let m be the largest hash value in Q
	if h(x) < m AND h(x) is not already in Q
		Replace m with h(x) in Q
Let m be the largest hash value in Q
return tn^3/m
```

#### Universal hash functions

$h_{ab}(x) = ((ax+b)\text{ mod }p)\text{ mod }r$

where the family of hash functions has

a drawn from 1,2,…,p-1

b drawn from 0,1,2…,p-1,p

p is a prime number at least as big as max(n, r)

---

## L5





























---











