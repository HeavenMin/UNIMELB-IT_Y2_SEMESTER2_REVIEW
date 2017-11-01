---
typora-copy-images-to: ./stream_image
---

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

When the probability of there is some pair of people with the same birthday is at least 1/2, the m is at least $1 + \sqrt{2n\log _e2}$

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

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/stream_image/screenshot.png)

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

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/stream_image/screenshot-9503860.png)

When item x arrives, set all of $h_1(x), h_2(x), h_3(x)…$ to true

Better k is $k = \ln(2)n/m$

$k$ is the num of hash functions

$n$ is the number of distinct elements in the stream

### Counters

> estimating counts 估计数量算法

__Morris counter__ : only use $loglogn$ bits space

```
Let z be 0
When a new item arrives:
	Flip a coin with success probability 1/2^z
	if success
		increment z
Return 2^z - 1 as the estimated count
```

$$
\begin{align*}
&\text{Let z be 0} \\
&\text{When a new item arrives:} \\
&\qquad\text{Flip a coin with success probability } 1/2^z \\
&\qquad\text{if success} \\
&\qquad\qquad\text{increment z} \\
&\text{Return } 2^z-1 \text{ as the estimated count}
\end{align*}
$$



if there are n items, The expected value of $Y_n = 2^{Z_n}$ is n + 1
$$
\text{The relationship between } Z_{n-1} \text{ and } Z_n \\
Pr[Z_n=j] = Pr[Z_{n-1} =j](1- \dfrac {1}{2^j}) + Pr[Z_{n-1}=j-1]\dfrac {1}{2^{j-1}} \\
E[2^{Z_n}] = \sum_jPr[Z_n=j]2^j = \sum_jPr[Z_{n-1}=j]2^j + \sum_j(2Pr[Z_{n-1}=j-1] - Pr[Z_{n-1}=j]) \\
\text{The first term on the right hand side is simply } E[2^{Z_{n-1}}] = E[Y_{n-1}] \\
\text{And the second term collapses to } \sum_jPr[Z_{n-1}=j] = 1 \\
\therefore E[Y_n] = E[Y_{n-1}] + 1 \\
\text{Since the expected value of } Y_0 \text{ is } 1. \text{The expected value of returned solution is } n
$$

### Sampling

> 取样

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

__space need__: $k \log m$ (logo is for the counter, where m is the length of the stream)

$c(x) - m/k \leq \hat{c}(x) \leq c(x)$

$c(x) - \dfrac {m- \tau}{k} \leq \hat{c}(x) \leq c(x)$

Therefore, if we want to have an estimate that is accurate within $\epsilon F_1$ , where $F_1$ is the sum of the frequencies(in this case, the same as m), we need k to be around $1/\epsilon$ 

####AMS(Alon Matias Szegedy)

> Distinct items
>
> determining the number of distinct items in a stream (数0方法)
>
> using 2-universal family mapping

If we record the largest power of two that divides into a hash value we’ve seen... ◦ That’s a good estimate of the number of distinct items in the stream 

```
# AMS algorithm
# choose a hash h function uniformly at random from a 2-universal family mapping {1,2…n} to {1,2…n}
Let z be 0
For each item x
    z gets max{z, zeros(h(x))}
return 2^(z+1/2)
```

`TODO` 

#### BJKST

> Distinct items
>
> we track the hash value of roughly $1/\epsilon^2$ items, where $\epsilon$ is an accuracy parameter

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

$$
\begin{align*}
&\text{#BJKST 1} \\
&\text{#choose a hash h function uniformly at random from a 2-universal family mapping {1,2…n} to \{1,2…}n^3\} \\
&\text{Let t be } 96/\epsilon^2 \\
&\text{Let Q record the t smallest hash values seen so far [initialize with t values > } n^3] \\
&\text{For each item x} \\
&\qquad\text{Let m be the largest hash value in Q} \\
&\qquad\text{If } h(x) < m \text{ AND } h(x) \text{ is not already in Q} \\
&\qquad\qquad\text{Replace m with h(x) in Q} \\
&\text{Let m be the largest hash value in Q} \\
&\text{Return } tn^3/m
\end{align*}
$$



#### Universal hash functions

$h_{ab}(x) = ((ax+b)\text{ mod }p)\text{ mod }r$

where the family of hash functions has

a drawn from 1,2,…,p-1

b drawn from 0,1,2…,p-1,p

p is a prime number at least as big as max(n, r)

`question` Two universial family?

---

## L6

### Estimating "higher" functions

$F_2$ is like $(\text{Frequency of "Matthias"})^2 + (\text{Frequency of "Tony"})^2 + …$

— Similarity join estimation in large databases

* $F_0$ : sum of (Frequency of x)$^0$ — distinct item
* $F_1$: sum of (Frequency of x)$^1$ — total count

$F_k \equiv \sum_xf_x^k$

__Another AMS__
$$
\begin{align*}
&\text{Choose an item y uniformly at random from the stream(or pick its position j)} \\
&\text{Let r be the number of occurrences of y after and including posiiton j} \\
&\text{Return the value } z =m(r^k - (r-1)^k), \text{where m is the length of the stream}
\end{align*}
$$
the expected value of the return value Z will be $\sum_xf_x^k$

Because it's variance is huge

We need to use __Median of means__ to deal with it

* first take the mean of several estimators
* then we just find the median of several of these grouped means

### $F_2$ sketch (AMS tug-of-war)

> also called the tug-of-war sketch, and it's incredibly small

$$
\begin{align*}
&\text{Pick a random hash function h mapping {1,2,...,n} to }  \pm 1, \text{from a four-universal family} \\
&\text{Let } z \leftarrow 0 \\
&\text{As each item x arrives, with "count" c:} \\
&z \leftarrow z + ch(x) \\
&\text{Output } z^2
\end{align*}
$$

#### How it works

The random variable Z is

$\sum_i f_i h(i)$

We return $z^2$ , so the expected value of it is:

$E[\sum_if_i^2h(i)^2 + \sum_i\sum_{j:j!=i}f_if_jh(i)h(j)]$

Because the property of h (being 4-universal) the expected value of h(i)h(j) is the product of the individual expected values.

And also, the hash function maps to $\pm1$ , there's equal chance it maps a particular input to +1 and -1. 

so $E[h(i)] = 0$
Also, $h(i)^2$ is always 1, so its expected value is 1

Hence, $E[Z^2] = \sum_if_i^2 = F_2$ 

Also $E[Z^4] = \sum_i\sum_j\sum_k\sum_l f_i f_j f_k f_l E[h(i)h(j)h(k)h(l)]$

because of 4-wise independence

We have some terms with $h(i)^4$ and others with $h(i)^2h(j)^2$

*  All other terms cancel
* And there are $(^4_2) = 6$ copies of each $h(i)^2h(j)^2$

So $E[Z^4] = F_4 + 6 \sum_i \sum_{j>i}f_i^2f_i^2$

---

## L7

### Count-min sketch

> used to estimate frequencies of items, and can help in finding "heavy hitters"
>
> also they work in turnstile stream — item can left after they arrive

 ![screenshot](/Users/heaven/Projects/UNIMELB-IT_Y2_SEMESTER2_REVIEW/stream_image/screenshot-9511587.png)

Count-min sketch is somewhat like the AMS tug of war, and also like a Bloom FIlters

* have a family of __d pairwise-independent__ hash functions and __w buckets__, so that each hash function maps an input to a value in {1,2,…,w}
* When an item x arrives with count increment c (can be negative), we add c to several counters, based on the hashed values of x: $h_1(x), h_2(x),…,h_d(x)$
* Later, we look in the same cells to estimate f(x)
* how to get frequency of y: we look at the d different cells that y would be mapped to, and return the  __minimum__ of the frequencies

Space footprint: $d \times w \times \log m + d \log n$ bits

total space is at most $O(\dfrac {1}{\epsilon}(\log \dfrac {1}{\delta} + \log n)(\log m + \log n))$

And the estimates are accurate within an additive term $\epsilon F_1$

The estimate returned by the count-min sketch is never an __underestimate__ of an item's frequency (non-negative turnstile model)

 `TODO` how to choose w and d?

### Count sketch

`question` explain the count sketch

* a family of d hash functions , which are also 2-universal, and map to {-1,+1}
* When we see an item count increment c, instead of adding c to one slot per row in the table, we add $g_j(x) \times c$
* Finally, we return the __median__ of the d values $g_j(y)T[j, h_j(y)]$  stored for a given item y
  * where T is the table of estimated counts

---

## L9

### Metric-style clustering

$d(x,y)$ represents distance between x and y

Key aspect is that we have __triangle inequality__

$d(x,z) \leq d(x,y) + d(y,z)$

Euclidean distance to calculate the distance

### k-Center

We simply focus on the customer that is furthest from all the facilities

Clustering cost is largest of these farthest point dist

> it's a computationally hard problem

#### The standard algorithm

> achieves a solution within twice the optimal cost

* Pick a point arbitrarily
* while we have picked fewer than k points:
  * Pick the point that is farthest from the points picked so far
* return the k picked points



__other two algorithm__

* doubling algorithm: $O(k)$ space
* Guha's $2+\epsilon$  algorithm: using $O(\dfrac {k}{\epsilon})$ space

### Doubling algorithm

* Initialize the algorithm by taking first k+1 elements from the stream and setting (y,z) to be the cloest pair in these first k+1 elements
* let $\tau$ be $d(y,z)$ and let out representatives R be the k+1 elements so far, except z
* For each item x
* If its minimum distance to an element of R is $>2\tau$
  * Add x to R
  * While |R|>k:
    * Double $\tau$ ($\tau \leftarrow 2\tau$)
    * Find a maximal subset R* of R so that for every pair of distinct items in R*, their distance is at least $\tau$
    * Let R be R*

__Three properties of doubling algorithm__

* For all pairs of items in R, their distance is at least $\tau$
* The k-center cost of R with the whole stream (so far) is at most $2\tau$
* After initialization, and before each reset of R: An optimal solution has cost at least $\tau/2$

### Guha's algorithm



















---











