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

if there are n items, The expected value of $Y_n = 2^{Z_n}$ is n + 1

































---











