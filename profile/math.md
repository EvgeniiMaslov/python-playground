[**Linear Algebra**](#Linear algebra)

* [Eigen decomposition](#Eigen decomposition)
* [Singular Value Decomposition](#Singular Value Decomposition)





**Statistic**

* [The main divisions of DS questions](#the-main-divisions-of-ds-questions)
* [Basic statistic](#basic-statistic)
  * [Estimates of location](#estimates-of-location)
  * [Estimates of variability](#estimates-of-variability)
  * [Correlation and Covariance](#correlation-and-covariance)
* [Distributions](#distributions)
  * [PMF, CDF, PDF](#PMF,-CDF,-PDF)
  * [Kernel density estimation](#Kernel density estimation)
  * [Sampling distribution of statistics](#Sampling distribution of statistics)
  * [The Bootstrap](#The Bootstrap)
  * [Confidence Interval](#Confidence interval)
  * [Skewness](#Skewness)
  * [Estimate Distribution](#Estimate distribution)
  * [Chebyshev’s Theorem](#Chebyshev's Theorem)
  * [Common Distributions](#common-distributions)
  * [Determine Unknow Parameters of Normal Distribution](#Determine Unknow Parameters of Normal Distribution)
  * [Dissimilarity measures between two distributions](#Dissimilarity measures between two distributions)

**Statistical Experiments and Significance testing**

* [A/B Testing](#A/B Testing)
* [Hypothesis Test](#Hypothesis Test)
* [Example: Flipping a Coin](#Example: Flipping a Coin)
* [Example: Running an A/B Test](#Example: Running an A/B Test)
* [P-value](#P-value)
* [Resampling](#Resampling)
* [Statistical Significance and P-Values](#Statistical Significance and P-Values)
* [t-Tests](#t-Tests)
* [Однофакторный дисперсионный анализ](#Однофакторный дисперсионный анализ)
* [Двухфакторный анализ дисперсии](#Двухфакторный анализ дисперсии)
* [Multiple Testing](#Multiple Testing)
* [ANOVA ](#ANOVA)
* [Chi-Square Test](#Chi-Square Test)
* [Multi-Arm Bandit Algorithm](#Multi-Arm Bandit Algorithm)
* [Power and Sample Size](#Power-and-Sample-Size)
* 



[Probability](#Probability)



* [Central limit theorem](#Central limit theorem)
* [Bayesian curve fitting](#bayesian-curve-fitting)
* [Curse of dimensionality](#Curse of dimensionality)









## Linear Algebra



### Eigen decomposition

$$
\bold A \bold v = \lambda \bold v
$$

\lambda = eigenvalue, **v** - eigenvector. We can form a matrix **V** by concatenating vectors **v** with one vector per column. Likewise, we can concatenate the eigenvalues to form a vector **λ**. The eigendecomposition of **A** is given by:
$$
\bold A = \bold V diag(\bold λ) \bold V^{-1}
$$
Every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and eigenvalues:
$$
\bold A = \bold Q \bold Λ \bold Q^T
$$
Where **Q** - orthogonal matrix composed of eigenvectors of **A**, and **Λ** is a diagonal matrix, Λ_i,i is associated with eigenvector in column i of **Q**, denoted as **Q**_:,i

Ортогона́льная ма́трица — квадратная матрица с вещественными  элементами, результат умножения которой на транспонированную матрицу  равен единичной матрице: или, что эквивалентно, её обратная матрица равна транспонированной  матрице.



### Singular Value Decomposition



Every real matrix has a SVD, but the same is not true of the eigen decomposition. The SVD is similar to eigen decomposition, except A is the product of three matrices:
$$
\bold A = \bold U \bold D \bold V^T
$$
**U** and **V** - orthogonal matrices, **D** - diagonal. The elements along the diagonal of **D** are known as the singular values of matrix **A**. The columns of **U** - left-singular vectors; the columns of **V** - right-singular vectors.

We can interpret SVD in terms of eigen decomposition of **A**. The left-singular vectors of **A** are the eigenvectors of **A**^T; the right-singular vectors are the eigenvectors **A**^T **A**. The non-zero singular values of **A** are the square root of eigenvalues of **A**^T **A**.

U: u_j - собственные векторы матрицы A * A^T

V: v_j - собственные векторы матрицы A^T * A

D: на диагонали - корни из собственных значений матриц A^T * A и  A * A^T







## Statistic



### The main divisions of DS questions

There are, broadly speaking, six categories in which data analyses fall. In the approximate order of difficulty, they are:

* Descriptive statistics - generate statistics that **describe** or **summarize** a set of data.
* Exploratory data analysis - the goal is to examine or **explore** the data and find **relationships** that weren’t previously known.  It can allow you to formulate hypotheses and drive the design of future studies and data collection, but exploratory analysis alone should never be used as the final say on why or how data might be related to each other.
* Inferential analysis - the goal is to use a relatively **small sample** of data to **infer** or say something about the **population** at large. Inferential analysis typically involves using the data you have to estimate that value in the population and then give a measure of your uncertainty about your estimate.
* Predictive analysis - the goal is to use **current** data to make **predictions** about **future** data. Essentially, you are using current and historical data to find patterns and predict the likelihood of future outcomes.
* Causal analysis - the goal is to see what happens to one variable when we manipulate another variable - looking at the **cause** and **effect** of a **relationship**.
* Mechanistic analyses - the goal is to understand the **exact changes in variables** that lead to **exact changes in other variables**.



### Basic statistic

#### Estimates of location

* Raw moment - A raw moment of order k is the average of all numbers in the set, with each number raised to the kth power before you average it. So the first raw moment is the arithmetic mean.
  $$
  \frac{1}{n} \sum_i x_i^k
  $$

* Mean - the sum of all values divided by the number of values
  $$
  \frac{\sum_{i=1}^{n}x_i}{n}
  $$

* Trimmed mean - The average of all values after dropping a fixed number of extreme values
  $$
  \frac{\sum_{i=p+1}^{n-p}x_i}{n-2p}
  $$

* Weighted mean - The sum of all values times a weight divided by the sum of the weights
  $$
  \frac{\sum_{i=1}^{n}w_ix_i}{\sum_{i=1}^{n}w_i}
  $$

* Median = The value such that the one-half of the data lies above and below

* Weighted median - The value such that one-half of the sum of the weights lies above and below the sorted data

* Mode - the most commonly occurring category or value in data set

* Expected value - The expectation of some function f(x) with respect to a probability distribution P(x) is the averaged or mean value that f takes on when x is drawn from P. For discrete variables this can be computed with a summation, while for continuous variables, it is computed with an integral:
  $$
  E_{x∼P}[f(x)] = \sum_x P(x)f(x)
  \\
  E_{x∼p}[f(x)] = \int p(x)f(x)dx
  $$

  If we are given a finite number N of points drawn from the probability distribution or probability density, then the expectation can be approximated as a finite sum over these points:
  $$
  E[f] \simeq \frac{1}N \sum_{n=1}^N f(x_n)
  $$
  We can also consider a conditional expectation with respect to a conditional distribution, so that:
  $$
  E_x[f|y] = \sum_x p(x|y)f(x)
  $$
  with an analogous definition for continuous variables.

  





#### Estimates of variability

* Deviation - The difference between the observed values and the estimate of location

* Central moments - is based on the average of deviations of numbers from their mean. When k = 2 the result is the second central moment, which you might recognize as variance.
  $$
  \frac{1}{n} \sum_i (x_i - \bar x)^k
  $$

* Variance - The sum of squared deviations from the mean divided by n-1 where n is the number of data values
  $$
  \frac{\sum{(x - \bar{x})^2}}{n-1}
  $$
  The variance gives a measure of how much values of a function of a random variable x vary as we sample different values of x from its probability distribution.
  $$
  Var(f(x)) = E[(f(x)-E[f(x)])^2]
  $$

* Standard deviation - The square root of the variance
  $$
  \sqrt{variance}
  $$

* Mean absolute deviation - The mean of the absolute value of the deviations from the mean.
  $$
  \frac{\sum_{i=1}^{n}{\lvert x_i - \bar{x} \rvert}}{n}
  $$

* Median absolute deviation from the median - The median of the absolute value of the deviations from the median. **Robust estimate.**
  $$
  median(\lvert x_1 - m \rvert, \lvert x_2 - m \rvert, ... , \lvert x_n - m \rvert)
  $$

* Range - The difference between the largest and smallest value in data set

* Order statistics - Metrics based on the data values sorted from smallest to biggest

* Percentile - The value such that P percent of the values take on this value or less and (100 - P) percent take on this value or more. **Robust estimate.** **In machine learning it's just weighted average**:
  $$
  (1 - w)x_j + wx_{j+1}
  $$


  ```python
  def PercentileRank(scores, your_score):
      count = 0
      for score in scores:
          if score <= your_score:
              count += 1
      percentile_rank = 100.0 * count / len(scores)
      return percentile_rank
  
  def Percentile(scores, percentile_rank):
  	scores.sort()
  	for score in scores:
  		if PercentileRank(scores, score) >= percentile_rank:
  			return score
  
  ```

* Interquartile range - The difference between 75th percentile and the 25th percentile

* z-score - Distance from mean in the units of standard deviation
  $$
  z = \frac{x - \bar x}{\sigma}
  $$



#### Correlation and Covariance

A correlation is a statistic intended to quantify the stren4gth of the relationship between two variables. Correlation shows linear dependencies between variable. **Sensitive to outliers.**

A challenge in measuring correlation is that the variables we want to compare are often not expressed in the same units. And even if they are in the same units, they come from different distributions. There are two common solutions to these problems: 

1. Transform each value to a standard score, which is the number of standard deviations from the mean. This transform leads to the “Pearson product-moment correlation coefficient.” 

   For sample:
   $$
   \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{(n - 1)\sigma_x\sigma_y} \\
   $$
   For population:

$$
p = \frac{Cov(X,Y)}{S_XS_Y}
$$

2. Transform each value to its rank, which is its index in the sorted list of values. This transform leads to the “Spearman rank correlation coefficient.” To compute Spearman’s correlation, we have to compute the rank of each value, which is its index in the sorted sample. Spearman’s rank correlation is an alternative that mitigates the effect of outliers and skewed distributions.

**The covariance** gives some sense of how much two values are linearly related to each other, as well as the scale of this variables:
$$
Cov(f(x), g(y)) = E[(f(x) - E[f(x)])(g(y) - E[g(y)])]
$$


Covariance is the mean of these products:
$$
Cov(X, Y) = \frac{1}{n}\sum dx_i dy_i
\\
dx_i = x_i - \bar x
\\
dy_i = y_i - \bar y
$$


### Distributions



#### PMF, CDF, PDF

* **Probability mass function** - a function that gives the probability that a discrete random variable is exactly equal to some value. Probability mass function is the probability distribution of a discrete random variable, and provides the possible values and their associated probabilities.

* **Cumulative distribution function** (функция распределения) - function that maps from a value to its percentile rank. To evaluate CDF(x) for a particular value of x, we compute the fraction of values in the distribution less than or equal to x

  ```python
  def eval_cdf(sample, x):
      count = 0
      for score in sample:
          if score <= x:
              count += 1
              
      prob = count/len(sample)
      return prob
  ```

* **Probability density function** (функция плотности вероятности) - The derivative of a CDF. Evaluating a PDF for a particular value of x is usually not useful. The result is not a probability; it is a probability density.

  

#### Kernel density estimation

https://www.youtube.com/watch?v=x5zLaWT5KPs

Density estimation involves selecting a probability distribution function and the parameters of that distribution that best explains the joint probability distribution of the observed data (*X*).

A density estimator is an algorithm that seeks to model the probability distribution that generated a dataset. For one-dimensional data, you are probably already familiar with one simple density estimator: the histogram. A histogram divides the data into discrete bins, counts the number of points that fall in each bin, and then visualizes the results in an intuitive manner.

Kernel density estimation (KDE) is an algorithm that takes a sample and finds an appropriately smooth PDF that fits the data.

We started with PMFs, which represent the probabilities for a discrete set of values. To get from a PMF to a CDF, you add up the probability masses to get cumulative probabilities. To get from a CDF back to a PMF, you compute differences in cumulative probabilities. A PDF is the derivative of a continuous CDF; or, equivalently, a CDF is the integral of a PDF. Remember that a PDF maps from values to probability densities; to get a probability, you have to integrate. To get from a discrete to a continuous distribution, you can perform various kinds of smoothing. One form of smoothing is to assume that the data come from an analytic continuous distribution (like exponential or normal) and to estimate the parameters of that distribution. Another option is kernel density estimation. The opposite of smoothing is discretizing, or quantizing. If you evaluate a PDF at discrete points, you can generate a PMF that is an approximation of the PDF. You can get a better approximation using numerical integration. To distinguish between continuous and discrete CDFs, it might be better for a discrete CDF to be a “cumulative mass function,” but as far as I can tell no one uses that term.



The free parameters of kernel density estimation are the kernel, which specifies the shape of the distribution placed at each point, and the kernel bandwidth, which con‐ trols the size of the kernel at each point. 



Let {xi}N i=1 be a one-dimensional dataset (a multi-dimensional case is similar) whose examples were drawn from a distribution with an unknown pdf f with xi œ R for all i = 1,...,N. We are interested in modeling the shape of f. Our kernel model of f, denoted as ˆfb, is given by
$$
\hat f_b(x) = \frac{1}{Nb} \sum_{i=1}^N k (\frac{x-\bar x}{b})
$$
where b is a hyperparameter that controls the tradeoff between bias and variance of our model and k is a kernel. We use a Gaussian kernel:
$$
k(z) = \frac{1}{\sqrt{2\pi}}exp(\frac{-z^2}{2})
$$


![](C:/Users/sqrte/python-playground/profile/img/KDE.png)

We look for such a value of b that minimizes the difference between the real shape of f and the shape of our model ˆfb. A reasonable choice of measure of this difference is called the mean integrated squared error (MISE):
$$
MISE(b) = E[\int_R(\hat f_b(x) - f(x))^2dx]
$$
Now, to find the optimal value b* for b, we minimize the cost defined as:
$$
\int_R \hat f^2_b(x)dx - \frac{2}{N} \sum_{i=1}^N \hat f^{(i)}_b(x_i)
$$


We can find b* using grid search. 



**Selecting the bandwidth via cross-validation** 

The choice of bandwidth within KDE is extremely important to finding a suitable density estimate, and is the knob that controls the bias–variance trade-off in the esti‐ mate of density: too narrow a bandwidth leads to a high-variance estimate (i.e., over‐ fitting), where the presence or absence of a single point makes a large difference. Too wide a bandwidth leads to a high-bias estimate (i.e., underfitting) where the structure in the data is washed out by the wide kernel. In machine learning contexts, we’ve seen that such hyperparameter tuning often is done empirically via a cross-validation approach. With this in mind, the KernelDen sity estimator in Scikit-Learn is designed such that it can be used directly within Scikit-Learn’s standard grid search tools. Here we will use GridSearchCV to optimize the bandwidth for the preceding dataset. 

```python
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut
bandwidths = 10 ** np.linspace(-1, 1, 100)

grid = GridSearchCV(KernelDensity(kernel='gaussian'), 
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut(len(x)))
grid.fit(x[:, None]);
```





#### Sampling distribution of statistics

* Sample statistic - A metric calculate for a subset of data set (mean, median, variance).

* Data distribution - The frequency distribution of individual values in a data set.

* Sampling distribution - The frequency distribution of a sample statistic over many samples or resamples

  Data distribution is the distribution of the observations in your data (for example: the scores of students taking statistics course).

  Sampling distribution of the sample mean: Let imagine you sample the data from population n times (randomly, each sample has N observations), for each sample you compute the mean. So you have n means of n samples. Then you have the distribution of the sample mean.

* **Central limit theorem** - The tendency of the sampling distribution to take a normal shape as sample size rises.

* Standard error - The variability (standard deviation) of a sample statistic over many samples (not to be confused with standard deviation, which, by itself, refers to variability of individual data values)


#### The Bootstrap

One easy and effective way to estimate the sampling distribution of a statistic, or of model parameters, is to draw additional samples, with replacement, from the sample itself and recalculate the statistic or model for each resample. Conceptually, you can imagine the bootstrap as replicating the original sample thousands or millions of times so that you have a hypothetical population that embodies all the knowledge from your original sample (it’s just larger).

1. Draw a sample value, record, replace it.

2. Repeat *n* times.

3. Record the mean of the *n* resampled values.

4. Repeat steps 1-3 *R* times.

5. Use the R results to:

   a. Calculate their standard deviation (this estimates sample standard error).

   b. Produce histogram or boxplot.

   c. Find a confidence interval.

#### Confidence interval



Suppose a dataset x1,...,xn is given, modeled as realization of random variables X1,...,Xn. Let θ be the parameter of interest, and γ a number between 0 and 1. If there exist sample statistics Ln = g(X1,...,Xn) and Un = h(X1,...,Xn) such that:
$$
P(L_n < \theta < U_n) = \gamma \text{ for every value of } \theta \text{, then}
\\
(l_n, u_n), 
$$
where ln = g(x1,...,xn) and un = h(x1,...,xn), is called a 100γ% confidence interval for θ. The number γ is called the confidence level.

**Normal distribution confidence interval for the mean:**

1. Variance known. If X1,...,Xn is a random sample from an N(µ, σ2) distribution, then X¯n has an N(µ, σ2/n) distribution, and from the properties of the normal distribution (see page 106), we know that:
   $$
   \frac{\bar X_n - \mu}{\sigma / \sqrt{n}} \text{ has an N(0, 1) distribution}
   $$
   If cl and cu are chosen such that P(c_l < Z < c_u) = \gamma for an N(0, 1) distributed random variable Z, then:
   $$
   \gamma = P(c_l < \frac{\bar X_n - \mu}{\sigma / \sqrt{n}} < c_u)
   \\
   = P(c_l \frac{\sigma}{\sqrt{n}}< \bar X_n - \mu < c_u \frac{\sigma}{\sqrt{n}})
   \\
   = P(\bar X_n - c_u \frac{\sigma}{\sqrt{n}} < \mu < \bar X_n - c_l \frac{\sigma}{\sqrt{n}})
   $$
   A common choice is to divide α = 1 − γ evenly between the tails,2 that is, solve cl and cu so that cu = zα/2 and cl = z1−α/2 = −zα/2 (z - critical value). Summarizing, the 100(1 − α)% confidence interval for µ is:
   $$
   (\bar x_n - z_{\alpha/2} \frac{\sigma}{\sqrt n}, \bar x_n + z_{\alpha/2} \frac{\sigma}{\sqrt n})
   $$

2. Variance unknown. If we substitute the estimator Sn for σ, the resulting random variable:
   $$
   \frac{\bar X_n - \mu}{S_n / \sqrt n}
   $$
   has a distribution that only depends on n and not on µ or σ. Moreover, its density can be given explicitly. A continuous random variable has a t-distribution with parameter m, where m ≥ 1 is an integer, if its probability density is given by:
   $$
   f(x) = k_m (1 + \frac{x^2}{m})^{- \frac{m+1}2} \text{ for } -\inf < x < \inf
   \\
   k_m = Γ(\frac{m+1}{2}) / (Γ(\frac{m}{2})\sqrt{mn})
   $$
   This distribution is denoted by t(m) and is referred to as the t-distribution with m degrees of freedom. The critical value tm,p is the number satisfying:
   $$
   P(T >= t_{m,p}) = p
   $$
   where T is a t(m) distributed random variable. Because the t-distribution is symmetric around zero, using the same reasoning as for the critical values of the standard normal distribution, we find that t_{m, 1-p} = - t_{m,p}

   For a random sample X1,...,Xn from an N(µ, σ2) distribution, the studentized mean has a t(n − 1) distribution, regardless of the values of µ and σ. From this fact and using critical values of the t-distribution, we derive that:
   $$
   P(-t_{n-1, \alpha/2} < \frac{\bar X_n - \mu}{S_n / \sqrt n} < t_{n-1, \alpha/2}) = 1 - \alpha 
   $$
   and in the same way as when σ is known it now follows that a 100(1 − α)% confidence interval for µ is given by:
   $$
   (\bar x - t_{n-1, \alpha/2}\frac{s_n}{\sqrt n}, \bar x + t_{n-1, \alpha/2}\frac{s_n}{\sqrt n})
   $$

3. Variance unknown, large samples. If n is large enough, we may use:
   $$
   (\bar x_n - z_{\alpha/2} \frac{s_n}{\sqrt n}, \bar x_n + z_{\alpha/2} \frac{s_n}{\sqrt n})
   $$







An x% confidence interval around a sample estimate should, on average, contain similar sample estimates x% of the time.

Given a sample of size n, and a sample statistic of interest, the algorithm for a bootstrap confidence interval is as follows:

1. Draw a random sample of size n with replacement from the data.
2. Record the statistic of interest for the resample.
3. Repeat steps 1-2 many (R)s times.
4. For an x% confidence interval, trim [(1 - [x/100])/2]% of the R resample results from either end of distribution.
5. The trim points are endpoints of an x% bootstrap confidence interval. 

**The lower the level of confidence you can tolerate, the narrower the confidence interval will be.**

**Here is a bootstrap algorithm for generating confidence intervals for regression parameters** (coefficients) for a data set with P predictors and n records (rows): 

1. Consider each row (including outcome variable) as a single “ticket” and place all the n tickets in a box. 
2. Draw a ticket at random, record the values, and replace it in the box. 
3. Repeat step 2 n times; you now have one bootstrap resample. 
4. Fit a regression to the bootstrap sample, and record the estimated coefficients.
5. Repeat steps 2 through 4, say, 1,000 times. 
6. You now have 1,000 bootstrap values for each coefficient; find the appropriate percentiles for each one (e.g., 5th and 95th for a 90% confidence interval).

A **prediction interval** pertains to uncertainty around a single value, while a confidence interval pertains to a mean or other statistic calculated from multiple values. 

**100 (1 - \alpha) Confidence interval for the difference between two population means: large, independent samples**
$$
(\bar x_1 - \bar x_2) \pm z_{\alpha/2} \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$
**100 (1 - \alpha) Confidence interval for the difference between two population means: small, independent samples**
$$
(\bar x_1 - \bar x_2) \pm t_{\alpha/2} \sqrt{ s_p^2 (\frac{1}{n_1} + \frac{1}{n_2})}
\\
s_p^2 = \frac{(n_1 - 1) s^2_1 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}
$$



**Example:**

Suppose the estimator T is unbiased for the speed of light θ. For the moment, also suppose that T has standard deviation σT = 100 km/sec (we shall drop this unrealistic assumption shortly). Then, applying formula, which was derived from Chebyshev’s inequality, we find:
$$
P(|T- \theta| < 2\sigma_T) >= 3/4
$$
In words this reads: with probability at least 75%, the estimator T is within 2σT = 200 of the true speed of light θ. We could rephrase this as:
$$
T ∈(\theta - 200, \theta + 200) \text{ with probability at least 75%}
$$
However, if I am near the city of Paris, then the city of Paris is near me: the statement “T is within 200 of θ” is the same as “θ is within 200 of T ,” and we could equally well rephrase as:
$$
\theta ∈ (T - 200, T + 200) \text{ with probability at least 75%}
$$
Note that of the last two equations the first is a statement about a random variable T being in a fixed interval, whereas in the second equation the interval is random and the statement is about the probability that the random interval covers the fixed but unknown θ. The interval (T − 200, T + 200) is sometimes called an interval estimator, and its realization is an interval estimate. Evaluating T for the Michelson data we find as its realization t = 299 852.4, and this yields the statement:
$$
\theta ∈ (299652.4 , 300052.4) \text{ with probability at least 75%}
$$












#### Skewness

Skewness is a property that describes the shape of a distribution. If the distribution is symmetric around its central tendency, it is unskewed. If the values extend farther to the right, it is “right skewed” and if the values extend left, it is “left skewed.”

Several statistics are commonly used to quantify the skewness of a distribution. Given a sequence of values, xi , the sample skewness, g1, can be computed like this:

```python
def StandardizedMoment(xs, k):
	var = CentralMoment(xs, 2)
	std = math.sqrt(var)
	return CentralMoment(xs, k) / std**k
def Skewness(xs):
	return StandardizedMoment(xs, 3)
```

g1 is the third standardized moment, which means that it has been normalized so it has no units.

In practice, computing sample skewness is usually not a good idea. If there are any outliers, they have a disproportionate effect on g1. Another way to evaluate the asymmetry of a distribution is to look at the relationship between the mean and median. Extreme values have more effect on the mean than the median, so in a distribution that skews left, the mean is less than the median. In a distribution that skews right, the mean is greater. Pearson’s median skewness coefficient is a measure of skewness based on the difference between the sample mean and median:
$$
g_p = 3(\bar x - m) / S
$$
Where ¯x is the sample mean, m is the median, and S is the standard deviation. 

#### Estimate distribution

Choose statistic(mean, median, etc.) that minimize RMSE between estimator and true value for central. You can also use maximum likelihood estimator.

*Guess the variance*. For large samples, S^2 is an adequate estimator, but for small samples it tends to be too low.  Because of this unfortunate property, it is called a biased estimator. An estimator is unbiased if the expected total (or mean) error, after many iterations of the estimation game, is 0. Fortunately, there is another simple statistic that is an unbiased estimator of σ^2 :
$$
S_{n-1}^2 = \frac{1}{n-1} \sum (x_i - \bar x)^2
$$


After you choose an estimator with appropriate properties, and use it to generate an estimate, the next step is to characterize the uncertainty of the estimate, which is the topic of the next section.

There are two common ways to summarize the sampling distribution: 

*  Standard error (SE) is a measure of how far we expect the estimate to be off, on average. For each simulated experiment, we compute the error, ¯x − µ, and then compute the root mean squared error (RMSE). In this example, it is roughly 2.5 kg.  
*  A confidence interval (CI) is a range that includes a given fraction of the sampling distribution. For example, the 90% confidence interval is the range from the 5th to the 95th percentile. In this example, the 90% CI is (86, 94) kg.

It is important to remember that confidence intervals and standard errors only quantify sampling error; that is, error due to measuring only part of the population. The sampling distribution does not account for other sources of error, notably sampling bias and measurement error, which are the topics of the next section.

### Chebyshev’s Theorem

For any numerical data set, 

1. at least 3/4 of the data lie within two standard deviations of the mean
2. at least 8/9 of the data lie within three standard deviations of the mean
3. at least 1−1/k^2 of the data lie within k standard deviations of the mean

#### Common Distributions

**Normal Distribution**
$$
PDF_{normal}(x) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
$$


Two parameters: mu and sigma. Its CDF is defined by an integral that does not have a closed form solution, but there are algorithms that evaluate it efficiently.

For the exponential distribution, and a few others, there are simple transformations we can use to test whether an analytic distribution is a good model for a dataset. For the normal distribution there is no such transformation, but there is an alternative called a normal probability plot.

1. Sort the values in the sample. 
2. From a standard normal distribution (µ = 0 and σ = 1), generate a random sample with the same size as the sample, and sort it. 
3. Plot the sorted values from the sample versus the random values.

If the distribution of the sample is approximately normal, the result is a straight line with intercept mu and slope sigma.  (QQ plot).



A QQ-Plot is used to visually determine how close a sample is to the normal distribution. The QQ-Plot orders the z-scores from low to high, and plots each value’s z-score (individual point standardizing) on the y-axis; the x-axis is the corresponding quantile of a normal distribution for that value’s rank. If the points roughly fall on the diagonal line, then the sample distribution can be considered close to normal.

**Провекра распределения на нормальность**

1. QQ-plot
2. Тест Шапиро-Уилка
3. Тест Колмогорова-Смирнова

**Long-Tailed Distribution**

* Skew - one tail of distribution is longer than the other.

In the QQ-Plot, the points are far below the line for low values and far above the line for high values. **Most data is not normally distributed.**

**The lognormal distribution**

If the logarithms of a set of values have a normal distribution, the values have a lognormal distribution.
$$
CDF_{lognormal}(x) = CDF_{normal}(logx)
$$
If a sample is approximately lognormal and you plot its CDF on a log-x scale, it will have the characteristic shape of a normal distribution. To test how well the sample fits a lognormal model, you can make a normal probability plot using the log of the values in the sample.

**The Pareto distribution**
$$
PDF(x) = \frac{\alpha}{x^{\alpha + 1}}
\\
CDF(x) = 1 - (\frac{x}{x_m})^{-\alpha}
$$


The parameters xm and α determine the location and shape of the distribution. xm is the minimum possible value. There is a simple visual test that indicates whether an empirical distribution fits a Pareto distribution: on a log-log scale, the CCDF looks like a straight line. 

**Student's t-Distribution**

The t-distribution is a normally shaped distribution, but a bit thicker and longer on the tails. Distributions of sample means are typically shaped like a t-distribution, and there is a family of t-distributions that differ depending on how large the sample is. The larger the sample, the more normally shaped the t-distribution becomes.

The t-distribution is actually a family of distributions resembling the normal distribution, but with thicker tails. It is widely used as a reference basis for the distribution of sample means, differences between two sample means, regression parameters, and more.

Используется для проверки гипотез, если неизвестно стандартное отклонение генеральной совокупности, необходимое для расчета стандартной ошибки, даже если объем выборки больше 30.

**Binomial distribution**

* Trial - event with discrete outcome


The binomial distribution is the frequency distribution of the number of successes (x) in a given number of trials (n) with specified probability (p) of success in each trial.
$$
Mean = n*p
\\
Variance = n*p(1-p)
$$

With large n, and provided p is not too close to 0 or 1, the binomial distribution can be approximated by the normal distribution.

Binomial distribution:
$$
p_{X}(k) = P(X= k) = C_n^k p^k (1-p)^{n-k}   \text{ for k = 0, 1, 2, ..., n}
$$


**Poisson (Пуассона) and Related Distributions**

* Lambda - The rate (per unit of time or space) at which events occur.

* Poisson distribution - The frequency distribution of the number of events in sampled units of time or space.

* Exponential distribution - The frequency distribution of the time or distance from one event to the next event.

* Weibull distribution - A generalized version of the exponential, in which the event rate is allowed to shift over time


From prior data we can estimate the average number of events per unit of time or space, but we might also want to know how different this might be from one unit of time/space to another. The **Poisson distribution** tells us the distribution of events per unit of time or space when we sample many such units. Variance of Poisson distribution is lambda.

Using the same parameter lambda that we used in the Poisson distribution, we can also model the distribution of the time between events: time between visits to a website or between cars arriving at a toll plaza. **It is exponential distribution**:
$$
CDF(X) = 1 - e^{-\lambda x}
\\
PDF_{expo}(x) = \lambda e ^ {-\lambda x}
\\
mean = 1/\lambda
$$


Complementary CDF, which is 1 − CDF(x), on a log-y scale. If CDF is not exactly straight, its indicates that the exponential distribution is not a perfect model for this data.

The **Weibull distribution** is an extension of the exponential distribution, in which the event rate is allowed to change, as specified by a shape parameter, beta. If beta > 1, the probability of an event increases over time, if beta < 1, it decreases. This is likely to be the case in mechanical failure—the risk of failure increases as time goes by. Because the Weibull distribution is used with time-to-failure analysis instead of event rate, the second parameter is expressed in terms of characteristic life, rather than in terms of the rate of events per interval. The symbol used is the Greek letter eta. It is also called the scale parameter. With the Weibull, the estimation task now includes estimation of both parameters, beta and eta. 

* For events that occur at a constant rate, the number of events per unit of time or space can be modeled as a Poisson distribution. 

* In this scenario, you can also model the time or distance between one event and the next as an exponential distribution. 

* A changing event rate over time (e.g., an increasing probability of device failure) can be modeled with the Weibull distribution.


#### Determine Unknow Parameters of Normal Distribution

Now suppose that we have a data set of observations x = (x1,...,xN )^T, representing N observations of the scalar variable x. We shall suppose that the observations are drawn independently from a Gaussian distribution whose mean µ and variance σ2 are unknown, and we would like to determine these parameters from the data set. Data points that are drawn independently from the same distribution are said to be independent and identically distributed, which is often abbreviated to i.i.d. We have seen that the joint probability of two independent events is given by the product of the marginal probabilities for each event separately. Because our data set x is i.i.d., we can therefore write the probability of the data set (likelihood function), given µ and σ2, in the form
$$
p(\bold x|\mu \sigma^2) = \prod_{n=1}^N N(x_n|\mu \sigma^2)
$$
One common criterion for determining the parameters in a probability distribution using an observed data set is to find the parameter values that maximize the likelihood function. 

In practice, it is more convenient to maximize the log of the likelihood function. Because the logarithm is a monotonically increasing function of its argument, maximization of the log of a function is equivalent to maximization of the function itself. The log likelihood function:
$$
ln(p(\bold x | \mu \sigma^2)) = - \frac{1}{2\sigma^2} \sum_{n=1}^N (x_n - \mu)^2 - \frac{N}{2} ln(\sigma^2) - \frac{N}{2}ln(2\pi)
$$


Maximizing (1.54) with respect to µ and σ2, we obtain the maximum likelihood solution given by:
$$
\mu_{ML} = \frac{1}{N} \sum_{n=1}^N x_n
\\
\sigma^2_{ML} = \frac{1}{N} \sum_{n=1}^N(x_n - \mu_{ML})^2
$$

### Dissimilarity measures between two distributions



![](C:/Users/sqrte/python-playground/profile/img/dissimilarity.png)

The function supremum, sup(S), used in the total variation (TV) measure, refers to the smallest value that is greater than all elements of S. In other words, sup(S) is the least upper bound for S. Vice versa, the infimum function, inf(S), which is used in EM distance, refers to the largest value that is smaller than all elements of S. 

* The first one, TV distance, measures the largest difference between the two distributions at each point. 
* The EM distance can be interpreted as the minimal amount of work needed to transform one distribution into the other. The infimum function in the EM distance is taken over Π(P,Q), which is the collection of all joint distributions whose marginals are P or Q. Then, 𝛾𝛾(𝑢𝑢, 𝑣𝑣) is a transfer plan, which indicates how we redistribute the earth from location u to v, subject to some constraints for maintaining valid distributions after such transfers. Computing EM distance is an optimization problem by itself, which is to find the optimal transfer plan, 𝛾𝛾(𝑢𝑢, 𝑣𝑣). 
* The Kullback-Leibler (KL) and Jensen-Shannon (JS) divergence measures come from the field of information theory. Note that KL divergence is not symmetric, that is, 𝐾L(𝑃‖𝑄) ≠ KL(𝑄‖𝑃) in contrast to JS divergence.







## Statistical Experiments and Significance testing



### A/B Testing

An A/B test is an experiment with two groups to establish which of two treatments, products, procedures, or the like is superior. Often one of the two treatments is the standard existing treatment, or no treatment. If a standard (or no) treatment is used, it is called the control. A typical hypothesis is that treatment is better than control.

* Treatment - Something to which a subject is exposed.
* Treatment group - A group of subjects exposed to a specific treatments.
* Control group - A group of subjects exposed to no (or standard) treatment.
* Subjects - The items that are exposed to treatments.
* Test statistic - The metric used to measure the effect of treatment.

1. Subjects are assigned to two (or more) groups that are treated exactly alike, except that the treatment under study differs from one to another. 

2. Ideally, subjects are assigned randomly to the groups

### Hypothesis Test

https://medium.com/dataseries/hypothesis-testing-in-machine-learning-what-for-and-why-ad6ddf3d7af2

Hypothesis tests, also called significance tests, are ubiquitous in the traditional statistical analysis of published research. Their purpose is to help you learn whether random chance might be responsible for an observed effect.

* Null hypothesis - The hypothesis that chance is to blame. (Гипотеза о том, что разница между двумя группами возникает только благодаря шансу)
* Alternative hypothesis - Counterpoint to the null (what you hope to prove).
* One-way test - Hypothesis test that counts chance results only in one direction.
* Two-way test - Hypothesis test that counts chance results in two directions.

Hypothesis tests use the following logic: “Given the human tendency to react to unusual but random behavior and interpret it as something meaningful and real, in our experiments we will require proof that the difference between groups is more extreme than what chance might reasonably produce.” This involves a baseline assumption that the treatments are equivalent, and any difference between the groups is due to chance. This baseline assumption is termed the null hypothesis. Our hope is then that we can, in fact, prove the null hypothesis wrong, and show that the outcomes for groups A and B are more different than what chance might produce.

A null hypothesis is a logical construct embodying the notion that nothing special has happened, and any effect you observe is due to random chance. The hypothesis test assumes that the null hypothesis is true, creates a “null model” (a probability model), and tests whether the effect you observe is a reasonable outcome of that model.

The goal of classical hypothesis testing is to answer the question, “Given a sample and an apparent effect, what is the probability of seeing such an effect by chance?” Here’s how we answer that question:  

* The first step is to quantify the size of the apparent effect by choosing a test statistic. In the NSFG example, the apparent effect is a difference in pregnancy length between first babies and others, so a natural choice for the test statistic is the difference in means between the two groups. 
* The second step is to define a null hypothesis, which is a model of the system based on the assumption that the apparent effect is not real. In the NSFG example the null hypothesis is that there is no difference between first babies and others; that is, that pregnancy lengths for both groups have the same distribution. 
* The third step is to compute a p-value, which is the probability of seeing the apparent effect if the null hypothesis is true. In the NSFG example, we would compute the actual difference in means, then compute the probability of seeing a difference as big, or bigger, under the null hypothesis. 
* The last step is to interpret the result. If the p-value is low, the effect is said to be statistically significant, which means that it is unlikely to have occurred by chance. In that case we infer that the effect is more likely to appear in the larger population.

### Example: Flipping a Coin

Imagine we have a coin and we want to test whether it’s fair. We’ll make the assumption that the coin has some probability p of landing heads, and so our null hypothesis is that the coin is fair—that is, that p = 0.5. We’ll test this against the alternative hypothesis p ≠ 0.5. In particular, our test will involve flipping the coin some number, n, times and counting the number of heads, X. Each coin flip is a Bernoulli trial, which means that X is a Binomial(n,p) random variable, which we can approximate using the normal distribution:

```python
from typing import Tuple
import math
def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
	"""Returns mu and sigma corresponding to a Binomial(n, p)"""
	mu = p * n
	sigma = math.sqrt(p * (1 - p) / n)
	return mu, sigma
```

Whenever a random variable follows a normal distribution, we can use normal_cdf to figure out the probability that its realized value lies within or outside a particular interval:

```python
def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma) / 2
            
normal_probability_below = normal_cdf
# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float,
	mu: float = 0,
	sigma: float = 1) -> float:
	"""The probability that an N(mu, sigma) is greater than lo."""
	return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than lo
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
	"""The probability that an N(mu, sigma) is between lo and hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if it's not between
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
	"""The probability that an N(mu, sigma) is not between lo and hi."""
	return 1 - normal_probability_between(lo, hi, mu, sigma)
```

We can also do the reverse—find either the nontail region or the (symmetric) interval around the mean that accounts for a certain level of likelihood. For example, if we want to find an interval centered at the mean and containing 60% probability, then we find the cutoffs where the upper and lower tails each contain 20% of the probability (leaving 60%):

```python
def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
	"""Returns the z for which P(Z <= z) = probability"""
	return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
	"""Returns the z for which P(Z >= z) = probability"""
	return inverse_normal_cdf(1 - probability, mu, sigma)
def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
	"""
	Returns the symmetric (about the mean) bounds
	that contain the specified probability
	"""
	tail_probability = (1 - probability) / 2
	# upper bound should have tail_probability above it
	upper_bound = normal_lower_bound(tail_probability, mu, sigma)
	# lower bound should have tail_probability below it
	lower_bound = normal_upper_bound(tail_probability, mu, sigma)
	return lower_bound, upper_bound
```

In particular, let’s say that we choose to flip the coin n = 1,000 times. If our hypothesis of fairness is true, X should be distributed approximately normally with mean 500 and standard deviation 15.8:

```python
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
```

We need to make a decision about significance—how willing we are to make a type 1 error (“false positive”), in which we reject H0 even though it’s true. For reasons lost to the annals of history, this willingness is often set at 5% or 1%. Let’s choose 5%. Consider the test that rejects H0 if X falls outside the bounds given by:

```python
# (469, 531)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
```

Assuming p really equals 0.5 (i.e., H0 is true), there is just a 5% chance we observe an X that lies outside this interval, which is the exact significance we wanted. Said differently, if H0 is true, then, approximately 19 times out of 20, this test will give the correct result.

We are also often interested in the power of a test, which is the probability of not making a type 2 error (“false negative”), in which we fail to reject H0 even though it’s false. In order to measure this, we have to specify what exactly H0 being false means. (Knowing merely that p is not 0.5 doesn’t give us a ton of information about the distribution of X.) In particular, let’s check what happens if p is really 0.55, so that the coin is slightly biased toward heads.

```python
# 95% bounds based on assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
# actual mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
# a type 2 error means we fail to reject the null hypothesis,
# which will happen when X is still in our original interval
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.887
```

Imagine instead that our null hypothesis was that the coin is not biased toward heads, or that p ≤ 0. 5. In that case we want a one-sided test that rejects the null hypothesis when X is much larger than 500 but not when X is smaller than 500. So, a 5% significance test involves using normal_probability_below to find the cutoff below which 95% of the probability lies:

```python
hi = normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)
type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.936
```

This is a more powerful test, since it no longer rejects H0 when X is below 469 (which is very unlikely to happen if H1 is true) and instead rejects H0 when X is between 526 and 531 (which is somewhat likely to happen if H1 is true).



### Example: Running an A/B Test

Being a scientist, you decide to run an experiment by randomly showing site visitors one of the two advertisements and tracking how many people click on each one. If 990 out of 1,000 A-viewers click their ad, while only 10 out of 1,000 B-viewers click their ad, you can be pretty confident that A is the better ad. But what if the differences are not so stark? Here’s where you’d use statistical inference.

Let’s say that NA people see ad A, and that nA of them click it. We can think of each ad view as a Bernoulli trial where pA is the probability that someone clicks ad A. Then (if NA is large, which it is here) we know that nA/NA is approximately a normal random variable with mean pA and standard deviation σA = √pA (1 − pA) /NA. 

Similarly, nB/NB is approximately a normal random variable with mean pB and standard deviation σB = √pB (1 − pB) /NB. We can express this in code as:

```python
def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
 p = n / N
 sigma = math.sqrt(p * (1 - p) / N)
 return p, sigma
```

If we assume those two normals are independent (which seems reasonable, since the individual Bernoulli trials ought to be), then their difference should also be normal with mean pB − pA and standard deviation √σ 2 A + σ 2 B .

**Note:** This is sort of cheating. The math only works out exactly like this if you know the standard deviations. Here we’re estimating them from the data, which means that we really should be using a t-distribution. But for large enough datasets, it’s close enough that it doesn’t make much of a difference.

This means we can test the null hypothesis that pA and pB are the same (that is, that pA − pB is 0) by using the statistic:

```python
def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
 p_A, sigma_A = estimated_parameters(N_A, n_A)
 p_B, sigma_B = estimated_parameters(N_B, n_B)
 return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)
```

which should approximately be a standard normal. For example, if “tastes great” gets 200 clicks out of 1,000 views and “less bias” gets 180 clicks out of 1,000 views, the statistic equals and the probability of seeing such a large difference if the means were actually equal would be:

```python
z = a_b_test_statistic(1000, 200, 1000, 180) # -1.14
two_sided_p_value(z) # 0.254 - probability
```





### **P-value**

An alternative way of thinking about the preceding test involves p-values. Instead of choosing bounds based on some probability cutoff, we compute the probability—assuming H0 is true—that we would see a value at least as extreme as the one we actually observed.

For our two-sided test of whether the coin is fair, we compute:

```python
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	How likely are we to see a value at least as extreme as x (in either
	direction) if our values are from an N(mu, sigma)?
	"""
	if x >= mu:
		# x is greater than the mean, so the tail is everything greater than x
        return 2 * normal_probability_above(x, mu, sigma)
	else:
        # x is less than the mean, so the tail is everything less than x
		return 2 * normal_probability_below(x, mu, sigma)
```

If we were to see 530 heads, we would compute:

```python
two_sided_p_value(529.5, mu_0, sigma_0) # 0.062
```

Why did we use a value of 529.5 rather than using 530? This is what’s called a continuity correction. It reflects the fact that normal_probability_between(529.5, 530.5, mu_0, sigma_0) is a better estimate of the probability of seeing 530 heads than normal_probability_between(530, 531, mu_0, sigma_0) is. Correspondingly, normal_probability_above(529.5, mu_0, sigma_0) is a better estimate of the probability of seeing at least 530 heads.

One way to convince yourself that this is a sensible estimate is with a simulation:

```python
import random
extreme_value_count = 0
for _ in range(1000):
	num_heads = sum(1 if random.random() < 0.5 else 0 # Count # of heads
                    for _ in range(1000)) # in 1000 flips,
	
    if num_heads >= 530 or num_heads <= 470: # and count how often
        extreme_value_count += 1 # the # is 'extreme'
# p-value was 0.062 => ~62 extreme values out of 1000
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"
```

Since the p-value is greater than our 5% significance, we don’t reject the null. If we instead saw 532 heads, the p-value would be:

```python
two_sided_p_value(531.5, mu_0, sigma_0) # 0.0463
```

which is smaller than the 5% significance, which means we would reject the null. It’s the exact same test as before. It’s just a different way of approaching the statistics.

**P-hacking**

What this means is that if you’re setting out to find “significant” results, you usually can. Test enough hypotheses against your dataset, and one of them will almost certainly appear significant. Remove the right outliers, and you can probably get your p-value below 0.05.

If you want to do good science, you should determine your hypotheses before looking at the data, you should clean your data without the hypotheses in mind, and you should keep in mind that p-values are not substitutes for common sense. 



### Resampling

Resampling in statistics means to repeatedly sample values from observed data, with a general goal of assessing random variability in a statistic. There are two main types of resampling procedures: the **bootstrap** and **permutation tests**. The bootstrap is used to assess the reliability of an estimate; it was discussed in the previous chapter. 

* Permutation tests - The procedure of combining two or more samples together, and randomly (or exhaustively) reallocating the observations to resamples.

* With or without replacement  - In sampling, whether or not an item is returned to the sample before the next draw.


Permutation tests are used to test hypotheses, typically involving two or more groups. Permute means to change the order of a set of values. The first step in a permutation test of a hypothesis is to combine the results from groups A and B (and, if used, C, D, …) together. We then test that hypothesis by randomly drawing groups from this combined set, and seeing how much they differ from one another. The permutation procedure is as follows:

1. Combine the results from the different groups in a single data set. 
2. Shuffle the combined data, then randomly draw (without replacing) a resample of the same size as group A.
3. From the remaining data, randomly draw (without replacing) a resample of the same size as group B. 
4. Do the same for groups C, D, and so on. 
5. Whatever statistic or estimate was calculated for the original samples (e.g., difference in group proportions), calculate it now for the resamples, and record; this constitutes one permutation iteration. 
6. Repeat the previous steps R times to yield a permutation distribution of the test statistic.

Now go back to the observed difference between groups and compare it to the set of permuted differences. If the observed difference lies well within the set of permuted differences, then we have not proven anything—the observed difference is within the range of what chance might produce.

In addition to the preceding random shuffling procedure, also called a random permutation test or a randomization test, there are two variants of the permutation test:

1. An exhaustive permutation test. 

2. A bootstrap permutation test.


In an exhaustive permutation test, instead of just randomly shuffling and dividing the data, we actually figure out all the possible ways it could be divided. This is practical only for relatively small sample sizes.

In a bootstrap permutation test, the draws outlined in steps 2 and 3 of the random permutation test are made with replacement instead of without replacement.

### Statistical Significance and P-Values

* P-Value (Достигаемый уровень значимости) - Given a chance model that embodies the null hypothesis, the p-value is the probability of obtaining results as unusual or extreme as the observed results (Вероятность при справедливости нулевой гипотезы получить значение статистики как в эксперементе или еще более экстремальное, чем меньше p-value, тем сильнее данные свидетельствуют против нулевой гипотезы в пользу альтернативе)
* Alpha - The probability threshold of “unusualness” that chance results must surpass, for actual outcomes to be deemed statistically significant. (Если P-value < alpha нулевая гипотеза отвергается)
* Type 1 error - Mistakenly concluding an effect is real (when it is due to chance). (Отвержение нулевой гипотезы когда она верна).
* Type 2 error - Mistakenly concluding an effect is due to chance (when it is real) (Принятие нулевой гипотезы когда она не верна).

The real problem is that people want more meaning from the p-value than it contains. Here’s what we would like the p-value to convey: 

*The probability that the result is due to chance.*

We hope for a low value, so we can conclude that we’ve proved something. This is how many journal editors were interpreting the p-value. But here’s what the p-value actually represents: 

*The probability that, given a chance model, results as extreme as the observed results could occur.*

**The ASA statement stressed six principles for researchers and journal editors: **

1. P-values can indicate how incompatible the data are with a specified statistical model. 
2. P-values do not measure the probability that the studied hypothesis is true, or the probability that the data were produced by random chance alone. 
3. Scientific conclusions and business or policy decisions should not be based only on whether a p-value passes a specific threshold. 
4. Proper inference requires full reporting and transparency. 
5. A p-value, or statistical significance, does not measure the size of an effect or the importance of a result. 
6. By itself, a p-value does not provide a good measure of evidence regarding a model or hypothesis.

For a data scientist, a p-value is a useful metric in situations where you want to know whether a model result that appears interesting and useful is within the range of normal chance variability. As a decision tool in an experiment, a p-value should not be considered controlling, but merely another point of information bearing on a decision. For example, p-values are sometimes used as intermediate inputs in some statistical or machine learning models—a feature night be included in or excluded from a model depending on its p-value.

### t-Tests

* Test statistic - A metric for the difference or effect of interest. 
* t-statistic - A standardized version of test statistic.
* t-distribution - A reference distribution which the observed t-statistic can be compared.

A test statistic could then be standardized and compared to the reference distribution.

One such widely used standardized statistic is the t-statistic.

**One sample**

Tests whether the mean of a normally distributed population is different from a specified value.

*Null hypothesis (H_0)*: states that the population mean is equal to some value (mu_0).

*Alternative hypothesis (H_1)*: states that the population mean is not equal/is greater than/is less than value (mu_0).

t-statistic: standardized the difference between mean(X) and mu_0
$$
t = \frac{\bar x - \mu_0}{\frac{s}{\sqrt{n{}}}}
$$


Read the table of t-distribution critical values for the p-value using calculated t-statistic and degrees of freedom.

H_a: mu > mu_0 --> the t-statistic is likely positive; read tables as given

H_a: mu < mu_0 --> the t-statistic is likely negative; the t-distribution is symmetrical so read probability as if the t-statistic were positive

Note: if the t-statistic is of the wrong sign, the p-value is 1 minus p given in the chart

H_a: mu != mu_0 --> read the p-value as if the t-statistic were positive and double it

If the p-value is less than the predetermined value for significance, reject the null hypothesis and accept alternative hypothesis.

**Example**

You are experiencing hair loss and skin discoloration and think it might be because of selenium toxicity. You decide to measure the selenium levels in your tap water once a day for one week. Your results are given below. The EPA maximum containment level for safe drinking water is 0.05mg/L. Does the selenium level in your tap water exceed the legal limit?

| Day  | Selenium mg/L |
| ---- | ------------- |
| 1    | 0.051         |
| 2    | 0.0505        |
| 3    | 0.049         |
| 4    | 0.0516        |
| 5    | 0.052         |
| 6    | 0.0508        |
| 7    | 0.0506        |

H_0: mu = 0.05; H_a: mu > 0.05;

Calculate the mean and standard deviation of your sample:

mean(x) = 0.0508

std(x) = 9.56 * 1e-4
$$
t = \frac{\bar x - \mu_0}{\frac{s}{\sqrt{n{}}}} = \frac{0.0508 - 0.05}{\frac{9.56 * 10^{-4}}{\sqrt{7}}} = 2.17
$$


And degrees of freedom are n-1 = 7-1 =6

Looking at the t-distribution of critical values table, 2.17 with 6 degrees of freedom is between p=0.05 and p=0.025. This means that the p-value is less than 0.05, so you can reject H_0 and conclude that the selenium level in your tap water exceed legal limit.

**Two sample**

Tests whether the means of two populations are significantly different from one another.

*Paired* - Each value of one group corresponds directly to a value in the other group; ie. before and after values after drug treatment for each individual patient.

Subtract the two values for each individual to get one set of values (the differences) and use mu_0 = 0 to perform a one-sample t-test

*Unpaired* - The two populations are independent.

H_0: states that the means of the two populations are equal.

H_a: states that the means of the two populations are unequal or one is greater than the other

t-statistic: (1) - assuming equal variances, (2) - assuming unequal variances:
$$
t = \frac{\bar x_1 - \bar x_2}{\sqrt{s^2(\frac{1}{n_1}+\frac{1}{n_2})}} (1)
\\
t = \frac{\bar x_1 - \bar x_2}{\sqrt{(\frac{s^2_1}{n_1}+\frac{s^2_2}{n_2})}} (2)
$$


degrees of freedom = (n_1 - 1) + (n_2 - 1)

Read the table of t-distribution critical values for the p-value using the calculated t-statistic and degrees of freedom. Remember to keep the sign of the t-statistic clear (order or subtracting the sample means) and to double the p-value for an H_a of mu_1 != mu_2.

**Example**

Consider the lifespan of 18 rats. 12 were fed a restricted calorie diet and lived an average 700 days (std = 21 days). The other 6 had unrestricted access to food and lived an average 668 days (std = 30 days). Does a restricted calorie diet increase the lifespan of rats?

H_0: mu_1 = mu_2

H_a: mu_1 > mu_2

We cannot assume that the variance of the two populations are equal because the different diets could also affect the variability in lifespan.
$$
t = \frac{\bar x_1 - \bar x_2}{\sqrt{(\frac{s^2_1}{n_1}+\frac{s^2_2}{n_2})}} = \frac{700 - 668}{\sqrt{(\frac{21^2}{12}+\frac{30^2}{6})}} = 2.342
\\
df = (n_1 - 1) + (n_2 - 1) = 16
$$


From the t-distribution table, the p-value falls between 0.01 and 0.02, so we do reject H_0. The restricted calorie diet does increase the lifespan of rats.

### Однофакторный дисперсионный анализ

Представим набор данных из трех групп:

| a    | b    | c    |
| ---- | ---- | ---- |
| 3    | 5    | 7    |
| 1    | 3    | 6    |
| 2    | 4    | 5    |

H_0: средние трех групп равны

H_a: средние групп не равны



Рассмотрим все наблюдения как одну выборку, посчитаем среднее:
$$
\bar{\bar x} = \frac{3+1+2+5+3+4+7+6+5}{9} = 4
$$
Рассчитаем:
$$
SST = \sum_i (x_i - \bar{\bar x})^2 = (3-4)^2 + (1-4)^2 + (2-4)^2 + (5 - 4)^2 + (4 - 4)^2 + (3-4)^2 + (7-4)^2 + (6-4)^2 + (5-4)^2 =30
\\
SST = SSB + SSW
\\
df = 9 - 1
\\
x_a = 2, x_b= 4, x_c = 6
\\
SSW = \sum_i \sum_j (x_{ij} - \bar x_i)  = (3-2)^2 + (1-2)^2 + (2-2)^2 + \\ + (5-4)^2 + (3-4)^2 + (4-4)^2 + (7 - 6)^2 + (6 -6 )^2 + (5 - 6)^2 = 6
\\
df = N - m = 9 - 3 = 6
\\
SSB = \sum_i n_i(\bar x_i - \bar{\bar x}) = 3(2-4)^2 + 3(4-4)^2+(6-4)^2 = 24
\\
df = m - 1 = 3 - 1 = 2
$$


где SST - total sum of squares, показатель который характеризует изменчивость данных без учета разбиения их на группы. 

SSB - сумма квадратов между групп. SSW - сумма квадратов внутри групп.

Если большая часть общей изменчивости обеспечивается благодаря сумме квадратов между групп это значит что группы значительно различаются между собой. 

Основной статистический показатель дисперсионного анализа:
$$
F = \frac{\frac{SSB}{m-1}}{\frac{SSW}{N - m}} = 12
$$


Если выборки не отличаются между собой то числитель дроби стремится к нулю.



**Множественные сравнения**

Дисперсионный анализ показывает что между хотя бы две группы отличаются. Для того чтобы узнать какие можно провести между всеми группами t-test, однако с ростом числа групп растет и вероятность того, что тест покажет различия в группах, даже если его нет (при фиксированном пороге p-value). Для того чтобы исправить это используется **поправка Бонферонни**, при которой порог p-value делится на число междугрупповых сравнений, и соответственно уровень значимости сравнивается с этим новым порогом.

Более корректным способом является использование **поправки Тьюки**.

Без поправок возрастает вероятность совершить ошибку первого рода.



### Двухфакторный анализ дисперсии

SST = SSW + SSB_A + SSB_B + SSB_A*SSB_B 

### Multiple Testing

* False discovery rate - Across multiple tests, the rate of making a Type 1 error.
* Adjustment of p-values - Accounting for doing multiple tests on the same data.

Multiplicity in a research study or data mining project (multiple comparisons, many variables, many models, etc.) increases the risk of concluding that something is significant just by chance. For situations involving multiple statistical comparisons (i.e., multiple tests of significance) there are statistical adjustment procedures. In a data mining situation, use of a holdout sample with labeled outcome variables can help avoid misleading results.

### ANOVA 

The statistical procedure that tests for a statistically significant difference among the groups is called analysis of variance, or ANOVA

* Pairwise comparison - A hypothesis test (e.g., of means) between two groups among multiple groups. 
* Omnibus test - A single hypothesis test of the overall variance among multiple group means. 
* Decomposition of variance - Separation of components. contributing to an individual value (e.g., from the overall average, from a treatment mean, and from a residual error). 
* F-statistic - A standardized statistic that measures the extent to which differences among group means exceeds what might be expected in a chance model.
* SS - “Sum of squares,” referring to deviations from some average value.

“Could all the pages have the same underlying stickiness, and the differences among them be due to the random way in which a common set of session times got allocated among the four pages?” 

The procedure used to test this is ANOVA. The basis for ANOVA can be seen in the following resampling procedure (specified here for the A-B-C-D test of web page stickiness): 

1. Combine all the data together in a single box.
2. Shuffle and draw out four resamples of five values each.
3. Record the mean of each of the four groups.
4. Record the variance among the four group means.
5. Repeat steps 2–4 many times (say 1,000)

What proportion of the time did the resampled variance exceed the observed variance? This is the p-value.

The **F-statistic** is based on the ratio of the variance across group means (i.e., the treatment effect) to the variance due to residual error.

ANOVA is a statistical procedure for analyzing the results of an experiment with multiple groups. It is the extension of similar procedures for the A/B test, used to assess whether the overall variation among groups is within the range of chance variation. A useful outcome of an ANOVA is the identification of variance components associated with group treatments, interaction effects, and errors.

### Chi-Square Test

**For goodness of Fit**

Checks whether or not an observed pattern of data fits some given distribution

H_0: The observed pattern fits the given distribution

H_a: The observed pattern does not fit the given distribution

The chi-square statistic is:
$$
\chi^2= \sum \frac{(O-E)^2}{E}
$$


O is observed value and E is expected;

Degrees of freedom = number of categories in the distribution - 1

Get the p-value from the table of \chi^2 critical value using the calculated \chi^2 and df values. If the p-value is less then \alpha, the observed data does not fit the expected distribution. If p>\alpha, the data likely fits the expected distribution.

**Example**

You breed puffskeins and would like to determine the pattern of inheritance for coat colar and purring ability. Puffskeins come in either pink or purple and can either purr or hiss. You breed a purebred, pink purring male with a purebred, purple hissing female. All individuals of the F1 generation are pink and purring. The F2 offspring a shown below. Do the alleles for coat color and purring ability assort independently?

| Pink and purring | Pink and Hissing | Purple and Purring | Purple and Hissing |
| ---------------- | ---------------- | ------------------ | ------------------ |
| 143              | 60               | 55                 | 18                 |

Independent assortment means a phenotypic ratio of 9:3:3:1, so:

H_0: the observed distribution of F2 offspring fits 9:3:3:1 distribution

H_a: the observed distribution of F2 offspring does not fit a 9:3:3:1 distribution

The expected values are:

| Pink and purring | Pink and Hissing | Purple and Purring | Purple and Hissing |
| ---------------- | ---------------- | ------------------ | ------------------ |
| 155.25           | 51.75            | 51.75              | 17.25              |

$$
\chi^2 = \sum \frac{(O - E)^2}{E} = 2.519
\\
df = 4-1= 3
$$



From the table of \chi^2 critical values, the p-value is greater than 0.25, so the alleles for coat color and purring ability do assort independently in puffskeins.

**For independence**

Checks whether two categorical variables are related or not 

H_0: the two variables are independent

H_a: the two variables are not independent

*Does not make any assumptions about an expected distribution*

The observed values (#1, #2, #3, #4) are usually presented as a table. Each row is a category of variable 1 and each column is a category of variable 2.

|            |            | Variable 1 |            | Totals            |
| ---------- | ---------- | ---------- | ---------- | ----------------- |
|            |            | Category x | Category y |                   |
| Variable 2 | Category a | #1         | #2         | #1 + #2           |
|            | Category b | #3         | #4         | #3 + #4           |
| Totals     |            | #1 + #3    | #2 + #4    | #1 + #2 + #3 + #4 |

The proportion of category x of variable 1 is the number of individuals in category x divided by total number of individuals. Assuming independence, the expected number of individuals that fall within category a of variable 2 is the proportion of category x multiplied by the number of individuals in category a. Thus the expected value:
$$
E = \frac{(rowtotal)(columntotal)}{grandtotal}
\\
df = (r-1)(c-1)
\\
\chi^2 = \sum \frac{(O - E)^2}{E}
$$
**Example**

Given example below, is there a relationship between fitness level and smoking habits?

|                         | Fitness level |            |             |      |      |
| ----------------------- | ------------- | ---------- | ----------- | ---- | ---- |
|                         | Low           | Medium-low | Medium-High | High |      |
| Never smoked            | 113           | 113        | 110         | 159  | 495  |
| Former smokers          | 119           | 135        | 172         | 190  | 616  |
| 1 to 9 cigarettes daily | 77            | 91         | 86          | 65   | 319  |
| > 9 cigarettes daily    | 181           | 152        | 124         | 73   | 530  |
|                         | 490           | 491        | 492         | 487  | 1960 |

H_0: Fitness level and smoking habits are independent

H_a: Fitness level and smoking habits are not independent

First we calculate the expected counts. For the first cell:
$$
E = \frac{(row total) (columntotal)}{grand total} = \frac{495*490}{1960} = 123.75
$$

|                         | Fitness level |            |             |        |
| ----------------------- | ------------- | ---------- | ----------- | ------ |
|                         | Low           | Medium-low | Medium-High | High   |
| Never smoked            | 123.75        | 124        | 124.26      | 122.99 |
| Former smokers          | 154           | 154.31     | 154.63      | 153.06 |
| 1 to 9 cigarettes daily | 79.75         | 79.91      | 80.08       | 79.26  |
| > 9 cigarettes daily    | 132.5         | 132.77     | 133.04      | 131.69 |

$$
\chi^2 = \sum \frac{(O-E)^2}{E} = 91.73
\\
df = (r-1)(c-1) = (4-1)(4-1)=9
$$



From the table of \chi^2 critical values, the p-value is less than 0.001, so we reject H_0 and conclude that there is a relationship between fitness level and smoking habits.



Example: https://r-analytics.blogspot.com/2012/08/blog-post.html

В общем виде можно сказать, что он используется для проверки нулевой гипотезы о подчинении наблюдаемой случайной величины определенному теоретическому закону распределения.

The chi-square test is used with count data to test how well it fits some expected distribution. The most common use of the chi-square statistic in statistical practice is with r*c (rows by columns) contingency tables, to assess whether the null hypothesis of independence among variables is reasonable


$$
R = \frac{Observed-Expected}{\sqrt{Expected}}
\\
chi = \sum_{i}^{r}\sum_{j}^{c}R^2
$$


Suppose you are testing three different headlines—A (14 clicks), B (8 clicks), and C(12 clicks) —and you run them each on 1,000 visitors. For this test, we need to have the “expected” distribution of clicks, and, in this case, that would be under the null hypothesis assumption that all three headlines share the same click rate, for an overall click rate of 34/3,000.

1. Constitute a box with 34 ones (clicks) and 2,966 zeros (no clicks). 
2. Shuffle, take three separate samples of 1,000, and count the clicks in each. 
3. Find the squared differences between the shuffled counts and the expected counts, and sum them. 
4. Repeat steps 2 and 3, say, 1,000 times. 
5. How often does the resampled sum of squared deviations exceed the observed? That’s the p-value.

A common procedure in statistics is to test whether observed data counts are consistent with an assumption of independence (e.g., propensity to buy a particular item is independent of gender). The chi-square distribution is the reference distribution (which embodies the assumption of independence) to which the observed calculated chi-square statistic must be compared.







### Multi-Arm Bandit Algorithm

* Multi-arm Bandit - An imaginary slot machine with multiple arms for the customer to choose from, each with different payoffs, here taken to be an analogy for a multi-treatment experiment.
* Arm - A treatment in an experiment 
* Win - The experimental analog of a win at the slot machine 

Bandit algorithms, which are very popular in web testing, allow you to test multiple treatments at once and reach conclusions faster than traditional statistical designs.

Traditional A/B tests envision a random sampling process, which can lead to excessive exposure to the inferior treatment. Multi-arm bandits, in contrast, alter the sampling process to incorporate information learned during the experiment and reduce the frequency of the inferior treatment. They also facilitate efficient treatment of more than two treatments. There are different algorithms for shifting sampling probability away from the inferior treatment(s) and to the (presumed) superior one.

Here is one simple algorithm, the epsilon-greedy algorithm for an A/B test: 

1. Generate a random number between 0 and 1. 
2. If the number lies between 0 and epsilon (where epsilon is a number between 0 and 1, typically fairly small), flip a fair coin (50/50 probability), and: a. If the coin is heads, show offer A. b. If the coin is tails, show offer B. 
3. If the number is ≥ epsilon, show whichever offer has had the highest response rate to date.



### Power and Sample Size

* Effect size - The minimum size of the effect that you hope to be able to detect in a statistical test, such as "a 20% improvements in click rates".
* Power - The probability of detecting a given size with a given sample size.
* Significance level - The statistical significance level at which the test will be conducted.

Here’s a fairly intuitive alternative approach: 

1. Start with some hypothetical data that represents your best guess about the data that will result (perhaps based on prior data)—for example, a box with 20 ones and 80 zeros to represent a .200 hitter, or a box with some observations of “time spent on website.” 
2. Create a second sample simply by adding the desired effect size to the first sample—for example, a second box with 33 ones and 67 zeros, or a second box with 25 seconds added to each initial “time spent on website.” 
3. Draw a bootstrap sample of size n from each box. 
4. Conduct a permutation (or formula-based) hypothesis test on the two bootstrap samples and record whether the difference between them is statistically significant. 
5. Repeat the preceding two steps many times and determine how often the difference was significant—that’s the estimated power.

Finding out how big a sample size you need requires thinking ahead to the statistical test you plan to conduct. You must specify the minimum size of the effect that you want to detect. You must also specify the required probability of detecting that effect size (power). Finally, you must specify the significance level (alpha) at which the test will be conducted.





### Probability



1. A probability function P on a finite sample space Ω assigns to each event A in Ω a number P(A) in [0,1] such that:
   $$
   P(\Omega) = 1
   \\
   P(A \cup B) = P(A) + P(B) \text{ - (any of events) if A and B disjoint}
   \\
   P(A \cup B) = P(A) + P(B) - P(A \cap B) \text{ - for any two events A and B}
   $$

2. The conditional probability of A given C is given by:
   $$
   P(A|C) = \frac{P(A\cap C)}{P(C)}
   \\
   P(A \cap C) = P(A|C) P(C)
   $$

3. **The law of total probability**. Suppose C1, C2, ..., Cm are disjoint events such that C1 ∪ C2 ∪···∪ Cm = Ω. The probability of an arbitrary event A can be expressed as:
   $$
   P(A) = P(A|C_1)P(C_1)+P(A|C_2)P(C_2) + ... + P(A|C_m)P(C_m)
   $$
   **Example: Testing for mad cow disease**

   Let B denote the event “the cow has BSE” and T the event “the test comes up positive”.  An infected cow has a 70% chance of testing positive, and a healthy cow just 10%. 
   $$
   P(T|B) = 0.70
   \\
   P(T|\neg B) = 0.10
   \\
   P(B) = 0.02
   $$
   *Suppose we want to determine the probability P(T) that an arbitrary cow tests positive*. The tested cow is either infected or it is not: event T occurs in combination with B or with Bc (there are no other possibilities). In terms of events:
   $$
   T= (T \cap B)\cup(T \cap \neg B)
   \\
   P(T) = P(T \cap B) + P(T \cap \neg B)
   $$
   because T ∩B and T ∩Bc are disjoint. Next, apply the multiplication rule (in such a way that the known conditional probabilities appear!):
   $$
   P(T\cap B) = P(T|B)P(B)
   \\
   P(T \cap \neg B) = P(T| \neg B) P(\neg B)
   \\
   P(T) = P(T|B)P(B) + P(T| \neg B) P(\neg B) = 0.02 * 0.7 + (1 - 0.02)*0.1 = 0.112
   \\
   $$


   Another, perhaps more pertinent, question about the BSE test is the following: *suppose my cow tests positive; what is the probability it really has BSE*?
$$
   P(B|T) = \frac{P(T \cap B)}{P(T)} = \frac{P(T|B)P(B)}{P(T|B)P(B) + P(T|\neg B)P(\neg B)} = \frac{0.7*0.02}{0.7*0.02+ 0.1*(1-0.02)} = 0.125
$$

4. **Bayes’ rule**. Suppose the events C1, C2, ..., Cm are disjoint and C1 ∪ C2 ∪···∪ Cm = Ω. The conditional probability of Ci, given an arbitrary event A, can be expressed as:
   $$
   P(C_i|A) = \frac {P(A|C_i)P(C_i)}{P(A|C_1)P(C_1)+ P(A|C_2)P(C_2) + ... + P(A|C_m)P(C_m)}
   $$

5. An event A is called **independent** of B if:
   $$
   P(A|B)=P(A)
   \\
   P(B|A) = P(B)
   \\
   P(A \cap B) = P(A)P(B)
   $$
   To show that A and B are independent it suffices to prove just one of the statements.






### Central limit theorem

https://www.youtube.com/watch?v=YAlJCEDH2uY

For a large number of independent identically distributed random variables X1,...,Xn, with finite variance, the average X¯n approximately has a normal distribution, no matter what the distribution of the Xi is.

One reason the normal distribution is so useful is the central limit theorem, which says (in essence) that a random variable defined as the average of a large number of independent and identically distributed random variables is itself approximately normally distributed.

In particular, if x1, ..., xn are random variables with mean μ and standard deviation σ, and if n is large, then:
$$
\frac{1}{n}(x_1 + ... + x_n)
$$
is approximately normally distributed with mean μ and standard deviation σ/√n. Equivalently (but often more usefully),
$$
\frac{(x_1 + ... x_n) - \mu n}{\sigma \sqrt{n}}
$$


is approximately normally distributed with mean 0 and standard deviation 1.

An easy way to illustrate this is by looking at binomial random variables, which have two parameters n and p. A Binomial(n,p) random variable is simply the sum of n independent Bernoulli(p) random variables, each of which equals 1 with probability p and 0 with probability 1 – p. The mean of a Bernoulli(p) variable is p, and its standard deviation is √p(1 − p). The central limit theorem says that as n gets large, a Binomial(n,p) variable is approximately a normal random variable with mean μ = np and standard deviation σ = √np(1 − p). 

The moral of this approximation is that if you want to know the probability that (say) a fair coin turns up more than 60 heads in 100 flips, you can estimate it as the probability that a Normal(50,5) is greater than 60, which is easier than computing the Binomial(100,0.5) CDF. (Although in most applications you’d probably be using statistical software that would gladly compute whatever probabilities you want.)



Средние значения подвыборок из неизвестного распределения при увеличении количества таких подвыборок образуют нормальное распределение.

### **Bayesian curve fitting**

In the curve fitting problem, we are given the training data **x** and **t**, along with a new test point x, and our goal is to predict the value of t. We therefore wish to evaluate the predictive distribution p(t|x, **x**, **t**). Here we shall assume that the parameters α and β are fixed and known in advance.

A Bayesian treatment simply corresponds to a consistent application of the sum and product rules of probability, which allow the predictive distribution to be written in the form:
$$
p(t|x, \bold x, \bold t) = \int p(t|x, \bold w) p(w|\bold x, \bold t) dw
$$


### Curse of dimensionality

In practice, to capture complex dependencies in the data, we may need to use a higher-order polynomial. For a polynomial of order M, the growth in the number of coefficients is like D^M. Although this is now a power law growth, rather than an exponential growth, it still points to the method becoming rapidly unwieldy and of limited practical utility

The reader should be warned, however, that not all intuitions developed in spaces of low dimensionality will generalize to spaces of many dimensions.