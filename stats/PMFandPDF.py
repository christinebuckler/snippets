
from scipy import stats # https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html
import matplotlib.pyplot as plt

# probability density function

# The SciPy implementation of the gamma distribution uses the shape and scale parameterization.
a=5
gamma_rv = stats.gamma(a, scale=10) # create gamma distribution
print gamma_rv.pdf(10) # evaluate the density of the gamma distribution for a given outcome 0.0015328310048810102

# how to plot gamma pdf?
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html




# probability mass function

# scipy.stats.distributions.poisson.pmf(x, poissonLambda)
# This calculates the probability that there are x events in an interval,
# where the argument "poissonLambda" is the average number of events per interval

mu=20
poisson_rv = stats.poisson(mu) # create analogous(?) poisson distribution
print poisson_rv.pmf(10) # consider what method of a poisson_rv object you should be calling

from scipy.stats import poisson
import numpy as np
fig, ax = plt.subplots(1, 1)
# mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
x = np.arange(poisson.ppf(0.01, mu),poisson.ppf(0.99, mu))
ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf') # dots
ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5) # lines
plt.show()

# What is the biggest difference between these values as associated with the gamma distribution compared to the Poisson distribution?
