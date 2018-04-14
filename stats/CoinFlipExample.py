# Simulating Data
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
n = 100 # num of trials
pcoin = 0.62 # actual value of p for coin
results = st.bernoulli(pcoin).rvs(n)
h = sum(results) # actual heads observed
print("We observed %s heads out of %s"%(h,n)) # We observed ~62 heads out of 100

# Expected distribution for fair coin
p = 0.5 # Ho
rv = st.binom(n,p) # create binomial distribution
mu = rv.mean()
sd = rv.std()
print("The expected distribution for a fair coin is mu=%s, sd=%s"%(mu,sd)) # The expected distribution for a fair coin is mu=50.0, sd=5.0

# Hypothesis testing with binomial test
print("binomial test p-value: %s"%st.binom_test(h, n, p)) # binomial test p-value: 0.000873719836912
# Hypothesis test with normal approximation for binomial (Z-Test with continuity correction)
z = (h-0.5-mu)/sd
print("normal approximation p-value: - %s"%(2*(1 - st.norm.cdf(z)))) # normal approximation p-value: 0.000966848284768

# Simulation
nsamples = 100000
xs = np.random.binomial(n, p, nsamples)
print("simulation p-value - %s"%(2*np.sum(xs >= h)/(xs.size + 0.0)))
# simulation p-value - 0.00062

# Maximum Likelihood Estimation (MLE)
bs_samples = np.random.choice(results, (nsamples, len(results)), replace=True)
bs_ps = np.mean(bs_samples, axis=1)
bs_ps.sort()
print("Maximum likelihood %s"%(np.sum(results)/float(len(results)))) # Maximum likelihood 0.67
print("Bootstrap CI: (%.4f, %.4f)" % (bs_ps[int(0.025*nsamples)], bs_ps[int(0.975*nsamples)])) # Bootstrap CI: (0.5800, 0.7600)

# Bayesian Estimation
# The Bayesian approach directly estimates the posterior distribution,
# from which all other point/interval statistics can be estimated.
fig  = plt.figure()
ax = fig.add_subplot(111)
a, b = 10, 10
prior = st.beta(a, b)
post = st.beta(h+a, n-h+b)
ci = post.interval(0.95)
map_ =(h+a-1.0)/(n+a+b-2.0)
xs = np.linspace(0, 1, 100)
ax.plot(prior.pdf(xs), label='Prior')
ax.plot(post.pdf(xs), label='Posterior')
ax.axvline(mu, c='red', linestyle='dashed', alpha=0.4)
ax.set_xlim([0, 100])
ax.axhline(0.3, ci[0], ci[1], c='black', linewidth=2, label='95% CI');
ax.axvline(n*map_, c='blue', linestyle='dashed', alpha=0.4)
ax.legend()
plt.show() # plt.savefig("coin-toss.png")

# Use the Python code above to play around with the prior specification a little bit.
# Does it seem to influence the results of the analysis (i.e., the resulting posterior distribution)?
# If so, how?

# How were the Bernoulli and binomial distributions used here?
# Explain the underlying hypothesis and the tests used to investigate it.
# Can you interpret the p-values based on this level of significance (assuming a=0.05)?
# Compare and contrast the Bayesian and Frequentist paradigms for estimation.
# Are there any other examples besides the coin-flip where you might apply what you have learned here?
