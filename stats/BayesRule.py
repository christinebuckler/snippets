# Return the probability of A conditioned on B
# p0 = P(A)
# p1 = P(B|A)
# p2 = (Not B|Not A)
def f(p0,p1,p2):
    return p0 * p1 / (p0 * p1 + (1 - p0) * (1 - p2))

# Return the probability of A conditioned on Not B
# p0 = P(A)
# p1 = P(B|A)
# p2 = (Not B|Not A)
def f(p0,p1,p2):
    return p0 * (1 - p1) / (p0 * (1 - p1) + (1 - p0) * (1 - p2))
