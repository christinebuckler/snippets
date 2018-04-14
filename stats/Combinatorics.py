# permutations
import math
print math.factorial(10)
# 3628800

from math import factorial
def permu(n, k):
    return factorial(n) / factorial(n - k)
print permu(3,2)
# 6

from itertools import permutations
print list(permutations("ABC",2))
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# On a baseball team with 12 players, how many different batting lineups are there?
# Hint: there are 9 players in a lineup.
print permu(12,9)
# 79833600


print '\n' ##################################################
#combinations
from itertools import combinations
print list(combinations("ABC",2))
# [('A', 'B'), ('A', 'C'), ('B', 'C')]

from math import factorial
def comb1(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))
print comb1(3,2)
#3

from scipy.misc import comb
print comb(3,2)
# 3.0

lefthand_beers = ["Milk Stout", "Good Juju", "Fade to Black", "Polestar Pilsner"]
lefthand_beers += ["Black Jack Porter", "Wake Up Dead Imperial Stout","Warrior IPA"]
# We have sampler plates that hold 4 beers. How many different ways can we combine these beers?
print comb1(7,4)
# Print a list of these pairs.
print list(combinations(lefthand_beers,4))
