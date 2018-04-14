# Three types of fair coins are in an urn: HH, HT, and TT
# You pull a coin out of the urn, flip it, and it comes up H
# Q: what is the probability it comes up H if you flip it a second time?

import random
import pandas as pd
coins = ['HH', 'HT', 'TT']
results = []
for i in range(1000):
    coin = random.choice(coins)
    results.append([random.choice(coin) for j in [1,2]])
df = pd.DataFrame(results, columns=['first', 'second']) == 'H'
# Now group first column/flip by true/H to see how many true/H on second column/flip
print df.groupby('first').mean() # True .8333 (5./6 will automatically be float if using py3)

# another way of simulating the conditional probability w/o pandas...
# import numpy as np
# n = 10000
# coins = ['HH', 'HT', 'TT']
# coins_selected = np.random.choice(coins,n)
# first_side_shown = np.array([c[np.random.random_integers(0,1,1)[0]] for c in coins_selected])
# coins_with_heads = coins_selected[np.where(first_side_shown == 'H')[0]]
# second_side_shown = np.array([c[np.random.random_integers(0,1,1)[0]] for c in coins_with_heads])
# print("%s/%s"%(np.where(second_side_shown=='H')[0].size,second_side_shown.size))
# print(np.where(second_side_shown=='H')[0].size/second_side_shown.size)
# 4096/4938 or 0.8295
