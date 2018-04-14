# The complement of the union of two sets is the same as the intersection of their complements
# The complement of the intersection of two sets is the same as the union of their complement

a = set(["A","B","C","D"])
b = set(["C","D","E","F"])
sample_space = set(["A","B","C","D","E","F","G"])

# https://docs.python.org/2/library/sets.html
# https://www.programiz.com/python-programming/set
print a.intersection(b)
# set(['C', 'D'])
print a.difference(b)
# set(['A', 'B'])
print a.union(b)
# set(['A', 'B', 'C', 'D', 'E', 'F'])
complement_a = sample_space.difference(a)
print complement_a
# set(['E', 'F', 'G'])

part1= sample_space.difference(a.union(b))
part2=sample_space.difference(b).intersection(sample_space.difference(a))
print part1==part2
# True
