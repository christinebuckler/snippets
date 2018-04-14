from scipy.stats import pearsonr
from scipy.stats import spearmanr

# There are a couple common choices for statistics that correspond to linear associations parameters.
# The Pearson correlation coefficient measures the linear relationship between two datasets.
# The alternative Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets,
# which is just a fancy way of saying that calculates the correlation on the ranks rather than original values.

# The input tuple is X and Y values

# print pearsonr([1,2,3,4,5],[5,6,7,8,7]) # (0.83205029433784372, 0.080509573298498519)
print 'Pearson correlation: {}, p-value: {}'.format(pearsonr([1,2,3,4,5],[5,6,7,8,7])[0],pearsonr([1,2,3,4,5],[5,6,7,8,7])[1])
print spearmanr([1,2,3,4,5],[5,6,7,8,7]) # SpearmanrResult(correlation=0.82078268166812329, pvalue=0.088587005313543812)

# The first value in the output tuple is the correlation.
# The second is a p-value of a statistical test of the null hypothesis of no association.
