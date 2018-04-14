import matplotlib.pyplot as plt
import numpy as np

def get_simple_regression_samples(n,b0=-0.3,b1=0.5,error=0.2,seed=2):
    trueX =  np.random.uniform(-1,1,n)
    trueT = b0 + (b1*trueX)
    return np.array([trueX]).T, trueT + np.random.normal(0,error,n)
n = 20
b0_true = -0.3
b1_true = 0.5
# y= b0 + b1*x
x,y = get_simple_regression_samples(n,b0=b0_true,b1=b1_true)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.plot(x[:,0],y,'ko')
ax.plot(x[:,0], b0_true + x[:,0]*b1_true,color='black',label='model mean')
ax.legend()
plt.show()

# mean squared error (MSE)
# least-squares solution for a linear matrix equation
def fit_linear_lstsq(xdata,ydata):
    matrix = []
    n,d = xdata.shape
    for i in range(n):
        matrix.append([1.0, xdata[i,0]])
    return np.linalg.lstsq(matrix,ydata)[0]
coefs_lstsq = fit_linear_lstsq(x,y)
y_pred_lstsq = coefs_lstsq[0] + (coefs_lstsq[1]*x[:,0])
print "true values: b0=%s,b1=%s"%(b0_true,b1_true)
print "lstsq fit: b0=%s,b1=%s"%(round(coefs_lstsq[0],3),round(coefs_lstsq[1],3))

# Calculate the RMSE (root mean squared error) for the data and prediction in the code above.
