# # Polynomial Regression - python, sklearn
# We have a couple options we could use to solve [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression) using [sklearn](http://scikit-learn.org). Below is one appraoch.
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# ## Generate Data
# We'll generate sample sample data with known coefficients and randomly generated error. Specifically;
# $$ y = \beta_0 + \sum_i (\beta_i x_i) + \epsilon   \thinspace \thinspace \thinspace \thinspace \thinspace     \forall i \in {1..3}$$
# $$ \beta_0: 4 $$
# $$ \beta_1: 6 $$
# $$ \beta_2: 2 $$
# $$ \beta_3: -1 $$

x1 = np.array([[i,] for i in np.random.uniform(low=-2.0, high=4.0, size=10000)])
poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(x1)
X = np.array([i[1:] for i in X])
calc_y  = lambda x: 4 + 6*x[0] + 2*x[1] - 1*x[2] + np.random.normal(0,4)
Y = np.apply_along_axis(calc_y, 1, X)

# ## Initialize & Run LinearRegression
# Notice we dropped our bias column above so we are going to include an intercept fit in our model below.

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X,Y)

dat = pd.DataFrame(data=np.dstack((np.append(np.array([regr.intercept_, ]), regr.coef_), [4,6,2,-1]))[0])
dat.columns = ['sklearn_coef', 'true_coef']
dat

# Plot Original Data Scatter Plot & Predict Function
x_range = np.array([[i,] for i in linspace(-2.0, 4.0, 100)])
x_values = np.array([i[1:] for i in poly.fit_transform(x_range)])
plt.scatter([i[0] for i in X], Y,  color='black')
plt.plot([i[0] for i in x_values], regr.predict(x_values), color='blue',linewidth=3)
