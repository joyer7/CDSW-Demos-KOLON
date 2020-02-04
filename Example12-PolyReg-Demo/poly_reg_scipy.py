# # Polynomial Regression - python, scipy
# We have a couple options we could use to solve [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression) using [SciPy](https://www.scipy.org/). Below is one appraoch.
import numpy as np
import pandas as pd
from scipy import optimize

# ## Generate Data
# We'll generate sample sample data with known coefficients and randomly generated error. Specifically;
# $$ y = \beta_0 + \sum_i (\beta_i x_i) + \epsilon   \thinspace \thinspace \thinspace \thinspace \thinspace     \forall i \in {1..3}$$
# $$ \beta_0: 4 $$
# $$ \beta_1: 6 $$
# $$ \beta_2: 2 $$
# $$ \beta_3: -1 $$

x1 = np.random.uniform(low=-2.0, high=4.0, size=10000)
x2 = np.power(x1,2)
x3 = np.power(x1,3)
y0  = 4 + 6*x1 + 2*x2 - 1*x3 + np.random.normal(0,4,10000)

# ## Initialize & Run Optimizer
# SciPy optimize can solve problems more general that [linear regression](https://en.wikipedia.org/wiki/Linear_regression). While we are technically solving the same problem when minimizing squared residuals, this form of the analysis is more commonly refer to as [linear least squares](https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)). 
# Our solutions invovles definining the function we want to minimize, residual. The optimizer leastsq function will solve for a solution which minimizes the residual sum of squares.
predict  = lambda b, x1, x2, x3: b[0] + b[1]*x1 + b[2]*x2 + b[3]*x3
residual = lambda b, y,x1,x2,x3: predict(b, x1,x2,x3) - y

# This appraoch requires that we initialize our coefficients, we'll choose to randomly initialize.
b0 = np.random.normal(0,1,4)

b, success = optimize.leastsq(residual, b0[:], args=(y0,x1,x2,x3))

# That's it. **b** is an array of our coefficients and it closely matches our known true values.
dat = pd.DataFrame(data=np.dstack((b, [4,6,2,-1]))[0])
dat.columns = ['scipy_coef', 'true_coef']
dat

# ## Plot Predict Function Over Observations
# Now, let's take a look at a plot of our trained parameters. We'll use bokeh so you can interactively see how slight changes in the coefficient can produce very bad results. Running [create_bokeh_poly_reg.py](../files/py/create_bokeh_poly_reg.py).
execfile('py/create_bokeh_poly_reg.py')
