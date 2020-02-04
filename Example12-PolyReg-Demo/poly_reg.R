# # Polynomial Regression - R
# To demonstrate the differences of running models in local R and sparklyr, we'll use a simple [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression).

# ## Generate Data
# We'll generate sample sample data for a multivariate linear regression with known coefficients and randomly generated error. Specifically;
# $$ y = \beta_0 + \sum_i (\beta_i x_i) + \epsilon   \thinspace \thinspace \thinspace \thinspace \thinspace     \forall i \in {1..3}$$
# $$ \beta_0: 4 $$
# $$ \beta_1: 6 $$
# $$ \beta_2: 2 $$
# $$ \beta_3: -1 $$

gen_dat    <- data.frame(x1=runif(10000,-2,4))
gen_dat$x2 <- gen_dat$x1^2
gen_dat$x3 <- gen_dat$x1^3
gen_dat$y  <- 4 + 6*gen_dat$x1 + 2*gen_dat$x2 - 1*gen_dat$x3 +  rnorm(10000,0,4)

# ## Visualize Data
# Eventhough we are evaluating a polynomial, there is only one dependent variable, y, and one independent variable, x. Plotting:

plot(gen_dat$x1,gen_dat$y, xlab="x", ylab="y", main="Randomly Generated Data Set")

# ## Fit Model
# We'll fit this model using local R's stats lib which is a core library. Specifcally, we'll fit using [glm](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/glm.html).

lm <- glm(y~x1+x2+x3, data=gen_dat)
summary(lm)

# By inspection, we see that our estimates are close to the fitted coefficients:

data.frame(lm.coef=lm$coefficients, true.coef=c(4,6,2,-1))

# ## Standard Plots
# R comes with many standard plots which are still accessible in CDSW:

plot(lm, which=1)
plot(lm, which=3)
plot(lm, which=2)
plot(lm, which=4)

# You can also predict and plot our prediction on top of our observations:
pdat    <- data.frame(x1=seq(from = -2, to = 4, by = 0.05))
pdat$x2 <- pdat$x1^2
pdat$x3 <- pdat$x1^3
pdat$y  <- predict(lm, pdat , interval = "prediction")

plot(gen_dat$x1,gen_dat$y, xlab="x", ylab="y", main="Randomly Generated Data Set")
lines(pdat$x1,pdat$y,col="blue",lw=4)

