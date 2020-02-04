
# # Polynomial Regression - sparklyR, MLlib
# To demonstrate the differences of running models in local R and sparklyr, we'll use a simple [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression).
# Note that there are two frameworks you can use to run machine learning models with [sparklyR](http://spark.rstudio.com/); [Spark MLlib](http://spark.rstudio.com/mllib.html) & [H2O](http://spark.rstudio.com/h2o.html). This demonstration is for **MLlib**.

# ## Preparing Project Environment
# Prior to running the following script you make have some pre-requisite library installs:
# * install.packages("sparklyr")
# * install.packages("mgvc")

# ## Initialize a sparklyr Spark Context
# Since procuring a spark context will be a pretty common task for any spark analysis, I recommend defining an initialization function to be sourced for every R session in the project. This is accomplished by writting the initialization function to a file, [init_sparklyr_mllib.r](http://github.mtv.cloudera.com/barker/demo_poly_reg/blob/master/init/init_sparklyr_mllib.r), and sourcing that file at startup using [.Rprofile](http://github.mtv.cloudera.com/barker/demo_poly_reg/blob/master/.Rprofile).
# Now you can initialize your sparkylr spark ml spark context via a function:

conf <- spark_config()
sc <- spark_connect(master = "yarn-client", config = conf)
assign("sc", sc, envir = .GlobalEnv)
f = file(sc$output_file, "r")
str = readLines(f)
close(f)
spark_version  <- sub(".*version ", "", str[grepl("Running Spark version ",str)][1])
application_id <- sub(" has.*", "", sub(".*application_", "", str[grepl("has started running",str)][1]))
str <- paste(paste0(sessionInfo()$R.version$version.string," ",sessionInfo()$R.version$nickname),
         paste0("sparklyR, MLlib Version: ",sessionInfo()$otherPkgs$sparklyr$Version),
         "      ____              __",
         "     / __/__  ___ _____/ /__",
         "    _\\ \\/ _ \\/ _ `/ __/  `_/",
         paste0("   /__ / .__/\\_,_/_/ /_/\\_\\   version ",spark_version),
         "      /_/", sep="\n")
cat(str)


# `init_sparklyr_mllib()` returns `sc` as a global environment variable. That is the spark context you will use for this demo.

# ## Generate Spark DataFrame Data
# We'll generate sample data for a multivariate linear regression with known coefficients and randomly generated error. Specifically;
# $$ y = \beta_0 + \sum_i (\beta_i x_i) + \epsilon   \thinspace \thinspace \thinspace \thinspace \thinspace     \forall i \in {1..3}$$
# $$ \beta_0: 4 $$
# $$ \beta_1: 6 $$
# $$ \beta_2: 2 $$
# $$ \beta_3: -1 $$

# Unfortunately, at the time of writing, there is no concise method for generating random data in sparklyr spark ml. So we will create our independent and error variables locally and copy to Spark:

gen_dat    <- data.frame(x1=runif(10000,-2,4))
gen_dat$x2 <- gen_dat$x1^2
gen_dat$x3 <- gen_dat$x1^3
gen_dat$y  <- 4 + 6*gen_dat$x1 + 2*gen_dat$x2 - 1*gen_dat$x3 +  rnorm(10000,0,4)

sdf_gen_dat  <- copy_to(dest = sc,
                          df = gen_dat ,
                          name = "gen_dat",
                          memory = FALSE,
                          overwrite = TRUE)

# ### Make a Referentiable Spark DataFrame
gen_dat  <- spark_dataframe(sdf_gen_dat)

# ### Train A Linear Regression
# Ok, this one requires a little more explaination. The sparklyr library provides bindings to Spark’s distributed [machine learning library](https://spark.apache.org/docs/2.0.0/mllib-guide.html). In particular, sparklyr allows you to access the machine learning routines provided by the spark.ml package which requires the data to be in spark DataFrame form.
# While the code here is orchestrated entirely within R, it is actually running in Java within the spark framework. It is not running anything using [sparkR](https://spark.apache.org/docs/2.0.0/sparkr.html).
# Bindings have been written for the following general transform pattern within sparklyr:

# * [Machine learning](http://spark.rstudio.com/mllib.html#algorithms) algorithms for analyzing data (ml_\*)
# * [Feature transformers](http://spark.rstudio.com/mllib.html#transformers) for manipulating individual features (ft_\*)
# * Functions for manipulating [Spark DataFrames](http://spark.rstudio.com/reference/index.html#section-spark-dataframes) (sdf_\*)

# We will only be evaluating a simple transform from the available machine learning algorithms:

lm_model <- gen_dat %>% ml_linear_regression(y ~ .)

# **NOTE:** While the **%>%** notations is relatively new and not universally adopted by R users it is being promoted by RStudio and is based upon [magrittr](https://cran.r-project.org/web/packages/magrittr/index.html). It does have the benefit of making a string of transforms comperable to what is done in spark scala and pyspark examples today.
# **NOTE:** There are other function groups sparklyr users will need to know. Most noteable is the [extension API](http://spark.rstudio.com/reference/index.html#section-extensions-api) for calling Java classes directly from R. At the moment this is awkward and should not be considered viable for interactive use. However it does open up the option for users to write thier own [wrapper functions](http://spark.rstudio.com/extensions.html#wrapper_functions) for tasks that are not yet binded.

# We do get a proper summary, but R standard plots from local R are not available.
summary(lm_model)

# ## Plot Polynomial Regression
# Plotting is always helpful, but the reason that we are modeling in spark in the first place is becuase the data is big. So preferably there is some method to reduce the number of data points. Not ideal, but we'll collect all our data back and plot using [ggplot2](http://ggplot2.org/).

library(ggplot2, quietly=TRUE)

plot_dat <- gen_dat  %>% collect()

c <- ggplot(plot_dat, aes(x1, y))
c  + geom_point() + stat_smooth() + 
  labs(x = "x",y = "y",title = "Polynomial Regression")
