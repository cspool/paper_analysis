# <span id="page-14-0"></span>C. Statistical principles utilized in this work

Coefficient of Determination  $R^2$ . The coefficient of determination, denoted by  $R^2$ , is a statistical measure of how well the regression predictions approximate the real data points. Formally, for a set of observed values  $\{y_i\}_{i=1}^n$  with mean  $\bar{y}$  and corresponding fitted values  $\{\hat{y}_i\}_{i=1}^n$ , it is defined as:

<span id="page-14-2"></span>
$$R^{2} = 1 - \frac{\sum_{i=1}^{n} (y_{i} - \hat{y}_{i})^{2}}{\sum_{i=1}^{n} (y_{i} - \bar{y})^{2}}.$$

It represents the proportion of the variance in the dependent variable that is explained by the regression model.

**P-value.** Given a null hypothesis  $H_0$  and a test statistic (based on a sample) used to decide whether to reject  $H_0$ , the *p-value* 

is the probability, under the assumption that  $H_0$  is true, of obtaining a test statistic value at least as extreme as the one that was actually observed. Symbolically, if T is the test statistic, and  $t_{\rm obs}$  its observed value,

p-value = 
$$P(T \ge t_{\text{obs}} \mid H_0)$$
,

for a one-sided test (or an analogous definition for two-sided tests). A smaller p-value indicates stronger evidence against  $H_0$ .

Beta Coefficients in Simple Linear Regression Consider a simple linear regression model:

$$Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i,$$

where:

 $\beta_0$  is the intercept (the predicted value of Y when X=0),

 $\beta_1$  is the slope (the expected change in Y for a one-unit increase in X).

 $\varepsilon_i$  is the error term, assumed to have mean zero.

In this context, the slope  $\beta_1$  is given by

$$\beta_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2},$$

which measures the strength and direction of the linear relationship between X and Y.

**T-test of the p-value** A *t-test* assesses whether the mean(s) of one or two groups differ(s) from a hypothesized value or from each other under the null hypothesis  $H_0$ . Let T be the test statistic calculated from the data (for instance, comparing sample mean(s) to the hypothesized mean(s)), and let  $t_{obs}$  be the observed value of T. The *p-value* for the t-test is then defined as:

p-value = 
$$P(|T| \ge |t_{\text{obs}}| \mid H_0)$$

for a two-sided test (or a correspondingly appropriate one-sided version). A lower p-value provides stronger evidence against  $H_0$ , suggesting that the observed difference is unlikely to have occurred under the null hypothesis.

## C.1. Definition of model-specific coefficients

**Definition C.1** (Model-Specific Coefficients). For the *Reasoning Language Models*, the fitted model is

$$\hat{Y}_R = \beta_{0,R} + \beta_{1,R} X,$$

where

$$\beta_{1,R} = -7.894.$$

For the Non-Reasoning Language Models, the fitted model is

$$\hat{Y}_{NR} = \beta_{0,NR} + \beta_{1,NR} X,$$

where

$$\beta_{1.NR} = -15.938.$$