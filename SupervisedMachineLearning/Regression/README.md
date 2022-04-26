
Regression analysis is a statistical technique to determine the relationship of a dependent variable (usually denoted by Y) and a series of  independent variables. It falls under the umbrella of Supervised learning and is used for predicting the labelled target variable which is continuous. Based on the number of exploratory variable involved, regression can be of two types.

a. Simple Linear Regression(SLR) used when are only two exploratory variables.

    Y = a + bX + ϵ

Y – variable that is dependent
X – Independent (explanatory) variable
a – Intercept
b – Slope
ϵ – Residual (error)
b. Multiple Linear Regression(MLR) used when are more than two exploratory variables.

   y=ß0+ ß1 x1+ …………..ßn xn + ϵ

Y – variable that is dependent
X – Independent (explanatory) variable
ß1 is the coefficient for regression of the first independent variable X1
ß0 is the intercept.
Application :  One of the applications of Regression is to determine the bike rental charge based on the different factors such as season, month, holiday, weekday, working day, day of the week, weather.  Spring/Summer-Monday may have a bike rental rate of 100 per day, while Autumn/Winter-Monday may have a rental rate of 20 per day. As the fare doesn't depend on one single factor but many, regression analysis could be used to determine the relationship.

Implementation :

LinearRegression() from scikit can be used for implementing the Linear Regression(in python).

However, we could experiment other models to improve accuracy.

# Metrics
R2 shows how well terms (data points) fit a curve or line. Higher the R2, better the model is.

Why adjusted R2?
Adjusted R-squared, a modified version of R-squared, adds precision and reliability by considering the impact of additional independent
variables that tend to skew the results of R-squared measurements.

RMSE is the most widely used metric as it has the same units as the dependent variable. 
MAE is more robust to data with outliers.
