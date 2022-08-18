from pandas import datetime
from matplotlib import pyplot
import pandas as pd 
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = pd.read_csv('sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')

print(series.head())
series.plot()
pyplot.show()

autocorrelation_plot(series)
pyplot.show()

# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())