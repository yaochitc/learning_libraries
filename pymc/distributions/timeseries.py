import pymc3 as pm
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt

gdp_year = ts.get_gdp_year()
gdp_year.set_index('year')
gdp_year = gdp_year[::-1]

gdp_year['gdp'] = gdp_year['gdp'].apply(lambda x: x/1000)

gdp_year['lag'] = gdp_year['gdp'].shift()

gdp_year.dropna(inplace=True)
with pm.Model() as model:
    sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
    nu = pm.Exponential('nu', 1. / 10)
    beta = pm.GaussianRandomWalk('beta', sigma ** -2, shape=len(gdp_year['gdp']))
    observed = pm.Normal('observed', mu=beta * gdp_year['lag'], sd=1 / nu, observed=gdp_year['gdp'])

    trace = pm.sample(1000, tune=1000, cores=2)

plt.plot(gdp_year.index,trace['beta'].T, 'b', alpha=.03)
plt.plot(gdp_year.index, 1 + (np.log(gdp_year['gdp']) - np.log(gdp_year['lag'])), 'r', label='True Growth Rate')
plt.show()