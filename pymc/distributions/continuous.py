import pymc3 as pm
import seaborn as sn
import matplotlib.pyplot as plt

with pm.Model() as model:
    uniform = pm.Uniform('uniform', lower=0, upper=1)
    normal = pm.Normal('normal', mu=0, sd=1)
    beta = pm.Beta('beta', alpha=0.5, beta=0.5)
    exponential = pm.Exponential('exponential', 1.0)

    trace = pm.sample(2000)

print(pm.summary(trace).round(2))

pm.traceplot(trace)
plt.show()