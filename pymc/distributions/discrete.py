import pymc3 as pm
import matplotlib.pyplot as plt

with pm.Model() as model:
    binomial = pm.Binomial('binomial', 10, 0.8)
    bernoulli = pm.Bernoulli('bernoulli', 0.2)

    trace = pm.sample(100, nchains=1)

pm.traceplot(trace)
plt.show()