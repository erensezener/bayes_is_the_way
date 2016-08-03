import numpy as np
import corner

ndim, nsamples = 3, 50000

# Generate some fake data.
np.random.seed(42)
data = np.random.normal(0, 1, (nsamples, ndim))

# Plot it.
figure = corner.corner(data, show_titles=True, title_kwargs={"fontsize": 12})