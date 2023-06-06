
# Location:
# /home/gagrawal/jsummers/03_plotting-data-Edited.ipynb
# In order to produce the table, run 01 and 02 beforehand (in that order)

import numpy as np

"""
Weighted quantile
From Stack Overflow: @Alleo
https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
"""
# Joe Summers
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!

    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.

    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

# Example of how we call it later
# The Median
# plmedian = weighted_quantile(a, .5, sample_weight=ploww/sum(ploww))
# The 68%
# plrange_bottom = weighted_quantile(a, .5-.34, sample_weight=ploww/sum(ploww))
# plrange_top = weighted_quantile(a, .5+.34, sample_weight=ploww/sum(ploww))
