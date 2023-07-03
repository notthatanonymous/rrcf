import numpy as np
import pandas as pd
import rrcf

# Read data
taxi = pd.read_csv('resources/nyc_taxi.csv',
                   index_col=0)

taxi.index = pd.to_datetime(taxi.index)
data = taxi['value'].astype(float).values

# Create events
events = {
'independence_day' : ('2014-07-04 00:00:00',
                      '2014-07-07 00:00:00'),
'labor_day'        : ('2014-09-01 00:00:00',
                      '2014-09-02 00:00:00'),
'labor_day_parade' : ('2014-09-06 00:00:00',
                      '2014-09-07 00:00:00'),
'nyc_marathon'     : ('2014-11-02 00:00:00',
                      '2014-11-03 00:00:00'),
'thanksgiving'     : ('2014-11-27 00:00:00',
                      '2014-11-28 00:00:00'),
'christmas'        : ('2014-12-25 00:00:00',
                      '2014-12-26 00:00:00'),
'new_year'         : ('2015-01-01 00:00:00',
                      '2015-01-02 00:00:00'),
'blizzard'         : ('2015-01-26 00:00:00',
                      '2015-01-28 00:00:00')
}
taxi['event'] = np.zeros(len(taxi))
for event, duration in events.items():
    start, end = duration
    taxi.loc[start:end, 'event'] = 1

# Set tree parameters
num_trees = 200
shingle_size = 48
tree_size = 1000

# Use the "shingle" generator to create rolling window
points = rrcf.shingle(data, size=shingle_size)
points = np.vstack([point for point in points])
n = points.shape[0]
sample_size_range = (n // tree_size, tree_size)

forest = []
while len(forest) < num_trees:
    ixs = np.random.choice(n, size=sample_size_range,
                           replace=False)
    trees = [rrcf.RCTree(points[ix], index_labels=ix)
             for ix in ixs]
    forest.extend(trees)

avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)

for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf)
                        for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)

avg_codisp /= index
avg_codisp.index = taxi.iloc[(shingle_size - 1):].index

avg_codisp = avg_codisp.to_frame()
avg_codisp.columns = ["codisp_score"]
avg_codisp['in_out_class'] = list(avg_codisp['codisp_score'] > avg_codisp['codisp_score'].quantile(0.99))

avg_outlier_score = (avg_codisp.loc[avg_codisp['in_out_class'] == True, :]['codisp_score'].mean())
avg_inlier_score = (avg_codisp.loc[avg_codisp['in_out_class'] == False, :]['codisp_score'].mean())

print(f"\n\n\nScore: {avg_outlier_score/avg_inlier_score}\n\n\n")
