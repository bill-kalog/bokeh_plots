'''
toy bokeh implementation of exploring the innards of fc layers
similar to what is found at
https://metamind.io/research/learning-when-to-skim-and-when-to-read
'''
import numpy as np
import pickle
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, save
from bokeh.models import HoverTool
# from bokeh.io import output_notebook
# output_notebook()
from sklearn.manifold import TSNE
import pandas as pd

path_ = 'metrics.pkl'
df = pd.read_pickle(path_)


# Run TSNE
tsne = TSNE(n_components=2, random_state=0)
tsne_repr = tsne.fit_transform(list(df['layer']))
# print (tsne_repr)

# assess scores
tp_ = []  # true positive
tp_ind = []
tn_ = []  # true negative
tn_ind = []
fp_ = []  # false positive
fp_ind = []
fn_ = []  # false negative
fn_ind = []
y_dev = list(df['y_dev'])
y_net = list(df['y_net'])
for i_, val_ in enumerate(y_dev):
    if y_net[i_] == y_dev[i_]:  # got it correct
        if val_ == 1:
            # positive
            tp_.append(tsne_repr[i_])
            tp_ind.append(i_)
        else:
            # negative
            tn_.append(tsne_repr[i_])
            tn_ind.append(i_)
    else:
        if val_ == 1:
            # was positive said negative
            fn_.append(tsne_repr[i_])
            fn_ind.append(i_)
        else:
            fp_.append(tsne_repr[i_])
            fp_ind.append(i_)


# build bokeh plot
size_ = 10
line_width_ = 2
fill_alpha_ = 0.7
# parametrize more hover @ http://bokeh.pydata.org/en/latest/docs/user_guide/tools.html#custom-tooltip
hover = HoverTool(
    tooltips=[
        ("(x,y)", "(@x, @y)"), ("sentence", "@desc"), ("prob", "@prob")]
)

TOOLS = ["pan,wheel_zoom,box_zoom,reset,save, crosshair", hover]
# (was , predicted) -- (line, filling) -- blue - > positive, red -> negative
COLORS = [('blue', 'blue'), ('red', 'red'), ('red', 'blue'), ('blue', 'red')]
LEGEND = ['true positive', 'true negative', 'false positive', 'false negative']
INDICES = [tp_ind, tn_ind, fp_ind, fn_ind]

plot = figure(
    title="FC network layer", tools=TOOLS,
    x_axis_location=None, y_axis_location=None)
# plot = figure(width=400, height=400)

for ind_ in range(len(INDICES)):
    if len(tsne_repr[INDICES[ind_]]) > 0:
        source = ColumnDataSource(
            data=dict(
                x=tsne_repr[INDICES[ind_]][:, 0],
                y=tsne_repr[INDICES[ind_]][:, 1],
                desc=np.asarray(df.get('x_dev'))[INDICES[ind_]],
                prob=np.asarray(df.get('prob_net'))[INDICES[ind_]]
            )
        )
        plot.circle(
            x='x', y='y', size=size_, source=source,
            color=COLORS[ind_][1], fill_alpha=fill_alpha_,
            line_width=line_width_, line_color=COLORS[ind_][0],
            legend=LEGEND[ind_])



show(plot)
save(plot, "plot.html")
