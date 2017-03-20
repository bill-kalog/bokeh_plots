'''
toy bokeh implementation of exploring the innards of fc layers
similar to what is found at
https://metamind.io/research/learning-when-to-skim-and-when-to-read
'''
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, save
from bokeh.models.widgets import Panel, Tabs
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


# asses probabilities
# probability indexing, check networks confidence on decision
vconf_ = []  # very confident
vconf_ind = []
conf_ = []  # confident
conf_ind = []
doubt_ = []  # doubtful
doubt_ind = []
vdoubt_ = []  # very doubtful
vdoubt_ind = []
step_ = [0.5 / 4] * 4  # steps to add in each iteration
start_ = 0.5
for i_, val_ in enumerate(y_dev):
    net_confidence = max(df.get('prob_net')[i_])
    if net_confidence < start_ + sum(step_[:1]):
        vdoubt_ind.append(i_)
    elif net_confidence < start_ + sum(step_[:2]):
        doubt_ind.append(i_)
    elif net_confidence < start_ + sum(step_[:3]):
        conf_ind.append(i_)
    else:
        vconf_ind.append(i_)

# build bokeh plot
size_ = 10
line_width_ = 2
fill_alpha_ = 0.7
# parametrize more hover @ http://bokeh.pydata.org/en/latest/docs/user_guide/tools.html#custom-tooltip
hover = HoverTool(
    tooltips=[
        ("(x,y)", "(@x, @y)"),
        ("sentence", "@desc"),
        ("(neg, pos)", "@prob"),
        ("label", "@label")
    ]
)

colors_ = ['red', 'blue']
#  hover boxes of points on the rightmost side always look awfull
#  something like margin-left: 200px; could be used in div as well but
#  not much of a difference
hover = HoverTool(
    tooltips="""
    <div style="max-width: 400px;">
        <div>
            <span style="font-size: 15px;">(neg, pos)</span>
            <span style="font-size: 15px; font-weight: bold;">(@prob)</span>
        </div>
        <div>
            <span style="font-size: 15px;">sentence: </span>
            <span style="font-size: 17px; color: @label_color;">@desc</span>
        </div>
    </div>
    """
)

TOOLS = ["pan,wheel_zoom,box_zoom,reset,save, crosshair", hover]
# (was , predicted) -- (line, filling) -- blue - > positive, red -> negative
COLORS = [('blue', 'blue'), ('red', 'red'), ('red', 'blue'), ('blue', 'red')]
LEGEND = ['true positive', 'true negative', 'false positive', 'false negative']
INDICES = [tp_ind, tn_ind, fp_ind, fn_ind]

plot = figure(
    title="FC network layer", tools=TOOLS,
    x_axis_location=None, y_axis_location=None, width=800, height=800)
# plot = figure(width=400, height=400)

for ind_ in range(len(INDICES)):
    if len(tsne_repr[INDICES[ind_]]) > 0:

        source = ColumnDataSource(
            data=dict(
                x=tsne_repr[INDICES[ind_]][:, 0],
                y=tsne_repr[INDICES[ind_]][:, 1],
                desc=np.asarray(df.get('x_dev'))[INDICES[ind_]],
                prob=np.asarray(df.get('prob_net'))[INDICES[ind_]],
                label=np.asarray(df.get('y_dev'))[INDICES[ind_]],
                label_color=[COLORS[ind_][0]] *
                len(tsne_repr[INDICES[ind_]][:, 0])

            )
        )
        plot.circle(
            x='x', y='y', size=size_, source=source,
            color=COLORS[ind_][1], fill_alpha=fill_alpha_,
            line_width=line_width_, line_color=COLORS[ind_][0],
            legend=LEGEND[ind_])
tab1 = Panel(child=plot, title="Predictions")

c = ['blue', 'cyan', 'grey', '#eff3ff']
size_ = 10
line_width_ = 2
fill_alpha_ = 0.7
# parametrize more hover @ http://bokeh.pydata.org/en/latest/docs/user_guide/tools.html#custom-tooltip
hover = HoverTool(
    tooltips=[
        ("(x,y)", "(@x, @y)"),
        ("sentence", "@desc"),
        ("(neg, pos)", "@prob"),
        ("label", "@label")
    ]
)

colors_ = ['red', 'blue']

hover = HoverTool(
    tooltips="""
    <div style="max-width: 400px;">
        <div>
            <span style="font-size: 15px;">(neg, pos)</span>
            <span style="font-size: 15px; font-weight: bold;">(@prob)</span>
        </div>
        <div>
            <span style="font-size: 15px;">sentence: </span>
            <span style="font-size: 17px; color: @label_color;">@desc</span>
        </div>
    </div>
    """
)

TOOLS = ["pan,wheel_zoom,box_zoom,reset,save, crosshair", hover]
# (was , predicted) -- (line, filling) -- blue - > positive, red -> negative
# COLORS_2 = [('blue','blue'), ('red','red'), ('red','blue'), ('blue','red')]
COLORS = [('black', c[0]), ('black', c[1]), ('black', c[2]), ('black', c[3])]
LEGEND = ['very confident', 'confident', 'Doubtful', 'very Doubtful']
INDICES = [vconf_ind, conf_ind, doubt_ind, vdoubt_ind]

plot_2 = figure(title="FC network layer", tools=TOOLS,
                x_axis_location=None, y_axis_location=None,
                width=800, height=800)

for ind_ in range(len(INDICES)):
    if len(tsne_repr[INDICES[ind_]]) > 0:
        source = ColumnDataSource(
            data=dict(
                x=tsne_repr[INDICES[ind_]][:, 0],
                y=tsne_repr[INDICES[ind_]][:, 1],
                desc=np.asarray(df.get('x_dev'))[INDICES[ind_]],
                prob=np.asarray(df.get('prob_net'))[INDICES[ind_]],
                label=np.asarray(df.get('y_dev'))[INDICES[ind_]],
                label_color=[['red', 'blue'][color_]
                    for color_ in np.asarray(df.get('y_dev'))[INDICES[ind_]]]
            )
        )
        plot_2.circle(
            x='x', y='y', size=size_, source=source,
            color=COLORS[ind_][1], fill_alpha=fill_alpha_,
            line_width=line_width_, line_color=COLORS[ind_][0],
            legend=LEGEND[ind_])
tab2 = Panel(child=plot_2, title="Probabilities")
tabs = Tabs(tabs=[tab1, tab2])

show(tabs)
save(tabs, "plot.html")
