import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, save
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import HoverTool
import bokeh.palettes
import sys
# from bokeh.io import output_notebook
# output_notebook()
from sklearn.manifold import TSNE
import pandas as pd

path_ = 'metrics_f.pkl'
df = pd.read_pickle(path_)


# Run TSNE
tsne = TSNE(n_components=2, random_state=0)
tsne_repr = tsne.fit_transform(list(df['layer']))

# dictionary having as key the type of assesment done
# eg. "0_1" was 0 pred 1. "3_4", was 3 predicted 4 and so on
class_types = {}
y_dev = list(df['y_dev'])
y_net = list(df['y_net'])
max_class_ind = 0
for i_, val_ in enumerate(y_dev):
    assesment_ = "{}_{}".format(y_dev[i_], y_net[i_])
    if assesment_ in class_types:
        t = class_types[assesment_]
        t.append(i_)
        class_types[assesment_] = t
    else:
        class_types[assesment_] = [i_]
    if y_dev[i_] > max_class_ind:
        max_class_ind = y_dev[i_]


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
        ("[per class probabilities]", "@prob"),
        ("label", "@label")
    ]
)

#  hover boxes of points on the rightmost side always look awfull
#  something like margin-left: 200px; could be used in div as well but
#  not much of a difference
hover = HoverTool(
    tooltips="""
    <div style="max-width: 400px;">
        <div>
            <span style="font-size: 11px;">[rounded per class probabilities]</span>
            <span style="font-size: 12px; font-weight: bold;">(@prob)</span>
        </div>
        <div>
            <span style="font-size: 15px;">sentence: </span>
            <span style="font-size: 17px; color: @label_color;">@desc</span>
        </div>
    </div>
    """
)

TOOLS = ["pan,wheel_zoom,box_zoom,reset,save, crosshair", hover]
COLORS = bokeh.palettes.brewer['Spectral'][max_class_ind + 1]
if max_class_ind == 4:  # a beetter to read color for neutral, if 5 classes
    COLORS[2] = "#F0E442"
COLORS.reverse()
COLORS_11 = COLORS

plot = figure(
    title="sentence representations", tools=TOOLS,
    x_axis_location=None, y_axis_location=None, width=900, height=900)

for ass_type in class_types:
    a = ass_type.split("_")
    
    # round probabilities so that they need less space when plotting
    b = np.asarray(df.get('prob_net'))[class_types[ass_type]]
    r_pr = [np.around(b[ii], decimals=4) for ii in range(len(b))]


    color_labels_ = [COLORS[int(a[0])], COLORS[int(a[1])]]

    source = ColumnDataSource(
        data=dict(
            x=tsne_repr[class_types[ass_type]][:, 0],
            y=tsne_repr[class_types[ass_type]][:, 1],
            desc=np.asarray(df.get('x_dev'))[class_types[ass_type]],
            prob=r_pr,
            label=np.asarray(df.get('y_dev'))[class_types[ass_type]],
            label_color=[color_labels_[0]] *
            len(tsne_repr[class_types[ass_type]][:, 0])

        )
    )

    plot.circle(
        x='x', y='y', size=size_, source=source,
        color=color_labels_[1], fill_alpha=fill_alpha_,
        line_width=line_width_, line_color=color_labels_[0],
        legend="was {} pred {}".format(a[0], a[1]))
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
            <span style="font-size: 11px;">[rounded per class probabilities]</span>
            <span style="font-size: 12px; font-weight: bold;">(@prob)</span>
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
                width=900, height=900)

for ind_ in range(len(INDICES)):
    if len(tsne_repr[INDICES[ind_]]) > 0:

        # round probabilities so that they need less space when plotting
        b = np.asarray(df.get('prob_net'))[INDICES[ind_]]
        r_pr = [np.around(b[ii], decimals=4) for ii in range(len(b))]

        source = ColumnDataSource(
            data=dict(
                x=tsne_repr[INDICES[ind_]][:, 0],
                y=tsne_repr[INDICES[ind_]][:, 1],
                desc=np.asarray(df.get('x_dev'))[INDICES[ind_]],
                prob=r_pr,
                label=np.asarray(df.get('y_dev'))[INDICES[ind_]],
                label_color=[COLORS_11[color_]
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
