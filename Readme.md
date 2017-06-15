## bokeh plot of a fully connected layer

toy example of plotting a fully connected layer as presented
in the blog post [Learning when to skim and when to read](https://metamind.io/research/learning-when-to-skim-and-when-to-read)

### Requirements

- Python 3
- sklearn
- numpy
- pandas
- bokeh

### instructions

the script `innards.py` expects a pandas dataframe similar to the one found in
`metrics.pkl`. which could have been created from something like:

```python
def save_info(x_dev, y_dev, y_net, prob_net, layer, path_):
    '''
    save test set info into a pandas dataframe and pickle it
    x_dev: sentences (list of strings)
    y_dev: sentiment label (list of ints) ex. 0 negative, 1 positive
    y_net: network output (list of ints) ex. 0 negative, 1 positive
    prob_net: networks probabilities/y
    layer: fully connected layer list of a vector list per sentence
    path_: path to save dataframe
    '''
    d = {'x_dev': x_dev, 'y_dev': y_dev, 'layer': layer,
         'y_net': y_net, 'prob_net': prob_net}
    df = pd.DataFrame(data=d)
    df.to_pickle(path_)
```

you can also check the jupyter notebook where a version of an LSTM fully connected
layer from [SST](https://nlp.stanford.edu/sentiment/index.html) is plotted. So far the
two plots of the paragraph `Exploring the innards` are only implemented

Usising the script `innards_finegrained.py` one can plot in space more than two classes.
An example dataset exists at `metrics_f.pkl` and the program's output at `plot_finegrained.html`.
