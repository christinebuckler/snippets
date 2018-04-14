'''
This file contains functions to generate line graphs, histograms,
scatterplots (with line of best fit), boxplots, and bar graphs using matplotlib.pyplot.
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use('ggplot')
mpl.rcParams.update({
    'font.size'           : 16.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'small',
    'legend.fontsize'     : 'small',})


def plot_function(x, y, axis=111, xlabel='', ylabel='', title='', color=None, label=None):
    '''
    INPUT:
    x = array/list of numerical values to input in function
    y = PDF, PMF, or similar function
    OUTPUT:
    Matplotlib plot of the function
    '''
    ax = fig.add_subplot(axis)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.plot(x, y, color=color, lw=2, label=label)

def plot_histogram(x, axis=111, xlabel='', ylabel='Frequencies', title='', color=None, label=None):
    '''
    INPUT:
    x = array/list of numerical values
    OUTPUT:
    Matplotlib histogram
    '''
    ax = fig.add_subplot(axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.hist(x, bins=bins, color=color, rwidth=0.95, label=label)
    plt.xticks(rotation='vertical')

def plot_scatter(x, y, axis=111, xlabel='', ylabel='', title='', color=None, label=None):
    '''
    INPUT:
    x = array/list of numerical values
    y = array/list of numerical values
    OUTPUT:
    Matplotlib scatterplot with line of best fit
    '''
    ax = fig.add_subplot(axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.scatter(x, y, color=color, label=label)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(x, x * slope + intercept, marker='o', color=color, lw=2)

def plot_boxplot(x, axis=111, xlabel='', ylabel='Observed values', title='', label=None):
    '''
    INPUT:
    x = array/list of numerical values
    OUTPUT:
    Matplotlib boxplot with median labeled
    '''
    ax = fig.add_subplot(axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    bplot = ax.boxplot(x, medianprops={'color': 'k'}, patch_artist=True, label=label)
    [patch.set_facecolor('gray') for patch in bplot['boxes']] # Fills boxes with gray
    for line in bplot['medians']:
        (x,y) = line.get_xydata()[1]
        ax.annotate(y, (x,y))

def plot_bar_graph(y, categories, num_groups=0, axis=111, xlabel='', ylabel='', title='', color=None, label=None):
    '''
    INPUT:
    y = array/list of numerical values
    categories = array/list of corresponding category names
    OUTPUT:
    Matplotlib bar graph
    '''
    ax = fig.add_subplot(axis)
    num_groups = range(len(y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(categories) # Set labels for each category on the x axis
    ax.legend(loc='best')
    ax.bar(num_groups, y, width=0.85, color=color, label=label)

def plot_100_images(images, title=''):
    '''
    INPUT:
    images = array of images
    OUTPUT:
    Matplotlib image subplots
    '''
    total = len(images)
    fig, axarr = pl.subplots(10,10) # 10x10
    for i in range(total):
        ax = axarr[int(i/10), i%10]
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    fig.tight_layout()


if __name__ == '__main__':

    x = np.arange([min], max, [step=1])
    x = np.linspace(min, max, num_vals)
    label = model.__class__.__name__

    '''single plot'''
    # with plt.style.context('ggplot'):
    fig = plt.figure(figsize=(10,10))

    plot_function(x, y, axis=111, xlabel='', ylabel='', title='', color=None, label=None)
    plot_histogram(x, axis=111, xlabel='', ylabel='Frequencies', title='', color=None, label=None)
    plot_scatter(x, y, axis=111, xlabel='', ylabel='', title='', color=None, label=None)
    plot_boxplot(x, axis=111, xlabel='', ylabel='Observed values', title='', label=None)
    plot_bar_graph(y, categories, num_groups=0, axis=111, xlabel='', ylabel='', title='', color=None, label=None)
    plot_line_graph()
    plot_100_images(images, title='')

    '''extras'''
    remove_border()
    ax.set_axis([xmin, xmax, ymin, ymax])
    ax.add axes(left,bottom,width,height)
    ax.autoscale(tight=True)
    ax.fill_between(x, y1, y2=0)
    plt.axvline(x)
    plt.axhline(y)
    plt.axhlines(y,xmin,xmax) # horizontal line between xmin and xmax
    plt.semilogx()
    plt.semilogy()
    plt.grid('off')

    '''multiple subplots'''
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8),
                         tight_layout=True)
    for ax in axes.ravel():
        ax.plot()

    fig, axarr = plt.subplots(4,2)
    for idx,col in enumerate(cols):
        axarr[idx, 0].set_xlabel('')
        axarr[idx, 0].set_ylabel(col)
        axarr[idx, 0].plot(x,y)
        axarr[idx, 1].set_xlabel('')
        axarr[idx, 1].set_ylabel(col)
        axarr[idx, 1].plot(x,y)
    plt.tight_layout()

    '''show or save figure'''
    plt.show()
    plt.savefig('_.png', dpi=300)
    plt.close()
