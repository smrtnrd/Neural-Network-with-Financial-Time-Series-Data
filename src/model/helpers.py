import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

from datetime import datetime

def plot_ili(df, name, label ):
    # summarize first 5 rows
    print(df.head(5))
    
    #save the image
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    plt.plot(df[name].values, color='red', label=label)
    plt.legend(loc='best')
    # plt.show()
    
    date = datetime.now() #today date
    path = 'fig/plot-1-' + date.strftime('%Y-%m-%d')
    fig.savefig(path)   # save the figure to file
    plt.close(fig)    # close the figure
    print("plot saved")

def plot_corr(df):
    #save the image
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    corr = df.corr()
    ax = sns.heatmap(corr, cmap="YlGnBu")
    #plt.show()

    date = datetime.now() #today date
    path = 'fig/plot-2-' + date.strftime('%Y-%m-%d')
    fig.savefig(path)   # save the figure to file
    plt.close(fig)    # close the figure
    print("plot saved")

def plot_ili_group(df, groups):
    print(df.head())
    values = df.values
    # specify columns to plot
    # groups = [0,1,2]
    i = 1
    # plot each column
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(df.columns[group], y=0.5, loc='right')
        plt.legend(loc='best')
        i += 1
    plt.show()

def plot_result(df, normalized_value_p, normalized_value_y_test):
    newp = denormalize(df, normalized_value_p)
    newy_test = denormalize(df, normalized_value_y_test)
    plt2.plot(newp, color='red', label='Prediction')
    plt2.plot(newy_test,color='blue', label='Actual')
    plt2.legend(loc='best')
    plt2.title('The test result for {}'.format(stock_name))
    plt2.xlabel('Days')
    plt2.ylabel('ILI Days')
    plt2.show()