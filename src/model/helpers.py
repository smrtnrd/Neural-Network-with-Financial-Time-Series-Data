import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

def plot_ili(df, name, label ):
    print(df.head())
    plt.plot(df[name].values, color='red', label=label)
    plt.legend(loc='best')
    plt.show()
    
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