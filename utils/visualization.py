from matplotlib import pyplot as plt
import numpy as np

def plot_train_valid_curve(train_ppl, valid_ppl, plot_every, path):
    """

    :param train_ppl: list of float number
    :param valid_ppl: list of float number
    :return:
    """
    title = "Training and validation ppl for every %d iterations" % plot_every
    plt.figure(dpi=500)
    plt.title(title)
    plt.xlabel("Checkpoints")
    plt.ylabel("Perplexity")
    num_checkpoints = len(train_ppl)
    X = list(range(num_checkpoints))
    plt.plot(X, train_ppl, label="training ppl")
    plt.plot(X, valid_ppl, label="validation ppl")
    plt.legend()
    plt.savefig(path + '.pdf')
    return

if __name__ == '__main__':
    train_ppl = [20.1,15.3,12.3,11.0,10.0]
    valid_ppl = [30.2,29.2,25.2,21.3,20.2]
    plot_every = 4000
    path = '../exp/debug/valid_train_curve'
    plot_train_valid_curve(train_ppl, valid_ppl, plot_every, path)
