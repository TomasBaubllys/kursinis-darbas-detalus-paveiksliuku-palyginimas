import matplotlib.pyplot as plt
import numpy as np


def imgshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(0, 0, text)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
