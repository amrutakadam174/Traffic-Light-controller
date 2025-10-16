import matplotlib.pyplot as plt
import numpy as np
import os

def show_plots():
    if os.path.exists('results/training_plot.png'):
        img = plt.imread('results/training_plot.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    if os.path.exists('results/avg_wait_before_after.png'):
        img2 = plt.imread('results/avg_wait_before_after.png')
        plt.figure()
        plt.imshow(img2)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    show_plots()
