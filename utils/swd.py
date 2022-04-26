import matplotlib.pyplot as plt
import numpy as np
import random

def plot_track_KL(input_text, choose_client=-1):
    r = lambda: random.randint(0,255)

    with open(input_text, 'r') as f:
        lines = [line.rstrip() for line in f]
        for i, line in enumerate(lines):
            KL_values = line.split(": ")[-1]
            KL_values = KL_values.split(", ")
            KL_values = [float(d.replace(",", "")) for d in KL_values]
            color = '#%02X%02X%02X' % (r(),r(),r())
            if choose_client == -1 or i == choose_client:
                plt.plot(KL_values, color=color, label="Client {}".format(i))

    plt.xlabel("Round on client", fontsize=30)
    plt.ylabel("KL distance", fontsize=30)

    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.tick_params(axis='y', which='major', labelsize=20)

    plt.legend(loc="upper right", prop={'size':15})
            
    plt.show()

plot_track_KL("../runs/G_Noise_MNIST0.1_trackKL/KL_track_200epochs.txt")