import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

def states(data_file, binwidth):

    data = genfromtxt(data_file, delimiter=',')
    plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    plt.show()

    return None

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("file", type=str,
                        help="csv of energies\n"
                             ""
                        )
    parser.add_argument("binwidth", type=float,
                        help="width of histogram bin"
                             "try 500"
                        )

    args = parser.parse_args()
    states(args.file, args.binwidth)
    return None

main()
