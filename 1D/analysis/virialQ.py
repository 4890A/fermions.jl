#!/usr/local/bin/python3
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

def states(data_file, e3):

    data = genfromtxt(data_file, delimiter=',')
    En = data * e3
    B = np.linspace(0,3,1000)
    Qb = np.zeros(1000)
    for i in range(1000):
        Qb[i] = np.sum(np.exp(data * e3 * B[i]))
    plt.plot(B, Qb)
    plt.xlabel("coldness")
    plt.ylabel("Q")
    plt.show()

    return None

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("file", type=str,
                        help="csv of energies\n"
                             ""
                        )
    parser.add_argument("e3", type=float,
                        help="trimer_binding_energy"
                             "try 500"
                        )

    args = parser.parse_args()
    states(args.file, args.e3)
    return None

main()
