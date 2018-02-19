import pickle
import matplotlib.pyplot as py

def plotFreqsPickle(filename):
    with open(filename, 'rb') as f:
        freqs = pickle.load(f)

    x, y = zip(*list(freqs.items()))
    py.scatter(x, y)
    py.show()