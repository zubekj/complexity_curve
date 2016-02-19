import sys
from sklearn.decomposition import FastICA
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import argparse
import pandas as pd
import StringIO

from common import load_dataset
from complexity_curve import hl_distances_from_set

class MajorityClassifier(object):
    """Classifier that always returns majority label."""
    def __init__(self):
        self.majority_label = None

    def fit(self, X, Y):
        counts, vals = np.histogram(Y, bins=(Y.max()-Y.min()))
        self.majority_label = vals[np.argmax(counts)]

    def predict(self, X):
        if(len(X.shape) == 1):
            return self.majority_label
        return np.array([self.majority_label] * X.shape[0])

    def score(self, X, Y):
        Yp = self.predict(X)
        return np.mean(Y == Yp)

def draw_generalization_curve(X, Y, classifiers, add_noise=False, k=20, use_ica=True, **kwargs):
    """
    Draws generalization curve for a given data set.
    """
    Xt = X.copy()
    if add_noise:
        Xt += np.random.normal(0, 0.01, Xt.shape)

    #l = np.linspace(3, x.shape[0], points) if type(points) is int else points
    l0 = np.linspace(1, 10, 10)
    #l1 = np.linspace(12, x.shape[0], 20)[1:]
    l1 = np.logspace(1, np.log10(X.shape[0]), 20)[1:]
    #l2 = np.linspace(0.1*x.shape[0], x.shape[0], 21)[1:]
    l = np.hstack((l0,l1))

    indices = [np.random.choice(range(Xt.shape[0]), int(i), replace=False)
                for i in l for j in xrange(k)]

    distances = np.zeros(len(indices))

    for yval in np.unique(Y):
        mask = (Y == yval)
        if use_ica > 0:
            try:
                Xt[mask] = FastICA(**kwargs).fit_transform(Xt[mask])
            except ValueError:
                print("FastICA encountered NaNs. Try adding noise to the data.")
        sets = [Xt[ind][Y[ind] == yval] for ind in indices]
        distances += np.array(list(hl_distances_from_set(sets, Xt[mask])))

    distances /= len(np.unique(Y))
    distances = np.reshape(distances, (len(indices)/k, k))

    distances = distances.mean(axis=1)
    order = np.argsort(distances)
    distances = distances[order]
    subset_sizes = l[order]

    scores = [[] for c in classifiers]
    for s in indices:
        for i, clf in enumerate(classifiers):
            if np.unique(Y[s]).size > 1:
                clf.fit(X[s], Y[s])
                scores[i].append(clf.score(X, Y))
            else:
                scores[i].append(float(np.sum(Y == Y[s][0]))/X.shape[0])

    out_array = [subset_sizes, distances]
    for score, clf in zip(scores, classifiers):
        score = np.reshape(score, (len(score)/k, k)).mean(axis=1)
        score = score[order]
        out_array.append(score)
        #plt.plot(distances, score, label=clf.__class__.__name__)

    out_array = pd.DataFrame(np.array(out_array).T)
    out_array.columns = ["Subset_size", "Distance"] + [clf.__class__.__name__ for clf in classifiers]
    o = StringIO.StringIO()
    out_array.to_csv(o, sep=" ",index=False)
    sys.stdout.write(o.getvalue())
    o.close()

    #plt.xlim(0,1)
    #plt.legend(loc=4)
    #plt.gca().invert_xaxis()
    #plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculating generalization curves for a given data set.')
    parser.add_argument('data_file', help='UCI file with a data set')
    parser.add_argument('-n', '--add_noise', action="store_true", help="Adds small normal noise to the data.")
    args = parser.parse_args()

    X, Y = load_dataset(args.data_file)

    classifiers = [MajorityClassifier(), GaussianNB(), KNeighborsClassifier(),
            DecisionTreeClassifier(), RandomForestClassifier(),
            LinearSVC(), SVC()]
    classifiers = [LogisticRegression()]
    classifiers = [DecisionTreeClassifier()]

    draw_generalization_curve(X, Y, classifiers, args.add_noise)
