from itertools import izip
from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simps, trapz
from scipy.optimize import brentq
import argparse

from common import load_dataset
#import fastica_ext

#from theano_tanh import tanh

def find_minimal_subset(data, t, tol=0.1, use_ica=True, **kwargs):
    """
    Finds approximate size of a minimal subset with Hellinger distance below threshold.
    """
    if use_ica:
        try:
            data = FastICA(**kwargs).fit_transform(data)
        except ValueError:
            print("FastICA encountered NaNs. Try adding noise to the data.")

    subset = [[]]

    def subset_iter():
        while True:
            yield data[subset[0]]

    hl_distances = hl_distances_from_set(subset_iter(), data, points=29)

    def f(x):
        if(x==data.shape[0]):
            #print(x,0)
            return -t
        subset[0] = np.random.choice(range(data.shape[0]), int(x), replace=False)
        xv = hl_distances.next()
        #print(x,xv)
        return xv-t

    brentq(f, 3, data.shape[0], xtol=1, rtol=tol)
    return subset[0]

def find_minimal_subset_cond(data, y, t, tol=0.1, use_ica=True, **kwargs):
    """
    Finds approximate size of a minimal subset with Hellinger
    distance below threshold.
    """
    ind = [[]]
    subsets = []
    weights = []
    yvals = np.unique(y)

    def subset_iter(i):
        while True:
            yield subsets[i]

    hl_distances = []
    for i, yval in enumerate(yvals):
        mask = (y == yval)
        if use_ica:
            try:
                data[mask] = FastICA(**kwargs).fit_transform(data[mask])
            except ValueError:
                print("FastICA encountered NaNs. Try adding noise to the data.")
        weights.append(float(sum(mask))/len(y))
        hl_distances.append(hl_distances_from_set(subset_iter(i), data[mask],
                points=29))
        subsets.append([])
    weights = np.array(weights)

    def f(x):
        if(x==data.shape[0]):
            #print(x,0)
            return -t
        ind[0] = np.random.choice(range(data.shape[0]), int(x), replace=False)
        for i, yval in enumerate(yvals):
            subsets[i] = data[ind][y[ind] == yval]
        xvs = np.array([hld.next() for hld in hl_distances])
        xv = np.dot(xvs, weights).sum()
        #print(x,xv)
        return xv-t

    brentq(f, 3, data.shape[0], xtol=1, rtol=tol)
    return ind[0]

def hl_distances_from_set(A_list, B, points=65, margin_factor=0.25):
    """
    Calculates Hellinger distances of A_list sets from the set B using
    continuous formula.
    """
    bw = B.shape[0]**(-1.0/5)*0.5
    yBs = []
    xs = []
    for j in xrange(B.shape[1]):
        minx, maxx = B[:,j].min(), B[:,j].max()
        margin = (maxx-minx)*margin_factor
        minx -= margin
        maxx += margin
        xs.append(np.linspace(minx, maxx, points))
        try:
            yBs.append(gaussian_kde(B[:,j],bw_method=bw)(xs[-1]))
        except (np.linalg.linalg.LinAlgError, ValueError) as e:
            print("Singular matrix -- unable to perform gaussian KDE.")
            yBs.append(np.zeros(xs[-1].shape))

    distances = []
    for A in A_list:
        if A.shape[0] < 2:
            yield 1.0
        else:
            integral = 1
            for j, yB, x in izip(xrange(len(yBs)), yBs, xs):
                #y = np.sqrt(gaussian_kde(A[:,j],bw_method=bw)(x)*yB)
                #integral *= simps(y, dx=(x[1]-x[0]))
                try:
                    y = (np.sqrt(gaussian_kde(A[:,j],bw_method=bw)(x))-np.sqrt(yB))**2
                    integral *= (1-0.5*simps(y, dx=(x[1]-x[0])))
                    del x, yB
                except np.linalg.linalg.LinAlgError:
                    integral = 0.0
            yield 1-integral

def complexity_curve(data, points=20, k=10, use_ica=True, **kwargs):
    """
    Calculates complexity curve for a given data set.
    """
    if use_ica:
        try:
            x = FastICA(**kwargs).fit_transform(data)
        except ValueError:
            print("FastICA encountered NaNs. Try adding noise to the data.")
    else:
        x = data

    l = np.linspace(1, x.shape[0], points) if type(points) is int else points

    sets = [x[np.random.choice(range(x.shape[0]), int(i), replace=False)]
                for i in l for j in xrange(k)]
    distances = np.reshape(list(hl_distances_from_set(sets, x)), (len(sets)/k, k))
    m = distances.mean(axis=1)
    s = distances.std(axis=1)

    l = np.hstack((0, l))
    m = np.hstack((1, m))
    s = np.hstack((0, s))
    return np.array([l,m,s]).T

def conditional_complexity_curve(data, y, points=20, k=10, use_ica=True, **kwargs):
    """
    Calculates complexity curve for a given data set conditioned on class.
    """
    #x = data+np.random.normal(0, 0.001, data.shape) # Small noise added.
    x = data
    l = np.linspace(1, x.shape[0], points) if type(points) is int else points

    indices = [np.random.choice(range(x.shape[0]), int(i), replace=False)
                for i in l for j in xrange(k)]

    distances = np.zeros(len(indices))

    for yval in np.unique(y):
        mask = (y == yval)
        if use_ica > 0:
            try:
                x[mask] = FastICA(**kwargs).fit_transform(x[mask])
            except ValueError:
                print("FastICA encountered NaNs. Try adding noise to the data.")
        #p = float(sum(mask))/len(y)
        sets = [x[ind][y[ind] == yval] for ind in indices]
        distances += np.array(list(hl_distances_from_set(sets, x[mask])))

    distances /= len(np.unique(y))
    distances = np.reshape(distances, (len(indices)/k, k))

    m = distances.mean(axis=1)
    s = distances.std(axis=1)

    l = np.hstack((0, l))
    m = np.hstack((1, m))
    s = np.hstack((0, s))
    return np.array([l,m,s]).T

def calculate_normalized_auc(y, x):
    """
    Calculates area under a curve normalized to [0,1].
    """
    area = trapz(y, x=x)
    return float(area)/x[-1]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculating complexity curve for a given data set.')
    parser.add_argument('data_file', help='UCI file with a data set')
    parser.add_argument('-c', '--conditional_curve', action="store_true",
            help="Calculates complexity curve conditioned on class.")
    parser.add_argument('-m', '--minimal_subset', type=float, help="Finds subset of specified distance.")
    parser.add_argument('-i', '--max_iter', type=int, default=100, help="Maximum number of iterations of FastICA.")
    parser.add_argument('-n', '--add_noise', action="store_true", help="Adds small normal noise to the data.")
    args = parser.parse_args()

    data, y = load_dataset(args.data_file)

    if(args.add_noise):
        data += np.random.normal(0, 0.01, data.shape)

    if(args.minimal_subset):
        if args.conditional_curve:
            subset = find_minimal_subset_cond(data, y, args.minimal_subset,
                                                max_iter=args.max_iter)
        else:
            subset = find_minimal_subset(data, args.minimal_subset,
                                            max_iter=args.max_iter)
        print(subset.shape[0], float(subset.shape[0])/data.shape[0])
    else:
        points = 60
        if args.conditional_curve:
            cc = conditional_complexity_curve(data, y, points=points,
                                              max_iter=args.max_iter)
        else:
            cc = complexity_curve(data, points=points,
                    max_iter=args.max_iter)
        for r in cc:
            print("{0} {1} {2}".format(r[0], r[1], r[2]))
