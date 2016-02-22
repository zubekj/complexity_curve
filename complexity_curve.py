from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simps, trapz
from scipy.optimize import brentq


def calculate_normalized_auc(y, x):
    """
    Calculates area under the curve normalized to [0,1].
    """
    area = trapz(y, x=x)
    return float(area)/x[-1]


def hl_distances_from_set(A_list, B, points=65, margin_factor=0.25):
    """
    Calculates Hellinger distances of A_list sets from the set B using
    continuous formula.
    """
    bw = B.shape[0]**(-1.0/5)*0.5
    yBs = []
    xs = []
    for j in range(B.shape[1]):
        minx, maxx = B[:, j].min(), B[:, j].max()
        margin = (maxx-minx)*margin_factor
        minx -= margin
        maxx += margin
        xs.append(np.linspace(minx, maxx, points))
        try:
            yBs.append(gaussian_kde(B[:, j], bw_method=bw)(xs[-1]))
        except (np.linalg.linalg.LinAlgError, ValueError) as _:
            print("Singular matrix -- unable to perform gaussian KDE.")
            yBs.append(np.zeros(xs[-1].shape))

    for A in A_list:
        if A.shape[0] < 2:
            yield 1.0
        else:
            integral = 1
            for j, yB, x in zip(range(len(yBs)), yBs, xs):
                try:
                    y = (np.sqrt(gaussian_kde(A[:, j], bw_method=bw)(x)) -
                         np.sqrt(yB))**2
                    integral *= (1-0.5*simps(y, dx=(x[1]-x[0])))
                    del x, yB
                except np.linalg.linalg.LinAlgError:
                    integral = 0.0
            yield 1-integral


def complexity_curve(X, points=20, k=10, use_ica=True, **kwargs):
    """Calculates complexity curve for a given data set.

    Args:
        X (numpy array): Matrix of data points.
        points (Optional[int]): Number of points to probe along the curve.
        k (Optional[int]): Number of subsets to draw in each point.
        use_ica (Optional[bool]): Whether to preprocess data with ICA first.

    Returns:
        Numpy array of shape (points, 3) with the following columns:
        subset size, mean Hellinger distance value, standard deviation of value.
    """
    if use_ica:
        try:
            X = FastICA(**kwargs).fit_transform(X)
        except ValueError:
            print("FastICA encountered NaNs. Try adding noise to the data.")

    l = np.linspace(1, X.shape[0], points) if type(points) is int else points

    sets = [X[np.random.choice(range(X.shape[0]), int(i), replace=False)]
            for i in l for j in range(k)]
    distances = np.reshape(list(hl_distances_from_set(sets, X)),
                           (len(sets)/k, k))
    m = distances.mean(axis=1)
    s = distances.std(axis=1)

    l = np.hstack((0, l))
    m = np.hstack((1, m))
    s = np.hstack((0, s))
    return np.array([l, m, s]).T


def conditional_complexity_curve(X, y, points=20, k=10, use_ica=True,
                                 **kwargs):
    """Calculates conditional complexity curve for a given data set.

    Args:
        X (numpy array): Matrix of data points.
        y (numpy array): Vector of class labels.
        points (Optional[int]): Number of points to probe along the curve.
        k (Optional[int]): Number of subsets to draw in each point.
        use_ica (Optional[bool]): Whether to preprocess data with ICA first.

    Returns:
        Numpy array of shape (points, 3) with the following columns:
        subset size, mean Hellinger distance value, standard deviation of value.
    """
    x = X
    l = np.linspace(1, x.shape[0], points) if type(points) is int else points

    indices = [np.random.choice(range(x.shape[0]), int(i), replace=False)
               for i in l for j in range(k)]

    distances = np.zeros(len(indices))

    for yval in np.unique(y):
        mask = (y == yval)
        if use_ica > 0:
            try:
                x[mask] = FastICA(**kwargs).fit_transform(x[mask])
            except ValueError:
                print("FastICA encountered NaNs. Try adding noise to the data.")
        sets = [x[ind][y[ind] == yval] for ind in indices]
        distances += np.array(list(hl_distances_from_set(sets, x[mask])))

    distances /= len(np.unique(y))
    distances = np.reshape(distances, (len(indices)/k, k))

    m = distances.mean(axis=1)
    s = distances.std(axis=1)

    l = np.hstack((0, l))
    m = np.hstack((1, m))
    s = np.hstack((0, s))
    return np.array([l, m, s]).T


def find_minimal_subset(X, t, tol=0.1, use_ica=True, **kwargs):
    """
    Finds a minimal subset with complexity curve value below threshold.

    Args:
        X (numpy array): Matrix of data points.
        t (int): Desired similarity value.
        tol (Optional[float]): Tolerance of convergence criterion.
        use_ica (Optional[bool]): Whether to preprocess data with ICA first.

    Returns:
        Array of indices of points from the original data set compromising
        minimal subset.
    """
    if use_ica:
        try:
            X = FastICA(**kwargs).fit_transform(X)
        except ValueError:
            print("FastICA encountered NaNs. Try adding noise to the data.")

    subset = [[]]

    def subset_iter():
        while True:
            yield X[subset[0]]

    hl_distances = hl_distances_from_set(subset_iter(), X, points=29)

    def f(x):
        if(x == X.shape[0]):
            return -t
        subset[0] = np.random.choice(range(X.shape[0]), int(x),
                                     replace=False)
        xv = next(hl_distances)
        return xv-t

    brentq(f, 3, X.shape[0], xtol=1, rtol=tol)
    return subset[0]


def find_minimal_subset_cond(X, y, t, tol=0.1, use_ica=True, **kwargs):
    """
    Finds a minimal subset with complexity curve value below threshold.

    Args:
        X (numpy array): Matrix of data points.
        y (numpy array): Vector of class labels.
        t (int): Desired similarity value.
        tol (Optional[float]): Tolerance of convergence criterion.
        use_ica (Optional[bool]): Whether to preprocess data with ICA first.

    Returns:
        Array of indices of points from the original data set compromising
        minimal subset.
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
                X[mask] = FastICA(**kwargs).fit_transform(X[mask])
            except ValueError:
                print("FastICA encountered NaNs. Try adding noise to the data.")
        weights.append(float(sum(mask))/len(y))
        hl_distances.append(hl_distances_from_set(subset_iter(i), X[mask],
                            points=29))
        subsets.append([])
    weights = np.array(weights)

    def f(x):
        if(x == X.shape[0]):
            return -t
        ind[0] = np.random.choice(range(X.shape[0]), int(x), replace=False)
        for i, yval in enumerate(yvals):
            subsets[i] = X[ind][y[ind] == yval]
        xvs = np.array([next(hld) for hld in hl_distances])
        xv = np.dot(xvs, weights).sum()
        return xv-t

    brentq(f, 3, X.shape[0], xtol=1, rtol=tol)
    return ind[0]


def generalization_curve(X, y, classifiers, add_noise=False, k=20,
                         use_ica=True, **kwargs):
    """
    Calculates generalization curve for a given data and classifiers.

    Args:
        X (numpy array): Matrix of data points.
        y (numpy array): Vector of class labels.
        classifiers (array): List of scikit-learn classifiers to test.
        add_noise (Optional[bool]): Whether to add small noise to the data
                                    (increases stability).
        k (Optional[int]): Number of subsets drawn at each point.
        use_ica (Optional[bool]): Whether to preprocess data with ICA first.

    Returns:
        Numpy array of shape (30, len(classifiers)+2) with the following
        columns: subset size, mean Hellinger distance, mean score for a given
        classifier.
    """
    Xt = X.copy()
    if add_noise:
        Xt += np.random.normal(0, 0.01, Xt.shape)

    l = np.hstack((np.linspace(1, 10, 10),
                   np.logspace(1, np.log10(X.shape[0]), 20)[1:]))

    indices = [np.random.choice(range(Xt.shape[0]), int(i), replace=False)
               for i in l for j in range(k)]

    distances = np.zeros(len(indices))

    for yval in np.unique(y):
        mask = (y == yval)
        if use_ica > 0:
            try:
                Xt[mask] = FastICA(**kwargs).fit_transform(Xt[mask])
            except ValueError:
                print("FastICA encountered NaNs. Try adding noise to the data.")
        sets = [Xt[ind][y[ind] == yval] for ind in indices]
        distances += np.array(list(hl_distances_from_set(sets, Xt[mask])))

    distances /= len(np.unique(y))
    distances = np.reshape(distances, (len(indices)/k, k))

    distances = distances.mean(axis=1)
    order = np.argsort(distances)
    distances = distances[order]
    subset_sizes = l[order]

    scores = [[] for c in classifiers]
    for s in indices:
        for i, clf in enumerate(classifiers):
            if np.unique(y[s]).size > 1:
                clf.fit(X[s], y[s])
                scores[i].append(clf.score(X, y))
            else:
                scores[i].append(float(np.sum(y == y[s][0]))/X.shape[0])

    out_array = [subset_sizes, distances]
    for score, clf in zip(scores, classifiers):
        score = np.reshape(score, (len(score)/k, k)).mean(axis=1)
        score = score[order]
        out_array.append(score)

    return np.array(out_array).T


if __name__ == '__main__':

    X = np.random.random((100, 10))
    y = (np.random.random(100) > 0.5)

    print(complexity_curve(X))

    print(conditional_complexity_curve(X, y))

    print(find_minimal_subset(X, 0.01))

    print(find_minimal_subset_cond(X, y, 0.01))

    from sklearn.tree import DecisionTreeClassifier

    print(generalization_curve(X, y, [DecisionTreeClassifier()]))
