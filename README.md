# Complexity curve

Script for calculating complexity of a data set according to complexity curve methodology.

More information: https://peerj.com/articles/cs-76/

Script contains the following functions:
* complexity_curve -- calculates full complexity curve for a given data set.
* conditional_complexity_curve -- calculates conditional complexity curve for a given data set.
* find_minimal_subset -- finds minimal subset of given data set with acceptable similarity to the original set (according to a given threshold).
* find_minimal_subset_cond -- finds minimal subset of given data set using conditional complexity curve.

Usage example:

```python
import numpy as np
from complexity_curve import *

X = np.random.random((100, 10))
y = (np.random.random(100) > 0.5)

print(complexity_curve(X))

print(conditional_complexity_curve(X, y))

print(find_minimal_subset(X, 0.01))

print(find_minimal_subset_cond(X, y, 0.01))
```
    
