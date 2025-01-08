cimport numpy as np
from .tree cimport SortItem

cdef tuple[double*, int*] analyze(
                const bint task,
                const int n_classes,
                const bint linear,
                np.ndarray[double, ndim=2] X, 
                np.ndarray[double, ndim=1] y,
                np.ndarray[double, ndim=1] sample_weight, 
                const int* sample_indices,
                SortItem* sort_buffer,
                const int n_samples,
                const int n_pair, 
                const bint* is_categorical,
                object rng,
                const double gamma, 
                const int maxiter,
                const double relative_change) noexcept