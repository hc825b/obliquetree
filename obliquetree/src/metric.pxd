from .tree cimport CategoryStat
from .utils cimport SortItem

cdef double calculate_impurity(
                const bint is_categorical, 
                const int n_classes,
                SortItem* sort_buffer,
                const double* sample_weight,
                const double* y,
                int* nan_indices,
                CategoryStat* categorical_stats,
                const int n_samples,
                const int n_nans,
                const int min_samples_leaf,
                double *threshold_c,
                int* left_count_c,
                bint* missing_go_left,
                const bint task) noexcept nogil

cdef double calculate_node_value(const double[::1] sample_weight, 
                            const double[::1] y, 
                            const int* sample_indices,
                            const int n_samples) noexcept nogil

cdef double calculate_node_gini(const int* sample_indices, 
                                const double[::1] sample_weight, 
                                const double[::1] y, 
                                const int n_samples, 
                                int n_class = ?) noexcept nogil

cdef double calculate_node_mse(const int* sample_indices,
                             const double[::1] sample_weight,
                             const double[::1] y,
                             const int n_samples) noexcept nogil

cdef void calculate_node_value_multiclass(const double[::1] sample_weight,
                                           const double[::1] y,
                                           const int* sample_indices,
                                           const int n_samples,
                                           const int n_classes,
                                           double** class_probs) noexcept nogil                        