from .tree cimport TreeNode

cdef struct WeakestLink:
    TreeNode* node
    double improvement_score
    int size_diff

cdef void prune_tree(TreeNode* root, const double alpha) noexcept nogil