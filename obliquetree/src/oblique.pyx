from libc.math cimport INFINITY, exp, log
from libc.string cimport strncpy

import numpy as np
cimport numpy as np

from scipy.optimize import _lbfgsb
from scipy.linalg.cython_blas cimport dgemv
from libc.stdlib cimport malloc
from .utils cimport sort_pointer_array

from itertools import combinations

cdef inline double get_y(double y, double current_class) noexcept nogil:
    return 1.0 if y == current_class else 0.0

cdef inline double sigmoid(const double x) noexcept nogil:
    return 1.0 / (1.0 + exp(-x))

cdef double fun_and_grad_nogil(
    double[::1, :] X,        # (N, d), FORTRAN-ordered (col-major)
    const double[::1] y,     # (N,)
    const double[::1] sample_weight,
    double[::1] w,           # (d,)
    const Py_ssize_t N,      # number of samples
    const Py_ssize_t d,      # number of features
    const double gamma,
    const double eps,
    const double total_weight,
    double[::1] z,           # (N,)  preallocated: X.dot(w)
    double[::1] p,           # (N,)  preallocated: sigmoid(z)
    double[::1] dp_dz,       # (N,)  preallocated
    double[::1] grad_w,      # (d,)  preallocated (output: gradient)
    double[::1] dP_R1_w,
    double[::1] dP_L1_w,
    double[::1] tmp_y_dp_vec,
    double[::1] dS_R_w,
    double[::1] y_dp_vec) noexcept nogil:
    """
    Sample-weighted Gini impurity calculation and gradient computation.
    Incorporates sample weights into the calculations while maintaining
    the efficient BLAS-based implementation.
    """
    cdef char trans
    cdef int m_ = <int>N
    cdef int n_ = <int>d
    cdef int incx_ = 1
    cdef int incy_ = 1
    cdef int lda_ = m_
    cdef double alpha = 1.0
    cdef double beta = 0.0

    cdef Py_ssize_t i
    cdef double S_L, S_R, p_y_sum, p_1_y_sum, tmp
    cdef double P_L1, P_R1
    cdef double impurity_L, impurity_R
    cdef double loss
    cdef double factor_L, factor_R
    cdef double dImp_L, dImp_R
    cdef double denom_L, denom_R
    cdef double tmp_y_dp, gj
    cdef double one_minus_tmp, yi, sw_i
    cdef double dp_dz_i, dS_R_w_i, dP_L1_w_i, dP_R1_w_i

    # 1) z = X dot w (unchanged)
    trans = b'N'
    dgemv(
        &trans,
        &m_, &n_,
        &alpha,
        &X[0, 0], &lda_,
        &w[0], &incx_,
        &beta,
        &z[0], &incy_
    )

    # 2) Calculate weighted probabilities and sums
    S_L = 0.0
    p_y_sum = 0.0
    p_1_y_sum = 0.0
    
    for i in range(N):
        sw_i = sample_weight[i]  # Get sample weight
        
        tmp = sigmoid(z[i])
        one_minus_tmp = 1.0 - tmp
        p[i] = tmp
        yi = y[i]
        
        # Apply sample weights to all sums
        S_L += tmp * sw_i
        p_y_sum += tmp * yi * sw_i
        p_1_y_sum += one_minus_tmp * yi * sw_i
        
        # Adjust gradient calculations for sample weights
        dp_dz_i = gamma * tmp * one_minus_tmp * sw_i
        dp_dz[i] = dp_dz_i
        y_dp_vec[i] = yi * dp_dz_i

    # Add eps scaled by total weight for numerical stability
    S_L += total_weight
    S_R = total_weight - S_L

    S_L += eps
    S_R += eps

    # 4) Weighted Gini parameters
    P_L1 = p_y_sum / S_L
    P_R1 = p_1_y_sum / S_R
    impurity_L = P_L1 * (1.0 - P_L1)
    impurity_R = P_R1 * (1.0 - P_R1)
    loss = S_L * impurity_L + S_R * impurity_R

    # 6) Weighted gradient calculation
    trans = b'T'
    dgemv(
        &trans,
        &m_, &n_,
        &alpha,
        &X[0, 0], &lda_,
        &dp_dz[0], &incx_,
        &beta,
        &grad_w[0], &incy_
    )

    for i in range(d):
        dS_R_w[i] = -grad_w[i]

    # 8) Weighted derivatives calculation
    denom_L = S_L * S_L
    denom_R = S_R * S_R
    factor_L = 1.0 - 2.0 * P_L1
    factor_R = 1.0 - 2.0 * P_R1

    trans = b'T'
    dgemv(
        &trans,
        &m_, &n_,
        &alpha,
        &X[0, 0], &lda_,
        &y_dp_vec[0], &incx_,
        &beta,
        &tmp_y_dp_vec[0], &incy_
    )

    for i in range(d):
        tmp_y_dp = tmp_y_dp_vec[i]
        gj = grad_w[i]
        dS_R_w_i = dS_R_w[i]
        dP_L1_w_i = (tmp_y_dp * S_L - p_y_sum * gj) / denom_L
        dP_R1_w_i = (-tmp_y_dp * S_R - p_1_y_sum * dS_R_w_i) / denom_R
        dP_L1_w[i] = dP_L1_w_i
        dP_R1_w[i] = dP_R1_w_i
        dImp_L = factor_L * dP_L1_w_i
        dImp_R = factor_R * dP_R1_w_i
        grad_w[i] = (gj * impurity_L
                    + S_L * dImp_L
                    + dS_R_w_i * impurity_R
                    + S_R * dImp_R)

    return loss

cdef double fun_and_grad_linear_reg_nogil(
    double[::1, :] X,             # shape=(N, d), Fortran-ordered
    const double[::1] y,          # shape=(N,)
    const double[::1] sample_weight,
    double[::1] w,                # shape=(d,)
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double total_weight,    # sum of sample weights
    double[::1] pred,             # shape=(N,)  preallocated for predictions
    double[::1] weighted_residuals,# shape=(N,)  preallocated
    double[::1] grad_w ) noexcept nogil: # shape=(d,)  preallocated (output)
    """
    Compute the weighted MSE loss and gradient for linear regression.
    Uses BLAS operations for efficiency.
    
    The loss function is:
    L = (1/2) * sum(sample_weight_i * (y_i - X_i dot w)^2) / total_weight
    
    The gradient is:
    grad = -X.T dot (sample_weight * (y - X dot w)) / total_weight
    """
    cdef char trans
    cdef int m_ = <int>N
    cdef int n_ = <int>d
    cdef int incx_ = 1
    cdef int incy_ = 1
    cdef int lda_ = m_
    cdef double alpha = 1.0
    cdef double beta = 0.0
    cdef Py_ssize_t i
    cdef double loss = 0.0
    cdef double residual
    cdef double sw_scaled

    # 1. Compute predictions: pred = X dot w
    trans = b'N'
    dgemv(&trans,
          &m_, &n_,
          &alpha,
          &X[0, 0], &lda_,
          &w[0], &incx_,
          &beta,
          &pred[0], &incy_)
    
    # 2. Compute weighted residuals and loss
    for i in range(N):
        residual = y[i] - pred[i]
        sw_scaled = sample_weight[i] / total_weight  # Scale the weights
        weighted_residuals[i] = residual * sw_scaled
        loss += 0.5 * sw_scaled * residual * residual
    
    # 3. Compute gradient: grad = -X.T dot weighted_residuals
    trans = b'T'
    alpha = -1.0  # No need to divide by total_weight here since we scaled the residuals
    beta = 0.0
    dgemv(&trans,
          &m_, &n_,
          &alpha,
          &X[0, 0], &lda_,
          &weighted_residuals[0], &incx_,
          &beta,
          &grad_w[0], &incy_)
    
    return loss

cdef double fun_and_grad_multiclass_nogil(
    double[::1, :] X,           # (N, d), FORTRAN-ordered
    const double[::1] y,                # (N,) class labels: 0,1,2,...
    const double[::1] sample_weight,  # (N,)
    double[::1] w,              # (d,)
    const Py_ssize_t N,         # number of samples
    const Py_ssize_t d,         # number of features
    const Py_ssize_t n_classes, # number of classes
    const double gamma,
    const double eps,
    const double total_weight,
    double[::1] z,              # (N,) preallocated
    double[::1] p,              # (N,) preallocated
    double[::1] dp_dz,          # (N,) preallocated
    double[::1] grad_w,         # (d,) preallocated
    double[::1] class_counts_L, # (n_classes,) preallocated
    double[::1] class_counts_R, # (n_classes,) preallocated
    double[::1] tmp_dp_vec,     # (d,) preallocated
    double[::1] dS_R_w,          # (d,) preallocated
    double[::1] P_k_L,
    double[::1] P_k_R) noexcept nogil:
    cdef:
        Py_ssize_t i, j, k
        char trans
        int m_ = <int>N
        int n_ = <int>d
        int incx_ = 1
        int incy_ = 1
        int lda_ = m_
        double alpha = 1.0
        double beta = 0.0
        double S_L = 0.0
        double S_R = 0.0
        double impurity_L = 0.0
        double impurity_R = 0.0
        double loss = 0.0
        double sw_i, tmp, one_minus_tmp
        double denom_L, denom_R
        double tmp_dp, gj
        int yi
    
    # 1) z = X dot w (unchanged)
    trans = b'N'
    dgemv(
        &trans,
        &m_, &n_,
        &alpha,
        &X[0, 0], &lda_,
        &w[0], &incx_,
        &beta,
        &z[0], &incy_
    )
    
    # 2) Initialize counts and compute probabilities
    for k in range(n_classes):
        class_counts_L[k] = 0.0
        class_counts_R[k] = 0.0
    
    # 3) Calculate weighted probabilities and sums
    for i in range(N):
        sw_i = sample_weight[i]
        yi = <int>y[i]
        
        tmp = 1.0 / (1.0 + exp(-z[i]))
        one_minus_tmp = 1.0 - tmp
        p[i] = tmp
        
        # Update class counts and totals
        class_counts_L[yi] += tmp * sw_i
        class_counts_R[yi] += one_minus_tmp * sw_i
        S_L += tmp * sw_i
        S_R += one_minus_tmp * sw_i
        
        # Store gradient components
        dp_dz[i] = gamma * tmp * one_minus_tmp * sw_i
    
    # Add eps for stability
    S_L += eps 
    S_R += eps

    # 4) Calculate class proportions and impurity
    denom_L = S_L * S_L
    denom_R = S_R * S_R
    
    for k in range(n_classes):
        P_k_L[k] = class_counts_L[k] / S_L
        P_k_R[k] = class_counts_R[k] / S_R
        impurity_L += P_k_L[k] * (1.0 - P_k_L[k])
        impurity_R += P_k_R[k] * (1.0 - P_k_R[k])
    
    loss = S_L * impurity_L + S_R * impurity_R
    
    # 5) Gradient calculation - first part
    trans = b'T'
    dgemv(
        &trans,
        &m_, &n_,
        &alpha,
        &X[0, 0], &lda_,
        &dp_dz[0], &incx_,
        &beta,
        &grad_w[0], &incy_
    )
    
    for i in range(d):
        dS_R_w[i] = -grad_w[i]
    
    # 6) Class-wise gradient contributions
    for k in range(n_classes):
        # Similar to binary case but summed over classes
        for i in range(d):
            gj = grad_w[i]
            dS_R_w_i = dS_R_w[i]
            
            # Similar structure to binary version but for each class
            tmp_dp = (class_counts_L[k] * gj / denom_L) * (1.0 - 2.0 * P_k_L[k])
            tmp_dp += (class_counts_R[k] * dS_R_w_i / denom_R) * (1.0 - 2.0 * P_k_R[k])
            
            grad_w[i] += (gj * impurity_L +
                         S_L * tmp_dp +
                         dS_R_w_i * impurity_R +
                         S_R * (-tmp_dp))
    
    return loss
    
cdef double fun_and_grad_reg_nogil(
    double[::1, :] X,             # shape=(N, d), Fortran-ordered
    const double[::1] y,          # shape=(N,)
    const double[::1] sample_weight,
    double[::1] w,                # shape=(d,)
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double gamma,
    const double eps,
    const double total_weight,          # sum of sample weights
    double[::1] z,                # shape=(N,)  preallocated
    double[::1] p,                # shape=(N,)  preallocated
    double[::1] dp_dz,            # shape=(N,)  preallocated
    double[::1] grad_w,           # shape=(d,)  preallocated (output)
    double[::1] temp_vec1,        # shape=(N,)  preallocated
    double[::1] temp_vec2,        # shape=(N,)  preallocated
    double[::1] temp_grad) noexcept nogil:         # shape=(d,)  preallocated
    cdef char trans
    cdef int m_ = <int>N
    cdef int n_ = <int>d
    cdef int incx_ = 1
    cdef int incy_ = 1
    cdef int lda_ = m_
    cdef double alpha = 1.0
    cdef double beta = 0.0

    cdef Py_ssize_t i, j
    cdef double loss = 0.0
    cdef double S_L = 0.0
    cdef double M_L = 0.0
    cdef double S_R = 0.0
    cdef double M_R = 0.0
    cdef double mL, mR
    cdef double p_val, dp_val, sw_i
    cdef double diffL, diffR, tmp
    cdef double dS_L_w
    cdef double dM_L_w
    cdef double d_mL_w
    cdef double d_mR_w
    for i in range(d):
        grad_w[i] = 0.0
        temp_grad[i] = 0.0

    trans = b'N'  # X is (N,d), w is (d,); result z is length N
    dgemv(&trans, &m_, &n_,
          &alpha,
          &X[0, 0], &lda_,
          &w[0], &incx_,
          &beta,
          &z[0], &incy_)

    for i in range(N):
        sw_i = sample_weight[i]
        p_val = sigmoid(gamma * z[i])
        p[i] = p_val

        dp_val = gamma * p_val * (1.0 - p_val)
        dp_dz[i] = dp_val

        S_L += sw_i * p_val
        M_L += sw_i * p_val * y[i]

        S_R += sw_i * (1.0 - p_val)
        M_R += sw_i * (1.0 - p_val) * y[i]

    S_L += eps
    S_R += eps

    mL = M_L / S_L
    mR = M_R / S_R

    for i in range(N):
        sw_i = sample_weight[i]

        diffL = y[i] - mL
        diffR = y[i] - mR

        # Accumulate to the total MSE-like loss
        loss += sw_i * p[i] * (diffL * diffL)
        loss += sw_i * (1.0 - p[i]) * (diffR * diffR)
        temp_vec1[i] = dp_dz[i] * sw_i * (diffL * diffL - diffR * diffR)

    loss /= total_weight

    trans = b'T'
    alpha = 1.0
    beta = 0.0
    dgemv(&trans, &m_, &n_,
          &alpha,
          &X[0, 0], &lda_,
          &temp_vec1[0], &incx_,
          &beta,
          &grad_w[0], &incy_)

    for i in range(d):
        temp_grad[i] = 0.0

    # Instead of a Python loop: multiply dp_dz[i]*sw_i once, store in temp_vec2, then do dgemv
    for i in range(N):
        temp_vec2[i] = dp_dz[i] * sample_weight[i]

    trans = b'T'
    alpha = 1.0
    beta = 0.0
    dgemv(&trans, &m_, &n_,
          &alpha,
          &X[0, 0], &lda_,
          &temp_vec2[0], &incx_,
          &beta,
          &temp_grad[0], &incy_)

    alpha = 1.0
    beta = 1.0

    for j in range(d):
        dS_L_w = temp_grad[j]
        dM_L_w = temp_grad[j]  # matches original logic
        d_mL_w = (S_L * dM_L_w - M_L * dS_L_w) / (S_L * S_L)
        d_mR_w = (S_R * (-dM_L_w) - M_R * (-dS_L_w)) / (S_R * S_R)

        for i in range(N):
            sw_i = sample_weight[i]
            temp_vec1[i] = -2.0 * sw_i * (
                p[i] * (y[i] - mL) * d_mL_w +
                (1.0 - p[i]) * (y[i] - mR) * d_mR_w
            )

        dgemv(&trans, &m_, &n_,
              &alpha,
              &X[0, 0], &lda_,
              &temp_vec1[0], &incx_,
              &alpha,
              &grad_w[0], &incy_)

    for i in range(d):
        grad_w[i] /= total_weight

    return loss

cdef double fun_and_grad_binary_linear_nogil(
    const double current_class,
    double[::1, :] X,             # shape=(N, d), Fortran-ordered
    const double[::1] y,          # shape=(N,)
    const double[::1] sample_weight,
    double[::1] w,                # shape=(d,)
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double total_weight,    # sum of sample weights
    double[::1] pred,             # shape=(N,)  preallocated for predictions
    double[::1] weighted_residuals,# shape=(N,)  preallocated
    double[::1] grad_w ) noexcept nogil: # shape=(d,)  preallocated (output)
    """
    Compute the weighted binary cross-entropy loss and gradient for linear classification.
    Uses BLAS operations for efficiency.
    
    The loss function is:
    L = -sum(sample_weight_i * (y_i * log(sigmoid(X_i dot w)) + (1-y_i) * log(1-sigmoid(X_i dot w)))) / total_weight
    """
    cdef char trans
    cdef int m_ = <int>N
    cdef int n_ = <int>d
    cdef int incx_ = 1
    cdef int incy_ = 1
    cdef int lda_ = m_
    cdef double alpha = 1.0
    cdef double beta = 0.0
    cdef Py_ssize_t i
    cdef double loss = 0.0
    cdef double prob, sw_scaled
    
    # 1. Compute linear predictions: pred = X dot w
    trans = b'N'
    dgemv(&trans,
          &m_, &n_,
          &alpha,
          &X[0, 0], &lda_,
          &w[0], &incx_,
          &beta,
          &pred[0], &incy_)
    
    # 2. Compute probabilities, weighted residuals and loss
    for i in range(N):
        y_i = get_y(y[i], current_class)
        prob = sigmoid(pred[i])
        sw_scaled = sample_weight[i] / total_weight
        
        # Compute cross-entropy loss
        if y_i:  # y[i] == 1
            loss -= sw_scaled * log(prob + 1e-15)
        else:  # y[i] == 0
            loss -= sw_scaled * log(1.0 - prob + 1e-15)
            
        # Store residuals for gradient computation
        weighted_residuals[i] = sw_scaled * (prob - y_i)
    
    # 3. Compute gradient: grad = X.T dot weighted_residuals
    trans = b'T'
    dgemv(&trans,
          &m_, &n_,
          &alpha,
          &X[0, 0], &lda_,
          &weighted_residuals[0], &incx_,
          &beta,
          &grad_w[0], &incy_)
    
    return loss


cdef my_lbfgs_b_minimize(
    const bint task_,
    const int n_classes,
    const double current_class,
    const bint linear,
    np.ndarray[double, ndim=1] x0,        # başlangıç parametreleri (d,)
    double[::1, :] X,  # shape=(N,d) fortran
    const double[::1] y,        # shape=(N,)
    const double[::1] sample_weight,
    const double sum_sample_weight,
    Py_ssize_t N,
    Py_ssize_t d,
    double[::1] z,
    double[::1] p,
    double[::1] dp_dz,
    np.ndarray[double, ndim=1] grad_w,
    double[::1] dP_R1_w,
    double[::1] dP_L1_w,
    double[::1] tmp_y_dp_vec,
    double[::1] dS_R_w,
    double[::1] y_dp_vec,
    double[::1] temp_vec1,
    double[::1] temp_vec2,
    double[::1] temp_grad,
    double[::1] class_counts_L,
    double[::1] class_counts_R,
    double[::1] P_k_L,
    double[::1] P_k_R,
    np.ndarray[double, ndim=1] lower_bnd,
    np.ndarray[double, ndim=1] upper_bnd,
    np.ndarray[int, ndim=1] nbd,
    double gamma_ = 1.0,
    int maxiter = 100,
    double relative_change = 1e-4,
    double eps_ = 1e-6,
    double factr_ = 1e7,   # L-BFGS-B param
    double pgtol = 1e-5,  # L-BFGS-B param
    int m = 10,           # L-BFGS-B memory
    int iprint = -1,
    int maxls = 20):
    """
    Çok basit bir L-BFGS-B döngüsü. 'fun_and_grad_nogil' fonksiyonunu
    kullanarak f ve g'yi hesaplayacak ve setulb'yi çağıracak.
    x0 dizisini yerinde güncelleyip en sonunda return edecek.

    lower_bnd, upper_bnd, nbd arrayleri kullanıcı tarafından
    doğru şekilde ayarlanmış olmalı.
    """
    cdef int n = x0.shape[0]

    # L-BFGS-B parametreleri
    cdef double factr = factr_ * <double>np.finfo(np.double).eps  # scipynin convert ettiği gibi

    # Çeşitli workspace ler:
    #  'wa' boyutu = 2*m*n + 5*n + 11*m*m + 8*m   (bkz. lbfgsb.f)
    cdef int wa_size = 2*m*n + 5*n + 11*m*m + 8*m
    cdef np.ndarray[double, ndim=1] wa = np.zeros(wa_size, dtype=np.float64)

    # iwa boyutu = 3*n
    cdef int iwa_size = 3 * n
    cdef np.ndarray[int, ndim=1] iwa = np.zeros(iwa_size, dtype=np.int32)

    cdef double f_val = 0.0
    cdef double f_old = INFINITY

    # Fortran setulb() icin char array
    cdef np.ndarray[np.uint8_t, ndim=1] task = np.zeros(60, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] csave = np.zeros(60, dtype=np.uint8)

    # lsave(4), isave(44), dsave(29)
    cdef np.ndarray[int, ndim=1] lsave = np.zeros(4,   dtype=np.int32)
    cdef np.ndarray[int, ndim=1] isave = np.zeros(44,  dtype=np.int32)
    cdef np.ndarray[double, ndim=1] dsave = np.zeros(29, dtype=np.float64)

    strncpy(<char*> &task[0], b"START", 5)

    cdef bytes task_bytes
    cdef bytes tstrip

    # L-BFGS-B iterasyon döngüsü:
    for i in range(maxiter):
        _lbfgsb.setulb(m, x0, lower_bnd, upper_bnd,
                  nbd,
                  f_val, grad_w,
                  factr, pgtol,
                  wa, iwa,
                  task, iprint, csave, lsave, isave, dsave, maxls)


        task_bytes = task.tobytes()  # b'START\x00\x00...'
        tstrip = task_bytes.rstrip(b'\x00')

        if tstrip.startswith(b"FG"):

            if task_ == 0:
                if linear:
                    f_val = fun_and_grad_binary_linear_nogil(current_class, X, y, sample_weight, x0, N, d, sum_sample_weight,temp_vec1, temp_vec2, grad_w)

                else:
                    if n_classes > 2:
                        f_val = fun_and_grad_multiclass_nogil(X, y,sample_weight,x0, N, d, n_classes,
                                            gamma_, eps_, sum_sample_weight,
                                            z, p, dp_dz, grad_w,class_counts_L,
                                            class_counts_R, tmp_y_dp_vec, dS_R_w, P_k_L, P_k_R)   

                    else:
                        f_val = fun_and_grad_nogil(X, y,sample_weight,x0, N, d,
                                            gamma_, eps_, sum_sample_weight,
                                            z, p, dp_dz, grad_w,dP_R1_w,
                                            dP_L1_w, tmp_y_dp_vec, dS_R_w,
                                            y_dp_vec)   
                                                
            else:
                if linear:
                    f_val = fun_and_grad_linear_reg_nogil(X, y, sample_weight, x0, N, d, sum_sample_weight, temp_vec1, temp_vec2, grad_w)
                else:
                    f_val = fun_and_grad_reg_nogil(X, y,sample_weight,x0, N, d,
                                                gamma_, eps_, sum_sample_weight, z, p, dp_dz, grad_w, temp_vec1,
                                                temp_vec2, temp_grad)

            if (((f_old - f_val) / f_old) <= relative_change) and (f_old != INFINITY):
                break

            f_old = f_val

        if tstrip.startswith(b'CONV'):
            break

    return x0, f_val

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
                const double relative_change,
              ) noexcept:

    cdef np.ndarray[int, ndim=1] sample_dx = np.frombuffer(<bytes>(<char*>sample_indices)[:n_samples * sizeof(int)], dtype=np.int32)
    X = X[sample_dx, :].copy(order="F") # TODO use transpose to avoid copy
    y = y[sample_dx].copy()
    sample_weight = sample_weight[sample_dx].copy()

    cdef double sum_sample_weight = sample_weight.sum()
    cdef Py_ssize_t N = X.shape[0], d = X.shape[1], i, min_idx
    cdef list feature_range 
    cdef list feature_pairs 

    cdef int* best_pair = <int*>malloc(n_pair * sizeof(int))
    cdef double* best_x = <double*>malloc(n_pair * sizeof(double))

    cdef double best_fx = INFINITY
    cdef Py_ssize_t n_left_can, n_left = 0

    if is_categorical:
        feature_range = [i for i in list(range(d)) if not is_categorical[i]]
        if len(feature_range) == 0:
            return best_x, best_pair

        feature_pairs = list(combinations(feature_range, n_pair))

    else:
        feature_pairs = list(combinations(range(d), n_pair))

    cdef np.ndarray[double, ndim=1] x0
    cdef Py_ssize_t n_feature_pairs = len(feature_pairs)
    cdef tuple pair
    cdef double f_val

    # preallocated buffers:
    cdef double[::1] z       = np.empty(N, dtype=np.float64)
    cdef double[::1] p       = np.empty(N, dtype=np.float64)
    cdef double[::1] dp_dz   = np.empty(N, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] grad_w  = np.empty(n_pair,   dtype=np.float64)
    
    
    cdef double[::1] dP_R1_w, dP_L1_w, tmp_y_dp_vec, dS_R_w, y_dp_vec #class
    cdef double[::1] temp_vec1, temp_vec2, temp_grad #reg
    cdef double[::1] class_counts_L, class_counts_R, P_k_L, P_k_R #mul

    if task == 0:
        if linear:
            temp_vec1 = np.empty(N, dtype=np.float64)   # (N,)  preallocated for calculations
            temp_vec2 = np.empty(N, dtype=np.float64)   # (N,)  preallocated for calculations

        dS_R_w  = np.empty(n_pair,   dtype=np.float64)
        tmp_y_dp_vec  = np.empty(n_pair,   dtype=np.float64)
        if n_classes > 2:
            class_counts_L  = np.empty(n_classes,   dtype=np.float64)
            class_counts_R  = np.empty(n_classes,   dtype=np.float64)
            P_k_L = np.zeros(n_classes, np.float64)
            P_k_R = np.zeros(n_classes, np.float64)

        else:
            dP_R1_w  = np.empty(n_pair,   dtype=np.float64)
            dP_L1_w  = np.empty(n_pair,   dtype=np.float64)
            y_dp_vec  = np.empty(N,   dtype=np.float64)

    else:
        temp_vec1 = np.empty(N, dtype=np.float64)   # (N,)  preallocated for calculations
        temp_vec2 = np.empty(N, dtype=np.float64)   # (N,)  preallocated for calculations
        temp_grad = np.empty(n_pair, dtype=np.float64)   # (d,)  preallocated for temporary gradient

    cdef np.ndarray[double, ndim=1] lower_bnd = np.zeros(n_pair, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] upper_bnd = np.zeros(n_pair, dtype=np.float64)
    cdef np.ndarray[int, ndim=1] nbd = np.zeros(n_pair, dtype=_lbfgsb.types.intvar.dtype)

    cdef tuple best_pair_py 
    cdef np.ndarray[double, ndim=1] best_x_py
    cdef int multi_range = 1 if n_classes <= 2 else n_classes
    cdef int current_class = 0

    for i, pair in enumerate(feature_pairs):

        x0 = rng.standard_normal(n_pair).astype(np.float64)

        for current_class in range(multi_range):
            x0, f_val = my_lbfgs_b_minimize(
                task,
                n_classes,
                <double>current_class,
                linear,
                x0,
                X[:, pair],
                y,        
                sample_weight,
                sum_sample_weight,
                N,
                n_pair,
                z,
                p,
                dp_dz,
                grad_w,
                dP_R1_w,
                dP_L1_w,
                tmp_y_dp_vec,
                dS_R_w,
                y_dp_vec,
                temp_vec1,
                temp_vec2,
                temp_grad,
                class_counts_L,
                class_counts_R,
                P_k_L,
                P_k_R,
                lower_bnd,
                upper_bnd,
                nbd,
                gamma,
                maxiter,
                relative_change)

            if f_val < best_fx:
                best_fx = f_val
                best_pair_py = pair
                best_x_py = x0 / np.abs(x0).max()

    for i in range(n_pair):
        best_pair[i] = best_pair_py[i]
        best_x[i] = best_x_py[i]

    cdef double[::1] best_values = np.dot(X[:, best_pair_py], best_x_py)

    with nogil:
        for i in range(n_samples):
            sort_buffer[i].value = best_values[i]
            sort_buffer[i].index = sample_indices[i]

        sort_pointer_array(sort_buffer, n_samples)

    return best_x, best_pair