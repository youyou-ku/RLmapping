import numpy as np

# 対数尤度計算（分散は推定値）
def log_likelihoods(errors):
    errors = np.array(errors)
    var = np.mean(errors ** 2, axis=0) + 1e-16
    dim = errors.shape[1]
    return np.mean(
        -0.5 * (
            dim * np.log(2 * np.pi) 
            + np.sum(np.log(var)) 
            + np.sum((errors ** 2) / var, axis=1)
        )
    )


# 二重中心化
def double_centering(L):
    L = np.array(L)
    # Step 1: 行方向中心化
    row_mean = np.mean(L, axis=1, keepdims=True)
    xi = L - row_mean

    # Step 2: 列方向中心化
    col_mean = np.mean(xi, axis=0, keepdims=True)
    q = xi - col_mean

    return q
