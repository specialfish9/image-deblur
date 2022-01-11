import string

import numpy as np
from numpy.random.mtrand import negative_binomial


def lstsq(x, a, b):
    """
    least-squares function implementation
    """
    from helpers import A

    return (1 / 2) * np.linalg.norm(A(x, a) - b) ** 2


def lstsq_grad(x, a, b):
    """
    least-squares gradient implementation
    """
    from helpers import A, AT

    return AT(A(x, a), a) - AT(b, a)


def lstsq_tikhonov(x, a, b, lamb):
    """
    lstsq with Tikhonov regularization
    """
    return lstsq(x, a, b) + (1 / 2) * lamb * np.linalg.norm(x) ** 2


def lstsq_tikhonov_grad(x, a, b, lamb):
    """
    least-squares gradient with Tikhonov regularization
    """
    return lstsq_grad(x, a, b) + lamb * x


def total_variation(u, eps=1e-2):
    """
    Total Variation formula implementation
    """
    dx, dy = np.gradient(u)
    n2 = np.square(dx) + np.square(dy)

    return np.sqrt(n2 + eps ** 2).sum()


def tot_var_grad(u, eps=1e-2):
    """
    Total variation gradient
    """
    dx, dy = np.gradient(u)

    n2 = np.square(dx) + np.square(dy)
    den = np.sqrt(n2 + eps ** 2)

    fx = dx / den
    fy = dy / den

    d_fdx = np.gradient(fx, axis=0)
    d_fdy = np.gradient(fy, axis=1)

    div = (d_fdx + d_fdy)

    return -div


def lstsq_tot_var(x, a, b, lamb, eps=1e-2):
    """
    lstsq with total variation regularization
    """
    return lstsq(x, a, b) + lamb * total_variation(x, eps)


def lstsq_tot_var_grad(x, a, b, lamb, eps=1e-2):
    """
    lstsq gradient with total variation regularization
    """
    return lstsq_grad(x, a, b) + lamb * tot_var_grad(x, eps)


def blur_picture(picture, k, s=0.004):
    """
    Blur a picture using gaussian-blur and gaussian noise
    @param picture: Original picture
    @param k: Kernel matrix
    @param s: Standard deviation of gaussian noise
    @return: Blurred picture
    """
    from helpers import A

    # Generate noise
    noise = np.random.normal(size=picture.shape) * s

    # Blur and add noise
    return A(picture, k) + noise


def naive_deblur(blurred, n, m, k, max_iter=50):
    """
    Deblur using naive method: find the absolute minimum of lstsq(x, k, blurred) using conjugate gradient
    @param blurred: Blurred image
    @param n: width
    @param m: height
    @param k: Gaussian blur matrix
    @param max_iter: Maximum number of iterations
    @return: de-blurred image
    """
    from scipy.optimize import minimize

    # Function to minimize
    f = lambda x: lstsq(x.reshape(n, m), k, blurred)
    # Gradient of f
    grad_f = lambda x: lstsq_grad(x.reshape(n, m), k, blurred).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    minimum = minimize(f, x0, jac=grad_f, options={'maxiter': max_iter}, method='CG')
    return minimum.x.reshape((n, m))


def regularized_deblur_cg(blurred, n, m, k, lamb, max_iter=50):
    """
    Deblur image finding the absolute minimum of lstsq  with Tikhonov regularization
    @param blurred: Image to de-blur
    @param n: Width
    @param m: Height
    @param k: Gaussian-kernel
    @param lamb: regularization parameter
    @param max_iter: Maximum number of iterations
    @return: De-blurred image
    """
    from scipy.optimize import minimize

    # Function to minimize
    f = lambda x: lstsq_tikhonov(x.reshape(n, m), k, blurred, lamb)
    # Gradient of f
    grad_f = lambda x: lstsq_tikhonov_grad(x.reshape(n, m), k, blurred, lamb).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    minimum = minimize(f, x0, jac=grad_f, options={'maxiter': max_iter}, method='CG')
    return minimum.x.reshape((n, m))


def next_step(x, f, grad_x, n, m):
    """
      Find next step for gradient descent method
      @param x: Current x_k (immutable)
      @param f: Target function
      @param grad_x: Gradient function evaluated on x_k
      @param n: width
      @param m: height
      @return: Next step
    """
    alpha = 1.1
    rho = 0.5
    c1 = 0.25
    p = -grad_x
    j = 0
    jmax = 10
    while (f(x.reshape(n, m) + alpha * p) > f(x.reshape(n, m)) + c1 * alpha * (
            grad_x.reshape(n * m).T @ p.reshape(n * m))) and j < jmax:
        alpha = rho * alpha
        j += 1

    if j > jmax:
        return -1
    else:
        return alpha


def gradient_descend(f, n, m, grad_f, x0, max_iter=50, absolute_stop=1.e-5):
    """
    Gradient descend method implementation.
    @param f: Function to minimize
    @param n: Width
    @param m: Height
    @param grad_f: Gradient of [f]
    @param x0: Initial guess
    @param max_iter: Maximum number of iterations
    @param absolute_stop: Absolute stop value
    @return: Min value calculated for [f]
    """
    k = 0

    while np.linalg.norm(grad_f(x0)) > absolute_stop and k < max_iter:
        k += 1
        gradient = grad_f(x0)
        alpha = next_step(x0, f, gradient, n, m)

        if alpha == -1:
            print("No convergence")
            return None

        x0 = x0 - alpha * gradient.reshape(n * m)

    return x0.reshape(n, m)


def regularized_deblur_gradient(blurred, n, m, ker, lamb, max_iter=50, absolute_stop=1.e-5):
    """
      Deblur an image using Gradient Descend method
      @param blurred: Blurred image
      @param n: Width
      @param m: Height
      @param ker: Gaussian kernel matrix
      @param max_iter: Maximum number of iterations
      @param absolute_stop: Absolute stop value
      @param lamb: Regularization parameter
      @return: De-blurred image
    """

    # initialize first values
    x0 = np.zeros(n * m)
    x0[0] = 1

    f = lambda x: lstsq_tikhonov(x.reshape(n, m), ker, blurred, lamb)
    grad_f = lambda x: lstsq_tikhonov_grad(x.reshape(n, m), ker, blurred, lamb)
    return gradient_descend(f, n, m, grad_f, x0, max_iter, absolute_stop)


def regularized_deblur_tot_var(blurred, n, m, ker, lamb, max_iter=50, absolute_stop=1.e-5):
    """
      Deblur an image using Gradient Descend method with total variation
      @param blurred: Blurred image
      @param n: Width
      @param m: Height
      @param ker: Gaussian kernel matrix
      @param max_iter: Maximum number of iterations
      @param absolute_stop: Absolute stop value
      @param lamb: regularization parameter
      @return: De-blurred image
    """
    x0 = np.zeros(n * m)
    x0[0] = 1

    f = lambda x: lstsq_tot_var(x.reshape(n, m), ker, blurred, lamb)
    grad_f = lambda x: lstsq_tot_var_grad(x.reshape(n, m), ker, blurred, lamb)

    return gradient_descend(f, n, m, grad_f, x0, max_iter, absolute_stop)


def print_res(original, deblurred):
    from skimage import metrics

    psnr = metrics.peak_signal_noise_ratio(original, deblurred)
    mse = metrics.mean_squared_error(original, deblurred)

    return psnr, mse


def elaborate_datasource(image_path, sigma, ker_len, noise_std_dev=5e-3, lamb=1e-5):
    import matplotlib.pyplot as plt
    from helpers import gaussian_kernel, psf_fft

    picture = plt.imread(image_path)
    picture = picture[:, :, 0]
    n, m = picture.shape


    """
    KERNEL DIM TEST
    """
    
    print("KERNEL DIM TEST\n")
    file = open("test_output/ker_dim_test.csv", "w")
    file.write("ker_dim,psnr,mse\n")

    values = [
        3,
        5,
        7,
        9,
        13,
        49,
        99,
        299,
        599
    ]
    
    for i in values:
        if n < i:
            print("ERROR: image to small")
            continue
            
        print(">iter:", i)
        kernel = gaussian_kernel(i, sigma)
        k = psf_fft(kernel, i, (n, m))

        print("Phase 1")
        blurred = blur_picture(picture, k, noise_std_dev)


        print("Phase 2")
        deblurred_naive = naive_deblur(blurred, n, m, k)
        p, mmm = print_res(picture, deblurred_naive)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 3.1")
        deblurred_gc = regularized_deblur_cg(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_gc)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 3.2")
        deblurred_gradient = regularized_deblur_gradient(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_gradient)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 4")
        deblurred_tot_var = regularized_deblur_tot_var(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_tot_var)
        file.write("{},{},{}\n".format(i, p, mmm))

    file.close()

    """
    SIGMA DIM TEST
    """
    
    print("SIGMA DIM TEST\n")
    file = open("test_output/sigma_test.csv", "w")
    file.write("sigma,psnr,mse\n")

    values = [
        0.1,
        0.5,
        1,
        1.3,
        1.7,
        2.2,
        2.6,
        3.5,
        4.7
    ]
    
    for i in values:
        if n < i:
            print("ERROR: image to small")
            continue
            
        print(">iter:", i)
        kernel = gaussian_kernel(ker_len, i)
        k = psf_fft(kernel, ker_len, (n, m))

        print("Phase 1")
        blurred = blur_picture(picture, k, noise_std_dev)


        print("Phase 2")
        deblurred_naive = naive_deblur(blurred, n, m, k)
        p, mmm = print_res(picture, deblurred_naive)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 3.1")
        deblurred_gc = regularized_deblur_cg(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_gc)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 3.2")
        deblurred_gradient = regularized_deblur_gradient(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_gradient)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 4")
        deblurred_tot_var = regularized_deblur_tot_var(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_tot_var)
        file.write("{},{},{}\n".format(i, p, mmm))

    file.close()



    """
    NOISE STD DEV TEST
    """
    print("NOISE STD DEV TEST\n")
    file = open("test_output/noise_std_dev_test.csv", "w")
    file.write("Std_dev,psnr,mse\n")

    values = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09
    ]

    for i in values:
        kernel = gaussian_kernel(ker_len, sigma)
        k = psf_fft(kernel, ker_len, (n, m))
        print(">iter:", i)

        print("Phase 1")
        blurred = blur_picture(picture, k, i)

        print("Phase 2")
        deblurred_naive = naive_deblur(blurred, n, m, k)
        p, mmm = print_res(picture, deblurred_naive)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 3.1")
        deblurred_gc = regularized_deblur_cg(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_gc)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 3.2")
        deblurred_gradient = regularized_deblur_gradient(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_gradient)
        file.write("{},{},{}\n".format(i, p, mmm))

        print("Phase 4")
        deblurred_tot_var = regularized_deblur_tot_var(blurred, n, m, k, lamb)
        p, mmm = print_res(picture, deblurred_tot_var)
        file.write("{},{},{}\n".format(i, p, mmm))

    file.close()


    """
    LAMBDA TEST
    """

    print("LAMBDA TEST\n")

    file = open("test_output/lambda_test.csv", "w")
    file.write("lambda,psnr,mse\n")

    values = [0.01,
              0.05,
              0.10,
              0.15,
              0.20,
              0.30,
              0.50]

    kernel = gaussian_kernel(ker_len, sigma)
    k = psf_fft(kernel, ker_len, (n, m))

    print("Phase 1")
    blurred = blur_picture(picture, k, noise_std_dev)

    print("Phase 3.1")
    for i in values:
        print(">iter:", i)
        deblurred_gc = regularized_deblur_cg(blurred, n, m, k, i)
        p, mmm = print_res(picture, deblurred_gc)
        file.write("{},{},{}\n".format(i, p, mmm))

    print("Phase 3.2")
    for i in values:
        print(">iter:", i)
        deblurred_gradient = regularized_deblur_gradient(blurred, n, m, k, i)
        p, mmm = print_res(picture, deblurred_gradient)
        file.write("{},{},{}\n".format(i, p, mmm))

    print("Phase 4")
    for i in values:
        print(">iter:", i)
        deblurred_tot_var = regularized_deblur_tot_var(blurred, n, m, k, i)
        p, mmm = print_res(picture, deblurred_tot_var)
        file.write("{},{},{}\n".format(i, p, mmm))

    file.close()
    

def elaborate_datasource_stats(image_path, sigma, ker_len, noise_std_dev=5e-3, lamb=1e-5):
    import matplotlib.pyplot as plt
    from helpers import gaussian_kernel, psf_fft

    picture = plt.imread(image_path)
    picture = picture[:, :, 0]
    n, m = picture.shape

    psnr = [None] * 4
    mse = [None] * 4
    
    kernel = gaussian_kernel(ker_len, sigma)
    k = psf_fft(kernel, ker_len, (n, m))

    print("Phase 1")
    blurred = blur_picture(picture, k, noise_std_dev)


    print("Phase 2")
    deblurred_naive = naive_deblur(blurred, n, m, k)
    psnr[0], mse[0] = print_res(picture, deblurred_naive)


    print("Phase 3.1")
    deblurred_gc = regularized_deblur_cg(blurred, n, m, k, lamb)
    psnr[1], mse[1] = print_res(picture, deblurred_gc)

    print("Phase 3.2")
    deblurred_gradient = regularized_deblur_gradient(blurred, n, m, k, lamb)
    psnr[2], mse[2] = print_res(picture, deblurred_gradient)

    print("Phase 4")
    deblurred_tot_var = regularized_deblur_tot_var(blurred, n, m, k, lamb)
    psnr[3], mse[3] = print_res(picture, deblurred_tot_var)

    return psnr, mse


def _er(o, c):
    import scipy as scipy
    return scipy.linalg.norm(o-c, 2) / scipy.linalg.norm(o,2) 

def elaborate_datasource_gc_vs_gd(image_path, sigma, ker_len, noise_std_dev=5e-3, lamb=1e-5):
    import matplotlib.pyplot as plt
    from helpers import gaussian_kernel, psf_fft

    file = open("test_output/gc_vs_gd.csv","w")

    picture = plt.imread(image_path)
    picture = picture[:, :, 0]
    n, m = picture.shape
    
    kernel = gaussian_kernel(ker_len, sigma)
    k = psf_fft(kernel, ker_len, (n, m))

    print("Blurring...")
    blurred = blur_picture(picture, k, noise_std_dev)

    i = 5
    file.write("max_iter:psnrCg:psnrCd:mseCg:mseCd:erCg:erCd\n")
    while i <= 50:
        print("New iteration: ", i)
        print("cg...")
        deblurred_gc = regularized_deblur_cg(blurred, n, m, k, lamb, max_iter=i)
        pcg, mcg = print_res(picture, deblurred_gc)


        print("gd...")
        deblurred_gradient = regularized_deblur_gradient(blurred, n, m, k, lamb, max_iter=i)
        pcd, mcd = print_res(picture, deblurred_gradient)
        
        ercg = _er(picture, deblurred_gc)
        ercd = _er(picture, deblurred_gradient)
        file.write("{}:{}:{}:{}:{}:{}:{}\n".format( i, pcg, pcd, mcg, mcd, ercg, ercd))
        i += 5

    file.close()


def stats():
    file = open("test_output/stats.csv", "w")

    naive = []
    tkgc = []
    tkgd = []
    tv = []

    p1, m1 = elaborate_datasource_stats('./datasource/one.png', 0.5, 9, 0.01, 0.01)
    p2, m2 = elaborate_datasource_stats('./datasource/two.png', 0.5, 9,0.01, 0.01)
    p3, m3 = elaborate_datasource_stats('./datasource/three.png', 0.5, 9,0.01, 0.01)
    p4, m4 = elaborate_datasource_stats('./datasource/four.png', 0.5, 9,0.01, 0.01)
    p5, m5 = elaborate_datasource_stats('./datasource/five.png', 0.5, 9,0.01, 0.01)
    p6, m6 = elaborate_datasource_stats('./datasource/six.png', 0.5, 9,0.01, 0.01)
    p7, m7 = elaborate_datasource_stats('./datasource/seven.png', 0.5, 9,0.01, 0.01)
    p8, m8 = elaborate_datasource_stats('./datasource/eight.png', 0.5, 9,0.01, 0.01)

    naive.append(p1[0])
    naive.append(p2[0])
    naive.append(p3[0])
    naive.append(p4[0])
    naive.append(p5[0])
    naive.append(p6[0])
    naive.append(p7[0])
    naive.append(p8[0])

    tkgc.append(p1[1])
    tkgc.append(p2[1])
    tkgc.append(p3[1])
    tkgc.append(p4[1])
    tkgc.append(p5[1])
    tkgc.append(p6[1])
    tkgc.append(p7[1])
    tkgc.append(p8[1])

    tkgd.append(p1[2])
    tkgd.append(p2[2])
    tkgd.append(p3[2])
    tkgd.append(p4[2])
    tkgd.append(p5[2])
    tkgd.append(p6[2])
    tkgd.append(p7[2])
    tkgd.append(p8[2])

    tv.append(p1[3])
    tv.append(p2[3])
    tv.append(p3[3])
    tv.append(p4[3])
    tv.append(p5[3])
    tv.append(p6[3])
    tv.append(p7[3])
    tv.append(p8[3])


    file.write("PSNR\n")
    file.write("method:mean/std\n")

    file.write("{}:{}\n".format("naive",  np.mean(naive)))
    file.write("{}:{}\n".format("naive",  np.std(naive)))
    file.write("{}:{}\n".format("tkgc",  np.mean(tkgc)))
    file.write("{}:{}\n".format("tkgc",  np.std(tkgc)))
    file.write("{}:{}\n".format("tkgd",  np.mean(tkgd)))
    file.write("{}:{}\n".format("tkgd",  np.std(tkgd)))
    file.write("{}:{}\n".format("tv",  np.mean(tv)))
    file.write("{}:{}\n".format("tv",  np.std(tv)))

    naive = []
    tkgc = []
    tkgd = []
    tv = []

    naive.append(m1[0])
    naive.append(m2[0])
    naive.append(m3[0])
    naive.append(m4[0])
    naive.append(m5[0])
    naive.append(m6[0])
    naive.append(m7[0])
    naive.append(m8[0])

    tkgc.append(m1[1])
    tkgc.append(m2[1])
    tkgc.append(m3[1])
    tkgc.append(m4[1])
    tkgc.append(m5[1])
    tkgc.append(m6[1])
    tkgc.append(m7[1])
    tkgc.append(m8[1])

    tkgd.append(m1[2])
    tkgd.append(m2[2])
    tkgd.append(m3[2])
    tkgd.append(m4[2])
    tkgd.append(m5[2])
    tkgd.append(m6[2])
    tkgd.append(m7[2])
    tkgd.append(m8[2])

    tv.append(m1[3])
    tv.append(m2[3])
    tv.append(m3[3])
    tv.append(m4[3])
    tv.append(m5[3])
    tv.append(m6[3])
    tv.append(m7[3])
    tv.append(m8[3])

    file.write("MSE\n")
    file.write("{}:{}\n".format("naive",  np.mean(naive)))
    file.write("{}:{}\n".format("naive",  np.std(naive)))
    file.write("{}:{}\n".format("tkgc",  np.mean(tkgc)))
    file.write("{}:{}\n".format("tkgc",  np.std(tkgc)))
    file.write("{}:{}\n".format("tkgd",  np.mean(tkgd)))
    file.write("{}:{}\n".format("tkgd",  np.std(tkgd)))
    file.write("{}:{}\n".format("tv",  np.mean(tv)))
    file.write("{}:{}\n".format("tv",  np.std(tv)))

    file.close()


def main():
    elaborate_datasource_gc_vs_gd("datasource/extra1.png", 1, 7)

if __name__ == "__main__":
    main()
