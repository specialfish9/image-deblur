import matplotlib.pyplot as plt
import helpers as helpers
import numpy as np
import scipy.linalg.decomp_lu as LUdec
import scipy.linalg
from scipy.optimize import minimize 

def blur_datasource(image_path, sigma, ker_len):
    # Open the image as a matrix and show it
    pic = plt.imread(image_path) 
    pic = pic[:, : , 0]

    # Get the kernel
    ker = helpers.gaussian_kernel(ker_len, sigma)

    # Appply Fourier
    K = helpers.psf_fft(ker, ker_len, (512,512))

    # Blur and show
    blurred = helpers.A(pic, K)
    plt.imshow(blurred, cmap='gray') 
    plt.show()
    return blurred


def f(x, b):
  return 0.5 * (np.linalg.norm(A(x) - b)**2)

def grad_f(x, b):
  return helpers.AT(A(x) - b)


def next_step(x, b, f, grad):         #backtracking per calcolo passo
  alpha = 1.1
  rho = 0.5
  c1 = 0.25
  while f(x - alpha * grad, b) > f(x, b) - alpha * c1 * np.linalg.norm(np.dot(grad.T, grad)):
    alpha = alpha * rho
  return alpha


def minimize(x0, b, maxit, abstop):      #funzione che trova il minimo
  x = x0
  grad = grad_f(x, b)
  k = 0
  while(np.linalg.norm(grad) > abstop) and (k < maxit):
    x = x - next_step(x, b, f, grad) * grad
    grad = grad_f(x, b)
    k = k + 1

  return(x, grad, k)

def conjugate_gradient(b1_ds1):

    n = 5 # Grado del polinomio approssimante
    x = b1_ds1
    y = b1_ds1
    N = x.size # Numero dei dati
    A = np.zeros((N, n+1))
    print(x.shape)
    for i in range(N):
      A[:, i] = x ** i 

    print("A = \n", A)
    # Per chiarezza, calcoliamo la matrice del sistema e il termine noto a parte
    ATA = np.dot(A.T, A) 
    ATy = np.dot(A.T,y)
    
    lu, piv = LUdec.lu_factor(ATA)
    
    print('LU = \n', lu)
    print('piv = ', piv)
    
    alpha_normali = LUdec.lu_solve((lu, piv), ATy) 
    return alpha_normali

def main():
    ds1 = './datasource/one.png'

    # TODO: add gaussian rumor between ]0; 0,05]
    b1_ds1 = blur_datasource(ds1, 0.5, 5)
    b2_ds1 = blur_datasource(ds1, 1, 7)
    b3_ds1 = blur_datasource(ds1, 1.3, 9)
    # Add here calls to phase2
    res = minimize(conjugate_gradient(b1_ds1), b2_ds1)
    plt.imshow(res, cmap='red') 
    plt.show()


if __name__ == "__main__":
    main()






# import matplotlib.pyplot as plt
# import helpers as helpers
# import numpy as np
# import scipy.linalg.decomp_lu as LUdec
# import scipy.linalg
# from scipy.optimize import minimize 

# def blur_datasource(image_path, sigma, ker_len):
#     # Open the image as a matrix and show it
#     pic = plt.imread(image_path) 
#     pic = pic[:, : , 0]

#     # Get the kernel
#     ker = helpers.gaussian_kernel(ker_len, sigma)

#     # Appply Fourier
#     K = helpers.psf_fft(ker, ker_len, (512,512))

#     # Blur and show
#     blurred = helpers.A(pic, K)
#     plt.imshow(blurred, cmap='gray') 
#     plt.show()
#     return blurred

# def conjugate_gradient(b1_ds1):

#     n = 5 # Grado del polinomio approssimante
#     x = b1_ds1
#     y = b1_ds1
#     N = x.size # Numero dei dati
#     A = np.zeros((N, n+1))
#     print(x.shape)
#     for i in range(N):
#       A[:, i] = x ** i 

#     print("A = \n", A)
#     # Per chiarezza, calcoliamo la matrice del sistema e il termine noto a parte
#     ATA = np.dot(A.T, A) 
#     ATy = np.dot(A.T,y)
    
#     lu, piv = LUdec.lu_factor(ATA)
    
#     print('LU = \n', lu)
#     print('piv = ', piv)
    
#     alpha_normali = LUdec.lu_solve((lu, piv), ATy) 
#     return alpha_normali

# def main():
#     ds1 = './datasource/one.png'

#     # TODO: add gaussian rumor between ]0; 0,05]
#     b1_ds1 = blur_datasource(ds1, 0.5, 5)
#     b2_ds1 = blur_datasource(ds1, 1, 7)
#     b3_ds1 = blur_datasource(ds1, 1.3, 9)
#     # Add here calls to phase2
#     res = minimize(conjugate_gradient(b1_ds1), b2_ds1)
#     plt.imshow(res, cmap='red') 
#     plt.show()


# if __name__ == "__main__":
#     main()
