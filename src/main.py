import matplotlib.pyplot as plt
import helpers as helpers
import scipy.optimize as opt
import numpy as np
from skimage import data, metrics

def lstsq(x, a, b):
    return (1/2) * np.linalg.norm(helpers.A(x, a) - b) ** 2

def grad(x, a, b):
    return helpers.AT(helpers.A(x, a), a) - helpers.AT(b, a)

def show_images(*pic):
    for i in range(len(pic)):
        plt.imshow(pic[i])
    plt.show()

def naive_deblur(blurred, n, m, k):
    # Function to minimize
    f = lambda x: lstsq(x.reshape(n, m), k, blurred)
    # Gradient of f
    grad_f = lambda x: grad(x.reshape(n, m), k, blurred).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    max_iter = 50
    
    minimum = opt.minimize(f, x0, jac=grad_f, options={'maxiter':max_iter}, method='CG')
    return minimum.x.reshape((n, m))

def blur_picture(picture, k):
    # Generate noise
    s = 0.004
    noise = np.random.normal(size=picture.shape) * s
    
    # Blur and add noise
    return helpers.A(picture, k) + noise


def f_reg(x, a, b, lamb=1.5):
  return (1/2) * (np.linalg.norm(helpers.A(x, a)-b)) ** 2 + (1/2) * lamb * np.linalg.norm(x) ** 2

def grad_reg(x, a, b, lamb=1.5):
  return helpers.AT(helpers.A(x, a), a) - helpers.AT(b, a) + lamb * x

def regular_deblur_cg(blurred, n, m, k, x=1.5):
    # Function to minimize
    lamb = 1.5
    f = lambda x: f_reg(x.reshape(n, m), k, blurred, lamb)
    # Gradient of f
    grad_f = lambda x: grad_reg(x.reshape(n, m), k, blurred, lamb).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    max_iter = 50
    
    minimum = opt.minimize(f, x0, jac=grad_f, options={'maxiter':max_iter}, method='CG')
    return minimum.x.reshape((n, m))

def get_alpha(x, f, grad, n, m):
  """
    x -> current x_k (immutable)
    f -> target function
    grad -> gradient calculated on x
  """
  alpha=1.1
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
  m1 = 1
  m2 = 0
  m3 = 0
  while(m1 > m2 + m3) and j < jmax :
    alpha = rho * alpha
    j += 1
    m1 = f(x.reshape(n, m) + alpha*p)
    m2 =  f(x.reshape(n , m))
    m3 = c1 * alpha * (grad.reshape(n * m).T @ p.reshape(n * m))

  if (j > jmax):
    return -1
  else:
    return alpha

def regular_deblur_gradient(blurred, n, m, ker, MAXITERATION=50, ABSOLUTE_STOP=1.e-5):
  """
    k -> kernel
    blurred -> blurred image
    n -> n
    m -> m
    MAXITERATION -> maximum number of iteration
    ABSOLUTE_STOP -> limit 
  """
  
  #initialize first values
  x_last = np.zeros(n * m)
  x_last[0] = 1
  alpha = 0
  k = 0
  
  f = lambda x: f_reg(x.reshape(n, m), ker, blurred)
  grad_f = lambda x: grad_reg(x.reshape(n, m), ker, blurred)

  while np.linalg.norm(grad_f(x_last)) > ABSOLUTE_STOP and k < MAXITERATION:
    k += 1
    grad = grad_f(x_last)
    alpha = get_alpha(x_last, f, grad, n, m)
    
    if(alpha == -1):
      print("No convergence")
      return None 
      
    x_last = x_last - alpha * grad.reshape(n * m)

  return x_last.reshape(n, m)


def elaborate_datasource(image_path, sigma, ker_len):
    # Open the image as a matrix and show it
    picture = plt.imread(image_path) 
    picture = picture[:, : , 0]
    n, m = picture.shape
    plt.imshow(picture, cmap='gray') 
    plt.show()

    # Create the kernel
    kernel = helpers.gaussian_kernel(ker_len, sigma)
    k = helpers.psf_fft(kernel, ker_len, (n, m))

    # Phase 1: blur
    blurred = blur_picture(picture, k)
    plt.imshow(blurred, cmap='gray') 
    plt.show()

    # Phase 2: naive solution
    deblurred = naive_deblur(blurred, n, m, k)
    plt.imshow(deblurred, cmap='gray') 
    plt.show()

    # Phase 3.1: regular solution with conjugate gradient
    deblurred_gc = regular_deblur_cg(blurred, n, m, k)
    plt.imshow(deblurred_gc, cmap='gray') 
    plt.show()

    # Phase 3.2: regular solution with gradient
    deblurred_gradient = regular_deblur_gradient(blurred, n, m, k)
    if deblurred_gradient is not None:
        plt.imshow(deblurred_gradient, cmap='gray') 
        plt.show()


def main():
    ds1 = './datasource/six.png'

    # b1_ds1 = elaborate_datasource(ds1, 0.5, 5)
    # b2_ds1 = elaborate_datasource(ds1, 1, 7)
    b3_ds1 = elaborate_datasource(ds1, 1.3, 9)


if __name__ == "__main__":
    main()
