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

def get_alpha(x, f, grad):
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
  while ((f(x + alpha*p) > f(x) +c1 * alpha * grad.T @ p) and j < jmax ):
    if (j > jmax):
        return -1
    else:
        return alpha

def regular_deblur_gradient(blurred, n, m, k, MAXITERATION=50, ABSOLUTE_STOP=1.e-5):
  """
    k -> kernel
    blurred -> blurred image
    n -> n
    m -> m
    MAXITERATION -> maximum number of iteration
    ABSOLUTE_STOP -> limit 
  """

  #declare x_k and gradient_k vectors
  x=np.zeros((1, MAXITERATION))
  norm_grad_list=np.zeros((1, MAXITERATION))
  function_eval_list=np.zeros((1, MAXITERATION))
  error_list=np.zeros((1, MAXITERATION))
  
  #initialize first values
  x_last = np.zeros(n * m)
  x_last[0] = 1
  #x[:,0] = x_last

  step = 0
  k=0
  
  f = lambda x: f_reg(x.reshape(n, m), k, blurred)
  grad_f = lambda x: grad_reg(x.reshape(n, m), k, blurred)

  function_eval_list[:, 0] = f(x_last)
  #error_list[:, 0] = np.linalg.norm(x_last - b)
  norm_grad_list[:, 0] = np.linalg.norm(grad_f(x_last))

  while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
    k += 1
    grad = grad_f(x_last)
    step = get_alpha(x_last, f, grad)
    
    if(step==-1):
      return None # no convergence

    # x_k+1 = x_k - alpha_k * Grad(x_k)
    x_last = x_last - step * grad;

    # x[k] = x_last

    # function_eval_list[:,k] = f_reg(x_last)
    # # error_list[:,k] = np.linalg.norm(x_last-b)
    # norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last))
    # function_eval_list = function_eval_list[:,:k+1]
    # # error_list = error_list[:,:k+1]
    # norm_grad_list = norm_grad_list[:,:k+1]
  
 
    # #plots
    # v_x0 = np.linspace(-5,5,500)
    # v_x1 = np.linspace(-5,5,500)
    # x0v,x1v = np.meshgrid(v_x0,v_x1)
    # z = f(x0v,x1v,b)

    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(v_x0, v_x1, z,cmap='viridis')
    # ax.set_title('Surface plot')
    # plt.show()

    # # plt.figure(figsize=(8, 5))
    # contours = plt.contour(x0v, x1v, z, levels=30)
    # plt.plot(x[0,0:k],x[1,0:k],'*')
    # #plt.axis([-5,5,-5,5])
    # plt.axis ('equal')
    # plt.show()
  print(x_last, f(x_last))
  return f(x_last)


def elaborate_datasource(image_path, sigma, ker_len):
    # Open the image as a matrix and show it
    picture = plt.imread(image_path) 
    picture = picture[:, : , 0]
    n, m = picture.shape
    # plt.imshow(picture, cmap='gray') 
    # plt.show()

    # Create the kernel
    kernel = helpers.gaussian_kernel(ker_len, sigma)
    k = helpers.psf_fft(kernel, ker_len, (n, m))

    # Phase 1: blur
    blurred = blur_picture(picture, k)
    plt.imshow(blurred, cmap='gray') 
    # plt.show()

    show_images(picture, blurred)

    # Phase 2: naive solution
    deblurred = naive_deblur(blurred, n, m, k)
    plt.imshow(deblurred, cmap='gray') 
    # plt.show()

    # Phase 3: regular solution
    # deblurred_gc = regular_deblur_cg(blurred, n, m, k)
    # plt.imshow(deblurred_gc, cmap='gray') 
    # plt.show()

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
