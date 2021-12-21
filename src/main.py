import matplotlib.pyplot as plt
import helpers as helpers

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


def main():
    ds1 = './datasource/one.png'
    blur_datasource(ds1, 0.5, 5)
    blur_datasource(ds1, 1, 7)
    blur_datasource(ds1, 1.3, 9)

if __name__ == "__main__":
    main()
