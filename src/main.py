import matplotlib.pyplot as plt

def blur_datasource():
    A = pic = plt.imread('./datasource/one.png') 
    plt.imshow(A, cmap='gray') 
    plt.show()


def main():
    blur_datasource()

if __name__ == "__main__":
    main()
