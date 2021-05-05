import numpy as np
from scipy import optimize
from scipy import misc

from utils import (
    forward_differences_second,
    backward_differences,
    forward_differences   
)
from numpy import linalg as LA


# Original implementation
# https://github.com/danoan/image-processing/blob/master/denoise.py

class TotalGradient:
    def __init__(self, image):
        fd = forward_differences(image)
        self.gradX = fd[0]
        self.gradY = fd[1]

        fd2 = forward_differences_second(image)
        self.grad2X = fd2[0]
        self.grad2Y = fd2[1]

    def norm(self):
        return self.gradX**2 + self.gradY**2

class Tikhonov:
    def __init__(self, image, alpha):
        self.image = image
        self.alpha = alpha

        self.shape = self.image.shape
        self.size = self.image.size

    def jacobian(self, x):
        _x = x.reshape(self.shape)
        TG = TotalGradient(_x)

        S = self.alpha * (TG.grad2X + TG.grad2Y)
        return (_x - self.image - S).reshape(self.size,)

    def tikhonov(self, x):        
        """
        # arg min_x || A(x) - y ||^2 + r(x)
        # r(x) - log prior
        # y - noisy image
        # A(x) - reconstructed image        
        """
        _x = x.reshape(self.shape)
        TG = TotalGradient(_x)

        v = 0.5*(LA.norm(_x - self.image)**2 + self.alpha*np.sum(TG.norm()))
        return v

def tikhonov_denoise_image(input_image, alpha, max_it=100, print_output=False):
    T = Tikhonov(input_image, alpha)
    solution = optimize.minimize(
        lambda x: T.tikhonov(x),
        np.zeros(T.image.size,),
        jac=lambda x: T.jacobian(x),
        method="CG",
        options={
            "maxiter":max_it, 
            "disp":print_output
        }
    )

    x = solution["x"].reshape(T.shape)
    return x

def tikhonov_denoise_images(images, alpha, num_samples=8, idx=0, shape=(28, 28)):
    images_ = images.reshape(-1, *shape)[idx:idx+num_samples]
    return np.array([tikhonov_denoise_image(image, alpha, max_it=100) for image in images_]).reshape(-1, 28*28)
