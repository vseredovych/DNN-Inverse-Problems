import numpy as np
import tensorflow as tf

from numpy import linalg as LA

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation. Reversing matrix.
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(imagePadded.shape[1]):
        # Exit Convolution
        if y > imagePadded.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(imagePadded.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > imagePadded.shape[0] - xKernShape:
                    break

                # Only Convolve if x has moved by the specified Strides
                if x % strides == 0:
                    output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()


    return output

def convmtx(v, n):
    """Generates a convolution matrix
    
    Usage: X = convm(v,n)
    Given a vector v of length N, an N+n-1 by n convolution matrix is
    generated of the following form:
              |  v(0)  0      0     ...      0    |
              |  v(1) v(0)    0     ...      0    |
              |  v(2) v(1)   v(0)   ...      0    |
         X =  |   .    .      .              .    |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |  v(N) v(N-1) v(N-2) ...  v(N-n+1) |
              |   0   v(N)   v(N-1) ...  v(N-n+2) |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |   0    0      0     ...    v(N)   |
    And then it's trasposed to fit the MATLAB return value.     
    That is, v is assumed to be causal, and zero-valued after N.
    """
    N = len(v) + 2*n - 2
    xpad = np.concatenate([np.zeros(n-1), v[:], np.zeros(n-1)])
    X = np.zeros((len(v)+n-1, n))
    # Construct X column by column
    for i in range(n):
        X[:,i] = xpad[n-i-1:N-i]
    
    return X.transpose()
    
def forward_differences(image):
    '''
        u: Image matrix of dimensions (y,x)
        returns matrix fu of dimensions (2,y,x)
    '''
    output = np.zeros([2] + list(image.shape))
    # derivative by y
    output[0,:-1,:] = image[1:,:] - image[:-1,:]

    # derivative by x
    output[1,:,:-1] = image[:,1:] - image[:,:-1]

    return output

def backward_differences(image):
    '''
        u: Image matrix of dimensions (y,x)
        returns matrix output of dimensions (2,y,x)
    '''
    output = np.zeros([2] + list(image.shape))

    # derivative by y
    output[0,1:,:] = image[1:,:] - image[:-1,:]

    # derivative by x
    output[1,:,1:] = image[:,1:] - image[:,:-1]

    return output

def forward_differences_second(image):
    '''
        u: Image matrix of dimensions (y,x)
        returns matrix output of dimensions (2,y,x)
    '''
    output = np.zeros([2] + list(image.shape))

    # derivative by y
    output[0,1:-1,:] = image[0:-2,:] - 2*image[1:-1,:] + image[2:,:]

    # derivative by y
    output[1,:,1:-1] = image[:,0:-2] - 2*image[:,1:-1] + image[:,2:]
    return output

def avgn_attack(img, std=0.0, mean=0.1):
    noisy_img = img + np.random.normal(mean, std, img.shape)
    
    # we might get out of bounds due to noise
    noisy_img_clipped = np.clip(noisy_img, 0, 1)
    return noisy_img_clipped

def ssim(x, y, shape=(28, 28, 1)):
  #, filter_size=28, filter_sigma=1.5, k1=0.01, k2=0.03
  return float(
    tf.image.ssim(x.reshape(shape), y.reshape(shape), 1))

def mse(x, y):
    return np.mean((x - y) ** 2)
