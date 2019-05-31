import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import xlogy


class Gaussian():
    # Gaussian
    #
    # Generate two samples represent corelated gaussian distribution
    def __init__(self, sample_size=400, rho=0.9, mean=[0, 0]):
        # sample_size is the number of sample representing the distribution
        # Rho: correlation for the gaussian. This is used to generate an covariance matrix diagonal equal to one and anti-diagonal equal to rho. That means, the covariance between first dimension of first variable and the last dimension of second variable is rho, the covariance between second dimension of first variable and the second last dimension of second variable is rho...
        # mean: an array representing the mean of two variables, first half of array representing the mean of first variable, and second half of the array representing the mean of second variable. We assume the dimension of two variables to be equal, thus we assume even size of mean array.
        self.sample_size = sample_size
        self.mean = mean
        self.rho = rho

    @property
    def data(self):
        """[summary]
        Returns:
            [np array] -- [N by 2 matrix]
        """
        if len(self.mean)%2 == 1:
            raise ValueError("length of mean array is assummed to be even")
        cov = (np.identity(len(self.mean))+self.rho*np.identity(len(self.mean))[::-1]).tolist()
        return np.random.multivariate_normal(
            mean=self.mean,
            cov=cov,
            size=self.sample_size)

    @property
    def ground_truth(self):
        # since the covariance matrices of each variable are identity matrices, and the two variables are co-varied with same rho dimension-by-dimension. Therefore we can simplify the ground truth to be the product of mutual information of one-dimension variables and number of dimension of each variable
        if len(self.mean)%2 == 1:
            raise ValueError("length of mean array is assummed to be even")
        dim = len(self.mean)//2
        return -0.5*np.log(1-self.rho**2)*dim