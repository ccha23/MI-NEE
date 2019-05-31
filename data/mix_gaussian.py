import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import xlogy


class MixedGaussian():
    # Mixture of two bivariate gaussians
    #
    # data(mix,Mode,Rho,N) generates N samples with
    # mix: mixing ration between 0 and 1
    # Rho[0] correlation for the first bivariate gaussian and Rho[1] for the second
    # Mode[0] separation between the two bivariate gaussians along the x-axis and Mode[1] is the separation along the y-axis

    def __init__(self, sample_size=400, mean1=0, mean2=0, rho1=0.9, rho2=-0.9, mix=0.5):
        # sample_size is the number of sample representing the distribution
        # mix: mixing ratio of two bivariate gaussian in between 0 and 1
        # Rho1: correlation for the first bivariate gaussian
        # Rho2: correlation for the second bivariate gaussian
        # mean1 is the mean of first variable and mean2 is the mean of second variable
        self.sample_size = sample_size
        self.covMat1 = np.array([[1, rho1], [rho1, 1]])
        self.covMat2 = np.array([[1, rho2], [rho2, 1]])
        self.sample_size = sample_size
        self.mix = mix
        self.mu = np.array([mean1, mean2])
        self.name = 'bimodal'

    @property
    def data(self):
        """[summary]

        Returns:
            [np.array] -- [N by 2 matrix]
        """
        N1 = int(self.mix*self.sample_size)
        N2 = self.sample_size-N1
        temp1 = np.random.multivariate_normal(mean=self.mu,
                                              cov=self.covMat1,
                                              size=N1)
        temp2 = np.random.multivariate_normal(mean=-self.mu,
                                              cov=self.covMat2,
                                              size=N2)
        X = np.append(temp1, temp2, axis=0)
        np.random.shuffle(X)
        return X

    @property
    def ground_truth(self):
        # fx and fy are  x and y marginal probability density functions(pdf) of mix-gaussian distribution
        # fxy is the joint probability density function of mix-gaussian distribution
        # the mutual information ground truth is the difference between sum of entropy of individual variables and joint entropy of all variables
        # the entropies are computed by integrating the expectation of pdf of variables involved
        mix, covMat1, covMat2, mu = self.mix, self.covMat1, self.covMat2, self.mu

        def fxy(x, y):
            X = np.array([x, y])
            temp1 = np.matmul(
                np.matmul(X-mu, np.linalg.inv(covMat1)), (X-mu).transpose())
            temp2 = np.matmul(
                np.matmul(X+mu, np.linalg.inv(covMat2)), (X+mu).transpose())
            return mix*np.exp(-.5*temp1) / (2*np.pi * np.sqrt(np.linalg.det(covMat1))) \
                + (1-mix)*np.exp(-.5*temp2) / \
                (2*np.pi * np.sqrt(np.linalg.det(covMat2)))

        def fx(x):
            return mix*np.exp(-(x-mu[0])**2/(2*covMat1[0, 0])) / np.sqrt(2*np.pi*covMat1[0, 0]) \
                + (1-mix)*np.exp(-(x+mu[0])**2/(2*covMat2[0, 0])
                                 ) / np.sqrt(2*np.pi*covMat2[0, 0])

        def fy(y):
            return mix*np.exp(-(y-mu[1])**2/(2*covMat1[1, 1])) / np.sqrt(2*np.pi*covMat1[1, 1]) \
                + (1-mix)*np.exp(-(y+mu[1])**2/(2*covMat2[1, 1])
                                 ) / np.sqrt(2*np.pi*covMat2[1, 1])

        lim = np.inf

        hx = quad(lambda x: -xlogy(fx(x), fx(x)), -lim, lim)
        
        hy = quad(lambda y: -xlogy(fy(y), fy(y)), -lim, lim)

        hxy = dblquad(lambda x, y: -xlogy(fxy(x, y), fxy(x, y)), -
                      lim, lim, lambda x: -lim, lambda x: lim)

        return hx[0] + hy[0] - hxy[0]