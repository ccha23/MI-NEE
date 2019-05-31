import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def _resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch


def _div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref


class MINE():
    r"""Class for Mutual Information Neural Estimation. 

    The mutual information is estimated using neural estimation of the divergence 
    from joint distribution to product of marginal distributions.

    Arguments:
    X (tensor): samples of X
        dim 0: different samples
        dim 1: different components
    Y (tensor): samples of Y
        dim 0: different samples
        dim 1: different components
    ma_rate (float, optional): rate of moving average in the gradient estimate
    ma_ef (float, optional): initial value used in the moving average
    lr (float, optional): learning rate
    hidden_size (int, optional): size of the hidden layers
    """
    class Net(nn.Module):
        def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.fc1.weight, std=sigma)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight, std=sigma)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight, std=sigma)
            nn.init.constant_(self.fc3.bias, 0)

        def forward(self, input):
            output = F.elu(self.fc1(input))
            output = F.elu(self.fc2(output))
            output = self.fc3(output)
            return output

    def __init__(self, X, Y, batch_size=32, lr=1e-3, ma_rate=0.1, hidden_size=100, ma_ef=1):
        self.lr = lr
        self.batch_size = batch_size
        self.ma_rate = ma_rate

        self.X = X
        self.Y = Y
        self.XY = torch.cat((self.X, self.Y), dim=1)

        self.X_ref = _resample(self.X, batch_size=self.X.shape[0])
        self.Y_ref = _resample(self.Y, batch_size=self.Y.shape[0])

        self.XY_net = MINE.Net(
            input_size=X.shape[1]+Y.shape[1], hidden_size=300)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(), lr=lr)

        self.ma_ef = ma_ef  # for moving average

    def step(self, iter=1):
        r"""Train the neural networks for one or more steps.

        Argument:
        iter (int, optional): number of steps to train.
        """
        for i in range(iter):
            self.XY_optimizer.zero_grad()
            batch_XY = _resample(self.XY, batch_size=self.batch_size)
            batch_XY_ref = torch.cat((_resample(self.X, batch_size=self.batch_size),
                                      _resample(self.Y, batch_size=self.batch_size)), dim=1)
            # define the loss function with moving average in the gradient estimate
            mean_fXY = self.XY_net(batch_XY).mean()
            mean_efXY_ref = torch.exp(self.XY_net(batch_XY_ref)).mean()
            self.ma_ef = (1-self.ma_rate)*self.ma_ef + \
                self.ma_rate*mean_efXY_ref
            batch_loss_XY = - mean_fXY + \
                (1 / self.ma_ef.mean()).detach() * mean_efXY_ref
            batch_loss_XY.backward()
            self.XY_optimizer.step()

    def forward(self, X=None, Y=None):
        r"""Evaluate the neural network on (X,Y). 

        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used. 
        The arguments are useful for testing/validation with a separate data set.
        """
        XY = None
        if X is None or Y is None:
            XY, X, Y = self.XY, self.X, self.Y
        else:
            XY = torch.cat((X, Y), dim=1)
        X_ref = _resample(X, batch_size=X.shape[0])
        Y_ref = _resample(Y, batch_size=Y.shape[0])
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)
        return _div(self.XY_net, XY, XY_ref).cpu().item()

    def estimate(self, X=None, Y=None):
        r"""Return the mutual information estimate.

        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used. 
        The arguments are useful for testing/validation with a separate data set.
        """
        return self.forward(X, Y)

    def state_dict(self):
        r"""Return a dictionary storing the state of the estimator.
        """
        return {
            'XY_net': self.XY_net.state_dict(),
            'XY_optimizer': self.XY_optimizer.state_dict(),
            'X': self.X,
            'Y': self.Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ma_rate': self.ma_rate,
            'ma_ef': self.ma_ef
        }

    def load_state_dict(self, state_dict):
        r"""Load the dictionary of state state_dict.
        """
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.XY_optimizer.load_state_dict(state_dict['XY_optimizer'])
        self.X = state_dict['X']
        self.Y = state_dict['Y']
        self.lr = state_dict['lr']
        self.batch_size = state_dict['batch_size']
        self.ma_rate = state_dict['ma_rate']
        self.ma_ef = state_dict['ma_ef']
