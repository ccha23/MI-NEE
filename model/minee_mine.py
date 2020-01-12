import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools


def _resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch


def _uniform_sample(data, batch_size):
    # Sample the reference uniform distribution
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])) + data_min


def _div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref_minee = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref_minee


class MINEE_MINE():
    r"""Class for Mutual Information Neural Entropic Estimation. 

    The mutual information is estimated using MINEE followed by MINE.

    Arguments:
    X (tensor): samples of X
        dim 0: different samples
        dim 1: different components
    Y (tensor): samples of Y
        dim 0: different samples
        dim 1: different components
    ref_batch_factor (float, optional): multiplicative factor to increase 
        reference sample size relative to sample size
    lr (float, optional): learning rate
    hidden_size (int, optional): size of the hidden layers
    """
    class Net(nn.Module):
        # Inner class that defines the neural network architecture
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

    def __init__(self, X, Y, batch_size=32, ref_batch_factor=1, lr=1e-3, ma_rate=0.1, hidden_size=100, ma_ef=1):
        self.lr = lr
        self.batch_size = batch_size
        self.ma_rate = ma_rate
        self.ref_batch_factor = ref_batch_factor
        self.X = X
        self.Y = Y
        self.XY = torch.cat((self.X, self.Y), dim=1)

        self.X_ref_minee = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        self.Y_ref_minee = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))

        self.X_ref_mine = _resample(self.X, batch_size=self.X.shape[0])
        self.Y_ref_mine = _resample(self.Y, batch_size=self.Y.shape[0])

        self.XY_net = MINEE_MINE.Net(
            input_size=X.shape[1]+Y.shape[1], hidden_size=100)
        self.X_net = MINEE_MINE.Net(input_size=X.shape[1], hidden_size=100)
        self.Y_net = MINEE_MINE.Net(input_size=Y.shape[1], hidden_size=100)
        self.XY_optimizer_minee = optim.Adam(self.XY_net.parameters(), lr=lr)
        self.X_optimizer_minee = optim.Adam(self.X_net.parameters(), lr=lr)
        self.Y_optimizer_minee = optim.Adam(self.Y_net.parameters(), lr=lr)
        self.XY_optimizer_mine = optim.Adam(itertools.chain(self.XY_net.parameters(),
                                                            self.X_net.parameters(),
                                                            self.Y_net.parameters()), lr=lr)

        self.ma_ef = ma_ef  # for moving average


    def cat(self, X, Y):
        # concatenate X and Y to XY
        return torch.cat((X, Y), dim=1)


    def split(self, XY):
        # split XY to X and Y
        return torch.split(XY, [self.X.shape[1], self.Y.shape[1]], dim=1)


    def step_minee(self, iter=1):
        r"""Train the neural networks for one or more steps using MINEE.

        Argument:
        iter (int, optional): number of steps to train.
        """
        for i in range(iter):
            self.XY_optimizer_minee.zero_grad()
            self.X_optimizer_minee.zero_grad()
            self.Y_optimizer_minee.zero_grad()
            batch_XY = _resample(self.XY, batch_size=self.batch_size)
            batch_X = _resample(self.X, batch_size=self.batch_size)
            batch_Y = _resample(self.Y, batch_size=self.batch_size)
            batch_X_ref_minee = _uniform_sample(self.X, batch_size=int(
                self.ref_batch_factor * self.batch_size))
            batch_Y_ref_minee = _uniform_sample(self.Y, batch_size=int(
                self.ref_batch_factor * self.batch_size))
            batch_XY_ref_minee = torch.cat(
                (batch_X_ref_minee, batch_Y_ref_minee), dim=1)

            batch_loss_XY = -_div(self.XY_net, batch_XY, batch_XY_ref_minee)
            batch_loss_XY.backward()
            self.XY_optimizer_minee.step()

            batch_loss_X = -_div(self.X_net, batch_X, batch_X_ref_minee)
            batch_loss_X.backward()
            self.X_optimizer_minee.step()

            batch_loss_Y = -_div(self.Y_net, batch_Y, batch_Y_ref_minee)
            batch_loss_Y.backward()
            self.Y_optimizer_minee.step()

    def step_mine(self, iter=1):
        r"""Train the neural networks for one or more steps using MINE.

        Argument:
        iter (int, optional): number of steps to train.
        """
        for i in range(iter):
            self.XY_optimizer_mine.zero_grad()
            batch_XY = _resample(self.XY, batch_size=self.batch_size)
            batch_X, batch_Y = self.split(batch_XY)
            # batch_X = _resample(self.X, batch_size=self.batch_size)
            # batch_Y = _resample(self.Y, batch_size=self.batch_size)
            batch_X_ref_mine = _resample(self.X, batch_size=self.batch_size)
            batch_Y_ref_mine = _resample(self.Y, batch_size=self.batch_size)
            batch_XY_ref_mine = self.cat(batch_X_ref_mine, batch_Y_ref_mine)
            # batch_XY_ref_mine = torch.cat(
            #     (batch_X_ref_mine, batch_Y_ref_mine), dim=1)
            # define the loss function with moving average in the gradient estimate
            mean_fXY = self.XY_net(batch_XY).mean(
            ) - self.X_net(batch_X).mean() - self.Y_net(batch_Y).mean()
            # mean_efXY_ref_mine = torch.logsumexp(self.XY_net(batch_XY_ref_mine) - self.X_net(batch_X_ref_mine) - self.Y_net(batch_Y_ref_mine), 0) - np.log(batch_XY_ref_mine.shape[0])
            mean_efXY_ref_mine = torch.exp(
                self.XY_net(batch_XY_ref_mine) - self.X_net(batch_X_ref_mine) - self.Y_net(batch_Y_ref_mine)).mean()
            self.ma_ef = (1-self.ma_rate)*self.ma_ef + \
                self.ma_rate*mean_efXY_ref_mine
            batch_loss_XY = - mean_fXY + \
                (1 / self.ma_ef.mean()).detach() * mean_efXY_ref_mine
            batch_loss_XY.backward()
            self.XY_optimizer_mine.step()

    def forward_minee(self, X=None, Y=None):
        r"""Evaluate the neural networks to return an array of 3 divergences estimates 
        (dXY, dX, dY).

        Outputs:
            dXY: divergence of sample joint distribution of (X,Y) 
                to the uniform reference
            dX: divergence of sample marginal distribution of X 
                to the uniform reference
            dY: divergence of sample marginal distribution of Y
                to the uniform reference

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
        X_ref_minee = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        Y_ref_minee = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))
        XY_ref_minee = torch.cat((X_ref_minee, Y_ref_minee), dim=1)
        dXY = _div(self.XY_net, XY, XY_ref_minee).cpu().item()
        dX = _div(self.X_net, X, X_ref_minee).cpu().item()
        dY = _div(self.Y_net, Y, Y_ref_minee).cpu().item()
        return dXY, dX, dY

    def estimate_minee(self, X=None, Y=None):
        r"""Return the mutual information estimate.

        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used. 
        The arguments are useful for testing/validation with a separate data set.
        """
        dXY, dX, dY = self.forward_minee(X, Y)
        return dXY - dX - dY

    def forward_mine(self, X=None, Y=None):
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
            XY = self.cat(X, Y)
        X_ref_mine = _resample(X, batch_size=X.shape[0])
        Y_ref_mine = _resample(Y, batch_size=Y.shape[0])
        XY_ref_mine = self.cat(X_ref_mine, Y_ref_mine)
        mean_f = self.XY_net(XY).mean() - \
            self.X_net(X).mean() - self.Y_net(Y).mean()
        log_mean_ef_ref_mine = torch.logsumexp(self.XY_net(XY_ref_mine) - self.X_net(
            X_ref_mine) - self.Y_net(Y_ref_mine), 0) - np.log(XY_ref_mine.shape[0])
        return (mean_f - log_mean_ef_ref_mine).cpu().item()

    def estimate_mine(self, X=None, Y=None):
        r"""Return the mutual information estimate.

        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used. 
        The arguments are useful for testing/validation with a separate data set.
        """
        return self.forward_mine(X, Y)

    def state_dict(self):
        r"""Return a dictionary storing the state of the estimator.
        """
        return {
            'XY_net': self.XY_net.state_dict(),
            'XY_optimizer_minee': self.XY_optimizer_minee.state_dict(),
            'X_net': self.X_net.state_dict(),
            'X_optimizer_minee': self.X_optimizer_minee.state_dict(),
            'Y_net': self.Y_net.state_dict(),
            'Y_optimizer_minee': self.Y_optimizer_minee.state_dict(),
            'X': self.X,
            'Y': self.Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ref_batch_factor': self.ref_batch_factor
        }

    def load_state_dict(self, state_dict):
        r"""Load the dictionary of state state_dict.
        """
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.XY_optimizer_minee.load_state_dict(
            state_dict['XY_optimizer_minee'])
        self.X_net.load_state_dict(state_dict['X_net'])
        self.X_optimizer_minee.load_state_dict(state_dict['X_optimizer_minee'])
        self.Y_net.load_state_dict(state_dict['Y_net'])
        self.Y_optimizer_minee.load_state_dict(state_dict['Y_optimizer_minee'])
        self.X = state_dict['X']
        self.Y = state_dict['Y']
        if 'lr' in state_dict:
            self.lr = state_dict['lr']
        if 'batch_size' in state_dict:
            self.batch_size = state_dict['batch_size']
        if 'ref_batch_factor' in state_dict:
            self.ref_batch_factor = state_dict['ref_batch_factor']
