"""
cross entropy
"""
import torch
from torch.autograd import Function

class CrossEntropy(Function):
    """
    CrossEntropy function with custom forward and backward propagation function
    """
    def forward(self, x, target):
        r"""
        forward propagation
        function:
        E(t_i, x) = - \sum_i t_i log x, where t_i is one-hot label
        """
        output = - target * torch.log(x).mean()
        self.target = target
        self.x = x
        return output

    def backward(self, grad_output):
        r"""
        backward propagation
        gradient:
        \frac{\partial E}{\partial x} = - t_i / x
        """
        grad_input = - grad_output * self.target / self.x /self.x.size(0)
        return grad_input, None
    

