import os
import torch
import numpy as np


def read_version(version_file):
    with open(version_file, encoding="utf-8") as v:
        return v.read().rstrip()


__version__ = read_version(os.path.join(os.path.dirname(__file__), "..", "version"))

torch.set_printoptions(precision=6)

# Set data type in: fromScratHamiltonianAT, analyzeOutput, getInput, initialize
use_cuda = torch.cuda.is_available()
if use_cuda:
    map_location = lambda storage, loc: storage.cuda(0)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    map_location = torch.device("cpu")

torch_dtype = torch.float32
torch.set_default_dtype(torch_dtype)
torch.set_default_device(device)


def normalize(x, p=1.0, dim=1):
    """
    Normalize a tensor or numpy array along a given dimension. Return a copy of the input.

    example:
    >>> a = np.row_stack((np.zeros((3,2)), np.ones((1, 2))))
    >>> normalize(a, p=2, dim=1)
    >>> b = torch.tensor(a)
    >>> normalize(b, p=2, dim=1)

    """
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.normalize(x, p=p, dim=dim)
    else:
        tmp = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
        gd = tmp.squeeze() > 0
        res = np.copy(x)
        res[gd] = res[gd] / tmp[gd]
        return res


#
# a = np.row_stack((np.zeros((3,2)), np.ones((1, 2))))
# print(a)
# print(a.__array_interface__['data'])
#
# b = normalize(a)
# print(b)
# print(b.__array_interface__['data'][0])
# print(a)
#
#
# c = torch.tensor(a)
# print(c.data_ptr())
#
# d = normalize(c)
# print(d.data_ptr())
#
# print(c)
# print(d)


import xmodmap.distance as distance
import xmodmap.deformation as deformation
import xmodmap.model as model