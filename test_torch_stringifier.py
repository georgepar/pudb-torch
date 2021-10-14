import numpy
import torch
from pudb.var_view import default_stringifier
from torch_stringifier import pudb_stringifier

def test_tensor():
    x = torch.randn(10, 5, 4)
    assert pudb_stringifier(x) == "Tensor[float32][cpu] [10, 5, 4]"


def test_conv_module():
    x = torch.nn.Conv2d(20, 10, 3)
    assert pudb_stringifier(x) == "Conv2d(20, 10, kernel_size=(3, 3), stride=(1, 1))[cpu] Params: 1810"


def test_linear_module():
    x = torch.nn.Linear(5, 2, bias=False)
    assert pudb_stringifier(x) == "Linear(in_features=5, out_features=2, bias=False)[cpu] Params: 10"


def test_long_module_repr_should_revert_to_type():
    x = torch.nn.Transformer()
    assert pudb_stringifier(x) == "Transformer[cpu] Params: 44140544"


def test_reverts_to_default_for_numpy():
    x = numpy.random.randn(10, 15)
    assert pudb_stringifier(x) == default_stringifier(x)


def test_reverts_to_default_for_str():
    x = "Everyone has his day, and some days last longer than others."
    assert pudb_stringifier(x) == default_stringifier(x)


def test_reverts_to_default_for_dict():
    x = {"a": 1, "b": 2, "c": 3}
    assert pudb_stringifier(x) == default_stringifier(x)


def test_reverts_to_default_for_list():
    x = list(range(1000))
    assert pudb_stringifier(x) == default_stringifier(x)
