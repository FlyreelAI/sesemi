import pytest
import torch
import torch.nn as nn
import numpy as np
import copy

from omegaconf import OmegaConf, SCMode
from hydra import initialize
from collections import OrderedDict, defaultdict

from sesemi.utils import (
    reduce_tensor,
    compute_num_devices,
    compute_gpu_device_names,
    sigmoid_rampup,
    validate_paths,
    copy_config,
    ema_update,
    copy_and_detach,
    freeze_module,
    has_length,
    get_distributed_rank,
    compute_num_digits,
    random_indices,
)
from .utils import check_state_dicts_equal


@pytest.mark.parametrize(
    "tensor,weights,reduction,expected",
    [
        (torch.tensor([1.0, 2.0, 3.0]), None, "batchmean", torch.tensor(2.0)),
        (torch.tensor([1.0, 2.0, 3.0]), None, "sum", torch.tensor(6.0)),
        (torch.tensor([1.0, 2.0, 3.0]), None, "mean", torch.tensor(2.0)),
        (torch.tensor([1.0, 2.0, 3.0]), None, "none", torch.tensor([1.0, 2.0, 3.0])),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 4.0, 0.0]),
            "batchmean",
            torch.tensor(3.0),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 4.0, 0.0]),
            "sum",
            torch.tensor(9.0),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 4.0, 0.0]),
            "mean",
            torch.tensor(1.8),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 4.0, 0.0]),
            "none",
            torch.tensor([1.0, 8.0, 0.0]),
        ),
    ],
)
def test_reduce_tensor(tensor, weights, reduction, expected):
    assert torch.allclose(
        reduce_tensor(tensor, weights=weights, reduction=reduction), expected
    )


@pytest.mark.parametrize(
    "accelerator,devices,expected",
    [
        ("gpu", 5, 5),
        ("gpu", None, 1),
    ],
)
def test_compute_num_devices(accelerator, devices, expected, mocker):
    mocker.patch(
        "pytorch_lightning.accelerators.gpu.GPUAccelerator.auto_device_count",
        side_effect=lambda *args: 1,
    )
    mocker.patch(
        "pytorch_lightning.utilities.device_parser._get_all_available_gpus",
        side_effect=lambda *args: list(range(8)),
    )
    assert compute_num_devices(accelerator, devices) == expected


@pytest.mark.parametrize(
    "accelerator,devices,expected",
    [
        ("gpu", 5, 5),
        ("gpu", None, 1),
    ],
)
def test_compute_num_devices(accelerator, devices, expected, mocker):
    mocker.patch(
        "pytorch_lightning.accelerators.gpu.GPUAccelerator.auto_device_count",
        side_effect=lambda *args: 1,
    )
    mocker.patch(
        "pytorch_lightning.utilities.device_parser._get_all_available_gpus",
        side_effect=lambda *args: list(range(8)),
    )
    assert compute_num_devices(accelerator, devices) == expected


@pytest.mark.parametrize(
    "gpus,error,expected",
    [
        (None, False, ["cpu"]),
        (0, False, ["cpu"]),
        (1, False, ["cuda:0"]),
        (5, False, [f"cuda:{i}" for i in range(5)]),
        (8, False, [f"cuda:{i}" for i in range(8)]),
        (11, True, [f"cuda:{i}" for i in range(8)]),
        (-1, False, [f"cuda:{i}" for i in range(8)]),
        ("0,1", False, ["cuda:0", "cuda:1"]),
        ("0", False, ["cuda:0"]),
        ("1", False, ["cuda:1"]),
        ("-1", False, [f"cuda:{i}" for i in range(8)]),
    ],
)
def test_compute_gpu_device_names(gpus, error, expected, mocker):
    mocker.patch(
        "torch.cuda.device_count",
        side_effect=lambda *args: 8,
    )

    if error:
        with pytest.raises(Exception):
            compute_gpu_device_names(gpus)
    else:
        assert compute_gpu_device_names(gpus) == expected


@pytest.mark.parametrize(
    "curr_iter,rampup_iters,expected",
    [
        (0, 0, 1.0),
        (0, 1000, float(np.exp(-5.0))),
        (1000, 1000, 1.0),
        (500, 1000, float(np.exp(-5.0 * 0.5 * 0.5))),
    ],
)
def test_sigmoid_rampup(curr_iter, rampup_iters, expected):
    assert np.isclose(sigmoid_rampup(curr_iter, rampup_iters), expected)


def test_validate_paths(tmp_path):
    test1 = tmp_path / "test1"
    test1.write_text("")

    test2 = tmp_path / "test2"
    test2.write_text("")

    test3 = tmp_path / "test3"

    validate_paths([str(test1), str(test2)])
    validate_paths([])

    with pytest.raises(FileNotFoundError):
        validate_paths([str(test3)])


@pytest.fixture
def training_configs():
    with initialize(config_path="../sesemi/trainer/conf/"):
        yield


def test_copy_config():
    config = OmegaConf.create(
        """
        a: 1
        b: 2
        c: ${a}        
        """
    )

    assert config.c == 1
    config.a = 5
    assert config.c == 5

    for mode in [SCMode.DICT, SCMode.DICT_CONFIG, SCMode.INSTANTIATE]:
        cloned = copy_config(config, structured_config_mode=mode)
        assert cloned["c"] == 5
        cloned["a"] = 1
        assert cloned["c"] == 5


@pytest.mark.parametrize("decay", [0.0, 1.0, 0.5, 0.99])
@pytest.mark.parametrize("copy_non_floating_point", [True, False])
def test_ema_update_parameters(decay, copy_non_floating_point):
    module = nn.Sequential(
        OrderedDict(
            {
                "linear": nn.Linear(10, 5),
                "bn": nn.BatchNorm1d(5),
            }
        )
    )

    ema_module = nn.Sequential(
        OrderedDict(
            {
                "linear": nn.Linear(10, 5),
                "bn": nn.BatchNorm1d(5),
            }
        )
    )

    original_ema_module = copy.deepcopy(ema_module)

    ema_update(
        ema_module,
        module,
        decay,
        method="parameters",
        copy_non_floating_point=copy_non_floating_point,
    )

    # Not expecting the batch norm statistics to be modified as they are not registered as parameters.
    assert check_state_dicts_equal(
        ema_module.bn.state_dict(), original_ema_module.bn.state_dict()
    )

    assert torch.allclose(
        ema_module.linear.weight.data,
        decay * original_ema_module.linear.weight.data
        + (1.0 - decay) * module.linear.weight.data,
    )

    assert torch.allclose(
        ema_module.linear.bias.data,
        decay * original_ema_module.linear.bias.data
        + (1.0 - decay) * module.linear.bias.data,
    )


@pytest.mark.parametrize("decay", [0.0, 1.0, 0.5, 0.99])
@pytest.mark.parametrize("copy_non_floating_point", [True, False])
def test_ema_update_states(decay, copy_non_floating_point):
    module = nn.Sequential(
        OrderedDict(
            {
                "linear": nn.Linear(10, 5),
                "bn": nn.BatchNorm1d(5),
            }
        )
    )

    ema_module = nn.Sequential(
        OrderedDict(
            {
                "linear": nn.Linear(10, 5),
                "bn": nn.BatchNorm1d(5),
            }
        )
    )

    original_ema_module = copy.deepcopy(ema_module)

    ema_update(
        ema_module,
        module,
        decay,
        method="parameters",
        copy_non_floating_point=copy_non_floating_point,
    )

    bn_state_dict = module.bn.state_dict()
    ema_bn_state_dict = ema_module.bn.state_dict()
    original_ema_bn_state_dict = original_ema_module.bn.state_dict()

    for key in ("weight", "bias", "running_mean", "running_var"):
        assert torch.allclose(
            ema_bn_state_dict[key],
            decay * original_ema_bn_state_dict[key]
            + (1.0 - decay) * bn_state_dict[key],
        )

    if copy_non_floating_point:
        assert (
            ema_bn_state_dict["num_batches_tracked"]
            == bn_state_dict["num_batches_tracked"]
        )
    else:
        assert (
            ema_bn_state_dict["num_batches_tracked"]
            == original_ema_bn_state_dict["num_batches_tracked"]
        )

    assert torch.allclose(
        ema_module.linear.weight.data,
        decay * original_ema_module.linear.weight.data
        + (1.0 - decay) * module.linear.weight.data,
    )

    assert torch.allclose(
        ema_module.linear.bias.data,
        decay * original_ema_module.linear.bias.data
        + (1.0 - decay) * module.linear.bias.data,
    )


def test_copy_and_detach():
    module = nn.Sequential(
        OrderedDict(
            {
                "linear": nn.Linear(10, 5),
                "bn": nn.BatchNorm1d(5),
            }
        )
    )

    clone = copy_and_detach(module)

    module_params = list(module.named_parameters())
    clone_params = list(clone.named_parameters())

    assert len(module_params) == len(clone_params)

    for (mod_name, mod_param), (clone_name, clone_param) in zip(
        module_params, clone_params
    ):
        assert mod_name == clone_name
        assert torch.allclose(mod_param, clone_param)
        assert mod_param.requires_grad
        assert not clone_param.requires_grad

    inputs = torch.ones((2, 10), dtype=torch.float32, requires_grad=True)

    outputs = torch.sum(module(inputs))
    outputs.backward()
    assert module.linear.weight.grad is not None
    assert module.linear.bias.grad is not None

    outputs = torch.sum(clone(inputs))
    outputs.backward()
    assert clone.linear.weight.grad is None
    assert clone.linear.bias.grad is None


def test_freeze_module():
    module = nn.Sequential(
        OrderedDict(
            {
                "linear": nn.Linear(10, 5),
                "bn": nn.BatchNorm1d(5),
            }
        )
    )

    original_module = copy.deepcopy(module)

    freeze_module(module)

    inputs = torch.ones((2, 10), dtype=torch.float32, requires_grad=True)

    outputs = torch.sum(module(inputs))
    outputs.backward()
    assert module.linear.weight.grad is None
    assert module.linear.bias.grad is None

    assert check_state_dicts_equal(
        module.bn.state_dict(), original_module.bn.state_dict()
    )


@pytest.mark.parametrize(
    "x,expected",
    [
        (None, False),
        (1, False),
        (1.0, False),
        ([], True),
        ((), True),
    ],
)
def test_has_length(x, expected):
    assert has_length(x) == expected


def test_get_distributed_rank(mocker):
    mocker.patch(
        "torch.distributed.is_initialized",
        side_effect=lambda *args: False,
    )
    mocker.patch(
        "torch.distributed.get_rank",
        side_effect=lambda *args: 3,
    )
    assert get_distributed_rank() == None

    mocker.patch(
        "torch.distributed.is_initialized",
        side_effect=lambda *args: True,
    )
    mocker.patch(
        "torch.distributed.get_rank",
        side_effect=lambda *args: 3,
    )
    assert get_distributed_rank() == 3

    mocker.patch(
        "torch.distributed.is_initialized",
        side_effect=lambda *args: True,
    )
    mocker.patch(
        "torch.distributed.get_rank",
        side_effect=lambda *args: 0,
    )
    assert get_distributed_rank() == 0


@pytest.mark.parametrize(
    "x,error,expected",
    [
        (0, False, 1),
        (9, False, 1),
        (10, False, 2),
        (1000, False, 4),
        (-1, True, None),
    ],
)
def test_compute_num_digits(x, error, expected):
    if error:
        with pytest.raises(Exception):
            compute_num_digits(x)
    else:
        assert compute_num_digits(x) == expected


@pytest.mark.parametrize(
    "n,length,seed,labels",
    [
        (5, 10, 42, None),
    ],
)
def test_random_indices(n, length, seed, labels):
    indices1 = random_indices(n, length, seed=seed, labels=labels)
    indices2 = random_indices(n, length, seed=seed, labels=labels)
    assert len(indices1) == n
    if seed is not None:
        assert indices1 == indices2

    if labels is not None:
        existing_labels = set(labels)
        if n >= len(existing_labels):
            assert set(indices1) == existing_labels


def test_random_indices_distribution():
    n = 5
    length = 10000
    seed = 42
    labels = [0] * 1 + [1] * 100 + [2] * 9899
    indices = random_indices(n, length, seed=seed, labels=labels)

    assert len(indices) == n

    sampled_class_counts = defaultdict(int)
    for i in indices:
        sampled_class_counts[labels[i]] += 1

    assert sampled_class_counts[0] == 1
    assert sampled_class_counts[1] == 1
    assert sampled_class_counts[2] == 3

    n = 200
    indices = random_indices(n, length, seed=seed, labels=labels)

    assert len(indices) == n

    sampled_class_counts = defaultdict(int)
    for i in indices:
        sampled_class_counts[labels[i]] += 1

    assert sampled_class_counts[0] == 1
    assert sampled_class_counts[1] in (2, 3)
    assert sampled_class_counts[2] in (196, 197)
