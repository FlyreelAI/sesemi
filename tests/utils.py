import pytest

from typing import Dict, Callable, Literal, Tuple
from PIL import Image

import torch
import numpy as np
import os
import subprocess
import sys
from pathlib import Path
from subprocess import TimeoutExpired

import pytorch_lightning
import yaml
import sesemi
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sesemi.config.structs import SESEMIBaseConfig


def run_trainer(
    tmp_path: Path, config: SESEMIBaseConfig, timeout: int = 1800
) -> Tuple[str, str, int]:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with open(str(config_dir / "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    command = ["open_sesemi", "-cd", str(config_dir), "-cn", "config"]

    # need to set the PYTHONPATH in case pytorch_lightning was not installed into the environment
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        env.get("PYTHONPATH", "")
        + f"{pytorch_lightning.__file__}:"
        + f"{sesemi.__file__}:"
    )

    # for running in ddp mode, we need to launch it's own process or pytest will get stuck
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )
    try:
        for c in iter(lambda: p.stdout.read(1), b""):
            sys.stdout.buffer.write(c)

        std, err = p.communicate(timeout=timeout)
        err = str(err.decode("utf-8"))
        if "Exception" in err:
            raise Exception(err)
    except TimeoutExpired:
        p.kill()
        std, err = p.communicate(timeout=10)
    return std, err, p.returncode


def state_dicts_equal(
    x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor], **kwargs
) -> bool:
    """Return if both state dicts are the equivalent."""
    equal = list(x.keys()) == list(y.keys())
    if not equal:
        return False

    for k, v in x.items():
        equal &= (v.dtype == y[k].dtype) and (v.shape == y[k].shape)
        if not equal:
            return False

        equal = torch.allclose(v, y[k].to(v.device), **kwargs)

    return equal


def check_state_dicts_equal(
    x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor], **kwargs
):
    assert state_dicts_equal(x, y, **kwargs)


def check_state_dicts_not_equal(
    x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor], **kwargs
):
    assert not state_dicts_equal(x, y, **kwargs)


def check_image_transform(
    transform: Callable, input_shape: Tuple[int, int], output_shape: Tuple[int, int]
):
    """Checks that the image transform produces valid outputs."""
    image = Image.fromarray(np.ones((*input_shape, 3), dtype=np.uint8))
    outputs = transform(image)
    if isinstance(outputs, Image.Image):
        assert outputs.size == (output_shape[1], output_shape[0])
    elif isinstance(outputs, torch.Tensor):
        assert outputs.shape == (3, *output_shape)


def count_parameters(module: torch.nn.Module) -> int:
    """Counts the number of parameter elements in the module."""
    return sum(x.numel() for x in module.parameters())


def round_nth_digit(num: int, n: int) -> int:
    """Rounds the nth digit.

    Args:
        num: The number of round.
        n: The 0-indexed digit.

    Return:
        The rounded integer.
    """
    assert n >= 0
    divisor = 10**n
    x = num / float(divisor)
    y = round(x)
    return int(y) * divisor


def check_model_frozen_state(
    model: torch.nn.Module,
    frozen: bool,
    input_shape: Tuple[int] = (2, 3, 32, 32),
):
    """Runs forward and backward passes to ensure that a model's parameters are frozen."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pre_gradient_parameters = list(x.clone() for x in model.parameters())

    generator = torch.Generator()
    generator.manual_seed(42)

    inputs = torch.randn(
        input_shape, dtype=torch.float32, requires_grad=True, generator=generator
    )
    outputs = model(inputs)

    loss = torch.sum(outputs + 100)
    loss.backward()
    optimizer.step()

    post_gradient_parameters = list(x.clone() for x in model.parameters())

    all_close = True
    for x, y in zip(pre_gradient_parameters, post_gradient_parameters):
        all_close &= torch.allclose(x, y)

    if frozen:
        assert all_close
    else:
        assert not all_close


def load_tensorboard_scalar_events(path: str) -> pd.DataFrame:
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()

    events = []
    for tag in event_accumulator.Tags()["scalars"]:
        for item in event_accumulator.Scalars(tag):
            events.append(
                dict(
                    tag=tag,
                    step=item.step,
                    wall_time=item.wall_time,
                    value=item.value,
                )
            )

    return pd.DataFrame(events, columns=["tag", "step", "wall_time", "value"])
