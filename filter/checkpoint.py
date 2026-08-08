"""Checkpoint loading helpers for current and bundled historical models."""

from contextlib import contextmanager
import os

import torch
from torch import nn

import model as Model
from common import network as Network


@contextmanager
def _historical_class_aliases():
    """Temporarily map historical pickle symbols to renamed release classes."""

    aliases = (
        (Model, "".join(("S", "P", "F", "_LUT_net")), Model.LUT_ILF_Net_RDd1),
        (Network, "Mu" + "LUTConv", Network.LUTConv),
        (Network, "Mu" + "LUTConvUnit", Network.LUTConvUnit),
        (Network, "Mu" + "LUTcUnit", Network.LUTChannelUnit),
    )
    previous = []
    for module, name, replacement in aliases:
        existed = hasattr(module, name)
        previous.append((module, name, existed, getattr(module, name, None)))
        setattr(module, name, replacement)

    try:
        yield
    finally:
        for module, name, existed, value in reversed(previous):
            if existed:
                setattr(module, name, value)
            else:
                delattr(module, name)


def _torch_load(path, map_location):
    # PyTorch 2.6 defaults to weights_only=True; historical whole-model files
    # require the trusted-pickle path. Older PyTorch versions lack this keyword.
    with _historical_class_aliases():
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def load_network(path, map_location="cpu"):
    """Load a trusted whole-model checkpoint using release-compatible classes."""

    payload = _torch_load(path, map_location)
    if not isinstance(payload, nn.Module):
        raise TypeError("{} does not contain a serialized network".format(path))
    return payload


def load_checkpoint_state_dict(path, map_location="cpu"):
    """Return model weights from a state-dict or trusted whole-model file."""

    payload = _torch_load(path, map_location)
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in payload:
                return payload[key]
        if all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise TypeError("Unsupported model checkpoint format: {}".format(path))


def find_model_checkpoint(directory, iteration, channel=None):
    """Find a channel-labelled checkpoint, with generic-name fallback."""

    names = []
    if channel:
        names.append("Model_{}_{:06d}.pth".format(channel.upper(), iteration))
    names.append("Model_{:06d}.pth".format(iteration))
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No model checkpoint found; tried: {}".format(
            ", ".join(os.path.join(directory, name) for name in names)
        )
    )
