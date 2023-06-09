# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.transforms import AffineTransform, ExpTransform
from pyro.infer.reparam import TransformReparam
from tests.common import assert_close

from .util import check_init_reparam


# Test helper to extract a few log central moments from samples.
def get_moments(x):
    assert (x > 0).all()
    x = x.log()
    m1 = x.mean(0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = xx.mean(0)
    m3 = xxx.mean(0) / m2**1.5
    m4 = xxxx.mean(0) / m2**2
    return torch.stack([m1, m2, m3, m4])


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("event_shape", [(), (5,)], ids=str)
def test_log_normal(batch_shape, event_shape):
    shape = batch_shape + event_shape
    loc = torch.empty(shape).uniform_(-1, 1)
    scale = torch.empty(shape).uniform_(0.5, 1.5)

    def model():
        fn = dist.TransformedDistribution(
            dist.Normal(torch.zeros_like(loc), torch.ones_like(scale)),
            [AffineTransform(loc, scale), ExpTransform()],
        )
        if event_shape:
            fn = fn.to_event(len(event_shape))
        with pyro.plate_stack("plates", batch_shape):
            with pyro.plate("particles", 200000):
                return pyro.sample("x", fn)

    with poutine.trace() as tr:
        value = model()
    assert isinstance(
        tr.trace.nodes["x"]["fn"], (dist.TransformedDistribution, dist.Independent)
    )
    expected_moments = get_moments(value)

    with poutine.reparam(config={"x": TransformReparam()}):
        with poutine.trace() as tr:
            value = model()
    assert isinstance(tr.trace.nodes["x"]["fn"], (dist.Delta, dist.MaskedDistribution))
    actual_moments = get_moments(value)
    assert_close(actual_moments, expected_moments, atol=0.05)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("event_shape", [(), (5,)], ids=str)
def test_init(batch_shape, event_shape):
    shape = batch_shape + event_shape
    loc = torch.empty(shape).uniform_(-1, 1)
    scale = torch.empty(shape).uniform_(0.5, 1.5)

    def model():
        fn = dist.TransformedDistribution(
            dist.Normal(torch.zeros_like(loc), torch.ones_like(scale)),
            [AffineTransform(loc, scale), ExpTransform()],
        )
        if event_shape:
            fn = fn.to_event(len(event_shape))
        with pyro.plate_stack("plates", batch_shape):
            return pyro.sample("x", fn)

    check_init_reparam(model, TransformReparam())
