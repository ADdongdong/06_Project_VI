# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro.distributions as dist
from pyro.ops.gaussian import Gaussian
from tests.common import assert_close


def random_gaussian(batch_shape, dim, rank=None, *, requires_grad=False):
    """
    Generate a random Gaussian for testing.
    """
    if rank is None:
        rank = dim + dim
    log_normalizer = torch.randn(batch_shape, requires_grad=requires_grad)
    info_vec = torch.randn(batch_shape + (dim,), requires_grad=requires_grad)
    samples = torch.randn(batch_shape + (dim, rank))
    precision = torch.matmul(samples, samples.transpose(-2, -1))
    precision.requires_grad_(requires_grad)
    result = Gaussian(log_normalizer, info_vec, precision)
    assert result.dim() == dim
    assert result.batch_shape == batch_shape
    return result


def random_mvn(batch_shape, dim, *, requires_grad=False):
    """
    Generate a random MultivariateNormal distribution for testing.
    """
    rank = dim + dim
    loc = torch.randn(batch_shape + (dim,), requires_grad=requires_grad)
    cov = torch.randn(batch_shape + (dim, rank))
    cov = cov.matmul(cov.transpose(-1, -2))
    cov.requires_grad_(requires_grad)
    return dist.MultivariateNormal(loc, cov)


def assert_close_gaussian(actual, expected):
    assert isinstance(actual, Gaussian)
    assert isinstance(expected, Gaussian)
    assert actual.dim() == expected.dim()
    assert actual.batch_shape == expected.batch_shape
    assert_close(actual.log_normalizer, expected.log_normalizer)
    assert_close(actual.info_vec, expected.info_vec)
    assert_close(actual.precision, expected.precision)
