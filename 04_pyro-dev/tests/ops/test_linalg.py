# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.ops.linalg import rinverse
from tests.common import assert_close, assert_equal


@pytest.mark.parametrize(
    "A",
    [
        torch.tensor([[17.0]]),
        torch.tensor([[1.0, 2.0], [2.0, -3.0]]),
        torch.tensor([[1.0, 2, 0], [2, -2, 4], [0, 4, 5]]),
        torch.tensor([[1.0, 2, 0, 7], [2, -2, 4, -1], [0, 4, 5, 8], [7, -1, 8, 1]]),
        torch.tensor(
            [
                [1.0, 2, 0, 7, 0],
                [2, -2, 4, -1, 2],
                [0, 4, 5, 8, -4],
                [7, -1, 8, 1, -3],
                [0, 2, -4, -3, -1],
            ]
        ),
        torch.eye(40),
    ],
)
@pytest.mark.parametrize("use_sym", [True, False])
def test_sym_rinverse(A, use_sym):
    d = A.shape[-1]
    assert_equal(rinverse(A, sym=use_sym), torch.inverse(A), prec=1e-8)
    assert_equal(torch.mm(A, rinverse(A, sym=use_sym)), torch.eye(d), prec=1e-8)
    batched_A = A.unsqueeze(0).unsqueeze(0).expand(5, 4, d, d)
    expected_A = torch.inverse(A).unsqueeze(0).unsqueeze(0).expand(5, 4, d, d)
    assert_equal(rinverse(batched_A, sym=use_sym), expected_A, prec=1e-8)


# Tests migration from torch.triangular_solve -> torch.linalg.solve_triangular
@pytest.mark.filterwarnings("ignore:torch.triangular_solve is deprecated")
@pytest.mark.parametrize("upper", [False, True], ids=["lower", "upper"])
def test_triangular_solve(upper):
    b = torch.randn(5, 6)
    A = torch.randn(5, 5)
    expected = torch.triangular_solve(b, A, upper=upper).solution
    actual = torch.linalg.solve_triangular(A, b, upper=upper)
    assert_close(actual, expected)
    A = A.triu() if upper else A.tril()
    assert_close(A @ actual, b)


# Tests migration from torch.triangular_solve -> torch.linalg.solve_triangular
@pytest.mark.filterwarnings("ignore:torch.triangular_solve is deprecated")
@pytest.mark.parametrize("upper", [False, True], ids=["lower", "upper"])
def test_triangular_solve_transpose(upper):
    b = torch.randn(5, 6)
    A = torch.randn(5, 5)
    expected = torch.triangular_solve(b, A, upper=upper, transpose=True).solution
    actual = torch.linalg.solve_triangular(A.T, b, upper=not upper)
    assert_close(actual, expected)
    A = A.triu() if upper else A.tril()
    assert_close(A.T @ actual, b)
