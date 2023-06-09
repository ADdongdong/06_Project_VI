# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


class TransformModule(torch.distributions.Transform, torch.nn.Module):
    """
    Transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()


class ComposeTransformModule(torch.distributions.ComposeTransform, torch.nn.ModuleList):
    """
    This allows us to use a list of `TransformModule` in the same way as
    :class:`~torch.distributions.transform.ComposeTransform`. This is needed
    so that transform parameters are automatically registered by Pyro's param
    store when used in :class:`~pyro.nn.module.PyroModule` instances.
    """

    def __init__(self, parts, cache_size=0):
        super().__init__(parts, cache_size=cache_size)
        for part in parts:
            if isinstance(part, torch.nn.Module):
                self.append(part)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()

    def with_cache(self, cache_size=1):
        if cache_size == self._cache_size:
            return self
        return ComposeTransformModule(self.parts, cache_size=cache_size)
