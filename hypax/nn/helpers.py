from typing import Callable

import jax

from hypax.array import ManifoldArray


def tangent_space_fn(tangent_fun: Callable[[jax.Array], jax.Array]) -> Callable[[ManifoldArray], ManifoldArray]:
    def fun(x: ManifoldArray) -> jax.Array:
        tangent = x.manifold.logmap(x.data, axis=x.axis)
        tangent = tangent_fun(tangent)
        result = x.manifold.expmap(tangent, axis=x.axis)
        return x.replace(data=result)
    return fun
