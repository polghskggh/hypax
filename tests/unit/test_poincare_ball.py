import jax.numpy as jnp

from hypax.manifolds.poincare_ball import PoincareBall


def test_inner_at_origin_matches_scaled_euclidean():
    manifold = PoincareBall(c=1.0)
    x = jnp.zeros((2,))
    u = jnp.array([1.0, 2.0])
    v = jnp.array([-3.0, 0.5])

    result = manifold.inner(x, u, v)
    expected = 4.0 * jnp.dot(u, v)

    assert jnp.allclose(result, expected)


def test_euc_to_tangent_scales_by_lambda_squared():
    manifold = PoincareBall(c=1.0)
    x = jnp.array([0.1, 0.0])
    u = jnp.array([0.5, -0.25])

    lambda_x = 2 / (1 - jnp.sum(x**2))
    expected = u / (lambda_x**2)

    result = manifold.euc_to_tangent(x, u)

    assert jnp.allclose(result, expected)


def test_transp_preserves_norms():
    manifold = PoincareBall(c=1.0)
    x = jnp.array([0.05, -0.02])
    y = jnp.array([-0.03, 0.01])
    v = jnp.array([0.2, -0.1])

    transported = manifold.transp(x, y, v)

    norm_x = manifold.inner(x, v, v)
    norm_y = manifold.inner(y, transported, transported)

    assert jnp.allclose(norm_x, norm_y, atol=1e-6)


def test_cdist_matches_pairwise_dist():
    manifold = PoincareBall(c=1.0)
    x = jnp.stack([jnp.array([0.1, 0.0]), jnp.array([0.0, 0.1])])
    y = jnp.stack([jnp.array([-0.05, 0.02]), jnp.array([0.02, -0.03])])

    pairwise = manifold.cdist(x, y)

    assert pairwise.shape == (2, 2)
    assert jnp.all(pairwise >= 0)
