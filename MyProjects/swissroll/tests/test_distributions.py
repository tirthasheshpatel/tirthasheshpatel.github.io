import numpy as np
import swissroll.distributions as srd
import pytest


@pytest.mark.parametrize("dist", [srd.Bernoulli(0.3)])
@pytest.mark.parametrize(
    "x, expected",
    [(np.array([1.0]), np.array([0.3])), (np.array([0.0]), np.array([0.7]))],
)
def test_bernoulli_pmf(dist, x, expected):
    assert dist.pmf(x) == expected


@pytest.mark.parametrize("dist", [srd.Bernoulli(0.3)])
@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([-1.0]), np.array([0.0])),
        (np.array([0.5]), np.array([0.7])),
        (np.array([1.5]), np.array([1.0])),
    ],
)
def test_bernoulli_cdf(dist, x, expected):
    assert dist.cdf(x) == expected


def test_bernoulli_meanvar():
    dist = srd.Bernoulli(0.3)
    assert dist.meanvar() == (0.3, 0.21)


def test_bernoulli_mle():
    dist = srd.Bernoulli(0.3)
    samples = np.array([1.0, 1.0, 0.0, 1.0])

    assert dist.mle(samples) == 0.75


# @pytest.mark.parametrize(
#     'dist', [srd.Binomial(5., 0.3)]
# )
# @pytest.mark.parametrize(
#     'x, expected', [(np.array([1.0]), np.array([0.3])),
#                     (np.array([0.0]), np.array([0.7]))]
# )
# def test_binomial_pmf(dist, x, expected):
#     assert dist.pmf(x) == expected


@pytest.mark.parametrize(
    "dist", [srd.Bernoulli(0.3), srd.Exponential(4.0), srd.Poisson(1.0),]
)
def test_like_loglike(dist):
    # present for coverage only!
    # not test required.
    samples = dist.sample(10)
    dist.plot_log_likelihood(samples)
    dist.plot_likelihood(samples)
    assert True


def test_like_loglike_binomial():
    # present for coverage only!
    # not test required.
    dist = srd.Binomial(20.0, 0.3)
    samples = dist.sample(10)
    dist.plot_likelihood(5.0, samples)
    dist.plot_log_likelihood(5.0, samples)
    assert True


def test_like_loglike_normal():
    # present for coverage only!
    # not test required.
    dist = srd.Normal(0.0, 1.0)
    samples = dist.sample(10)
    dist.plot_likelihood(samples, mu=0.0)
    dist.plot_likelihood(samples, var=1.0)
    dist.plot_log_likelihood(samples, mu=0.0)
    dist.plot_log_likelihood(samples, var=1.0)


if __name__ == "__main__":
    pytest.main()
