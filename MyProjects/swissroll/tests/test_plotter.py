import pytest
import numpy as np
import swissroll.plotter as srp
import swissroll.distributions as srd


def test_plot_pmf_length():
    msg = f"length of support must be equal" f" to distributions!"

    dists = [srd.Bernoulli(0.3), srd.Binomial(10.0, 0.2)]
    support = [np.array([1.0, 2.0])]

    with pytest.raises(ValueError, match=msg):
        srp.plot_pmf(dists, support)


def test_plot_pmf_continuous():
    msg = (
        f"Continuous distributions don't have"
        f" a mass function! Try swissroll.plotter.plot_pdf"
    )

    dists = [srd.Exponential(1.0)]
    support = [np.array([0.0, 1.0, 2.0])]

    with pytest.raises(ValueError, match=msg):
        srp.plot_pmf(dists, support)


def test_plot_pdf_length():
    msg = f"length of support must be equal" f" to distributions!"

    dists = [srd.Exponential(3), srd.Cauchy(0.0, 10.0)]
    support = [np.array([1.0, 2.0])]

    with pytest.raises(ValueError, match=msg):
        srp.plot_pdf(dists, support)


def test_plot_pdf_discrete():
    msg = (
        f"Discrete distributions don't have"
        f" a density function!"
        f" Try `swissroll.plotter.plot_pmf`"
    )

    dists = [srd.Bernoulli(1.0)]
    support = [np.array([0.0, 1.0, 2.0])]

    with pytest.raises(ValueError, match=msg):
        srp.plot_pdf(dists, support)


def test_plot_cdf_length():
    msg = f"length of support must be equal" f" to distributions!"

    dists = [srd.Exponential(3), srd.Cauchy(0.0, 10.0)]
    support = [np.array([1.0, 2.0])]

    with pytest.raises(ValueError, match=msg):
        srp.plot_cdf(dists, support)


def test_plot_ecdf():
    # present for coverage only!
    # no test required!
    srp.plot_ecdf(np.array([1.0]))

    assert True


def test_pmf_pdf_cdf_support_typecasting():
    dist1, dist2 = srd.Binomial(20.0, 0.3), srd.Binomial(20.0, 0.7)
    dist3, dist4 = srd.Normal(0.0, 1.0), srd.Normal(0.0, 4.0)

    support = np.arange(0.0, 20.0, dtype=np.float32)

    srp.plot_pmf([dist1, dist2], support)
    srp.plot_pdf([dist3, dist4], support)
    srp.plot_cdf([dist1, dist2], support)
    srp.plot_cdf([dist3, dist4], support)

    assert True


def test_plot_hist():
    # present for coverage only!
    # no test required!
    srp.plot_hist(np.array([1.0]))

    assert True


def test_day_what():
    # present for coverage only!
    # no test required!
    srp.say_what()

    assert True


if __name__ == "__main__":
    pytest.main()
