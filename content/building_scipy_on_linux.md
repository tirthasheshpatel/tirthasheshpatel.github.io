Title: Building SciPy from source on Ubuntu 20.04/18.04
Author: Tirth Patel
Date: 2022-02-04 17:30
Category: SciPy
Tags: SciPy
Alias: /blogs/building_scipy_ubuntu.html, /blogs/building_scipy_ubuntu/index.html
<!-- Modified: 2022-02-04 17:30 -->

## Introduction

In this post, I will go through two ways to build [SciPy](https://github.com/scipy/scipy) on Ubuntu 20.04 which hopefully also work on other distros! SciPy has a lot of build-time dependencies and there is no proper guide on the documentation pages that explicitly lists all them and how to get them. This makes it harder for new devs to get used to the development environment. The primary focus of this post is to guide you step-by-step from installation of dependencies to building SciPy from source and running tests on a Debian distro.

At the time of writing this, SciPy 1.9.0 is in development and instructions are written accordingly. I will try to keep this up-to-date with the latest release of SciPy. In the future, you will find an edit log below.

### Step 0: Getting SciPy.

Clone SciPy using either this:

```shell
git clone git@github.com:scipy/scipy.git # if you are using HTTPS, use "git clone https://github.com/scipy/scipy.git" instead
cd scipy
git submodule update --init
```

or this:

```shell
mkdir scipy
cd scipy
git init
git remote add upstream git@github.com:scipy/scipy.git
# if you have a fork, set up a remote accordingly: git remote add origin <link_to_your_remote>
git fetch upstream
git checkout --track -b main upstream/main
git submodule update --init
```

If you already have SciPy cloned and you tried to make a build but failed, just clean the environment:

```shell
cd /path/to/your/scipy/clone
git clean -xdf
git pull  # make sure you have latest SciPy source code
```

### Step 1: Setting up Python

SciPy uses the Python C API heavily. So you will need Python header files to build it. If you want to use relatively newer versions of Python, you can add the `deadsnakes` PPA repo to your apt sources:

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

Now, you can get your desired version of Python and its header files by running the following command:

```shell
# replace the 3.9 suffix with the version you want. Python >= 3.8 is required for SciPy 1.9.0.
sudo apt install python3.9 python3.9-dev
```

Note that since SciPy 1.9.0, only Python >= 3.8 is supported.

### Step 2: Getting Ubuntu Dependencies

SciPy relies on gcc, g++, gfortran, ccache (optional), OpenBLAS, ATLAS, LAPACK, suitesparse, and some other arithmetic libraries. You can get all the dependencies by running:

```shell
sudo apt-get install -y build-essential libopenblas-dev libatlas-base-dev liblapack-dev gfortran libgmp-dev libmpfr-dev libsuitesparse-dev ccache libmpc-dev
```

### Step 3: Getting Python dependencies

To install Python dependencies, you will need either pip or (ana)conda.

#### Step 3.1: Using `pip`

Usually, on Ubuntu, Python doesn't come with `pip` preinstalled, so you will have to get it using:

```shell
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py
```

Once you have `pip`, you will need a package to create and manage virtual environments. I use `virtualenv`:

```shell
python3.9 -m pip install -U pip
python3.9 -m pip install virtualenv
```

You can also get and use `venv` using `sudo apt install python3.9-venv` (replace `python3.9` with your version of Python).

It's good practice to install Python dependencies in a virtual environment. So, first, create and activate an environment:

```shell
mkdir -p ~/.virtualenvs
python3.9 -m virtualenv ~/.virtualenvs/scipy-dev
source ~/.virtualenvs/scipy-dev/bin/activate
python3.9 -m pip install -U pip wheel
```

Note that you should not create a virtual environment in the same directory where the SciPy source exists. This creates some problems when building with Meson (See [Step 4.2](#step-42-getting-build-time-dependencies-for-meson-build)).

The following Python packages are required to build SciPy and run tests:

- `Cython`: SciPy heavily relies on Cython to accelarate Python code.
- `numpy>=1.18.5`: SciPy relies on NumPy for array representation and array operations.
- `pybind11`: To create Python bindings for C++ code.
- `pythran>=0.9.12`: Another dependency to accelarate Python code.
- `pytest`: To run Python tests.
- `pytest-xdist`: For some pytest utilities.
- `pytest-cov` (optional): To generate coverage reports with pytest.
- `mpmath` (optional): To run some tests against `mpmath`
- `gmpy2` (optional): To run some tests against `gmpy2`
- `flake8` (optional): For static analysis
- Dependencies listed in `mypy_requirements.txt` for static analysis. (optional)
- Dependencies listed in `doc_requirements.txt` to build docs. (optional)

You can get all the required dependencies using:

```shell
pip install -U Cython 'numpy>=1.18.5' pybind11 'pythran>=0.9.12' pytest pytest-xdist
```

To install optional dependencies, do:

```shell
pip install -U pytest-cov mpmath gmpy2 flake8
pip install -r mypy_requirements.txt
pip install -r doc_requirements.txt
```

#### Step 3.2: Using `conda`

Download Anaconda or Miniconda from [this guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

Once you have anaconda or miniconda installled, you can build an environment and install all the dependencies by running the following command from SciPy's root directory:

```shell
conda env create -f environment.yml
conda activate scipy-dev
```

You are now all set to build and test SciPy!

### Step 4: Building and Testing SciPy

SciPy allows building using Python's [distutils module](https://docs.python.org/3/library/distutils.html) or using the [Meson build system](https://mesonbuild.com/Manual.html). It is recommended to use Meson since distutils is a candidate for deprecation starting Python 3.10.

#### Step 4.1: Using disutils

To build using Python's distutils module, you will first need to downgrade setuptools to any version less than v60.0.0.

```shell
pip install -U 'setuptools<60.0.0'
```

Now, you can start building SciPy either using the `runtests.py` utility:

```shell
python runtests.py --build-only
```

Or you can create an inplace build (recommended if you are going to change SciPy source code):

```shell
python setup.py build_ext -i
```

To verify the inplace build, run `python -c 'import scipy; print(scipy.__version__)'`. This should exit normally and print something like `1.9.0dev...` if the build succeeded.

That's it! You have now built SciPy from source and can start testing it.

**Running Tests**

If you used `runtests.py` to build SciPy, run the following command to start testing:

```shell
# change the `--tests` option according to the tests you want to run.
python runtests.py --tests scipy/stats  # tests scipy.stats submodule
python runtests.py --submodule special  # tests scipy.special submodule
python runtests.py  # runs full test suite
```

If you created an inplace build, use `pytest` to run tests

```shell
pytest scipy/stats  # runs tests in the scipy.stats submodule
pytest scipy/stats/tests/test_sampling.py  # runs all the tests in the file `test_sampling.py`
```

#### Step 4.2: Using Meson (recommended)

Meson is the recommended way to build SciPy as it allows having multiple builds in parallel and is much faster (~8-10 times) than distutils. Also, as distutils is being deprecated in Python 3.10 and will be removed in Python 3.12, SciPy will only support Meson as the primary build system in the future.

Make sure you have `setuptools>60.0.0` before running meson.

```shell
pip install -U 'setuptools>60.0.0'
```

You will need to install Meson and Ninja build systems using either `pip` or `conda`.

**Using `pip`**

```shell
pip install -U meson ninja
```

**Using `conda`**

```shell
conda install meson ninja
```

Now, you can build SciPy with Meson either using either the `dev.py` utility (recommended):

```shell
# If you have more/less CPUs change the `-j12` option accordingly
python dev.py -j12 --build-only
```

Or you can build by configuring Meson manually:

```shell
meson setup builddir --prefix '$PWD/installdir'
cd builddir
# If you have more/less CPUs, change the `-j12` option accordingly
ninja install -j12
# Replace `python3.9` below with the version of Python you are using
export PYTHONPATH=$PWD/installdir/lib/python3.9/site-packages
```

Note that if you configure Meson manually, you will need to correctly set the `PYTHONPATH` environment variable to be able to import SciPy.
If you configured Meson manually, you can test if the build was successful by trying to import scipy from a non-root directory:

```shell
cd builddir
mkdir test && cd test
python -c 'import scipy; print(scipy.__version__)'
```

This should print a single line with a version like `1.9.0dev...` if the build succeeded.

That's it! You have now built SciPy from source and can start testing it.

**Running Tests**

If you used `dev.py` to build SciPy, run the following command to start testing:

```shell
# change the `--tests` option according to the tests you want to run.
python dev.py --tests scipy/stats  # tests scipy.stats submodule
python dev.py --submodule special  # tests scipy.special submodule
python dev.py  # runs full test suite
```

If you manually configured Meson, import `pytest` in a Python shell from a non-root directory to run tests:

```python
>>> import scipy
>>> scipy.test()  # runs full test suite.
```
