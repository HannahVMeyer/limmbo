[![PyPI version](https://badge.fury.io/py/limmbo.svg)](https://badge.fury.io/py/limmbo)
# LiMMBo

LiMMBo enables multivariate analysis of high-dimensional phenotypes based on
linear mixed models with bootstrapping (LiMMBo). LiMMBo is available as an open
source Python module. It builds on and can be used in combination with
[LIMIX](https://github.com/limix/limix), a flexible and efficient linear mixed
model library with interfaces to Python.

A description of the public interface can be found [here
](https://limmbo.readthedocs.io/en/latest/index.html).

## Install

LiMMBo is available on the [Python Package Index](https://pypi.python.org).
LiMMBo is dependent on Limix [LIMIX](https://github.com/limix/limix), however
the latest Limix release does not include multi-variate models (see [issue #7](https://github.com/HannahVMeyer/limmbo/issues/7)).
While waiting for the updated Limix release please install as follows:

Install LIMIX with multi-variate support (v1.0.18):
```bash
pip3 install "limix<2"
```

After successful installation of LIMIX, simply install LiMMBo via
```bash
pip3 install limmbo
```
Recently, an R wrapper package for limmbo was created - this independent project
can be found [here](https://github.com/fboehm/limmbo2).

## Problems

If you encounter any issue, please, [submit them
](https://github.com/HannahVMeyer/limmbo/issues).


## License

This project is licensed under the Apache License (Version 2.0, January 2004) -
see the [LICENSE](LICENSE) file for details
