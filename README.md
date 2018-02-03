# LiMMBo

LiMMBo enables multivariate analysis of high-dimensional phenotypes based on
linear mixed models with bootstrapping (LiMMBo). LiMMBo is available as an open
source Python module. It builds on and can be used in combination with
[Limix](https://github.com/limix/limix), a flexible and efficient linear mixed
model library with interfaces to Python.

A description of the public interface can be found [here
](https://www.ebi.ac.uk/~hannah/limmbo/index.html)

## Install

LiMMBo is currently available on the [Python Package
Index](https://pypi.python.org) and will in the future be available through
(conda-forge[(https://conda-forge.org/#about]. The latter platform provides the
recommended installation of [Limix](https://github.com/limix/limix), which
LiMMBo heavily relies on.

The recommended way of installing both packages is the following:
Install LIMIX via [conda](http://conda.pydata.org/docs/index.html)
```bash
conda install -c conda-forge limix
```

After successful installation of LIMIX, simply install LiMMBo via
```bash
pip install limmbo
```

## Problems

If you encounter any issue, please, [submit them
](https://github.com/HannahVMeyer/limmbo/issues).


## License

This project is licensed under the Apache License (Version 2.0, January 2004) -
see the [LICENSE](LICENSE) file for details
