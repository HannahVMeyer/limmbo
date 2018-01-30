from __future__ import absolute_import

__version__ = "0.1.2"

from .testit import test
from . import core
from . import io
from . import utils
from . import plot
from . import bin

__all__ = ['test', 'core', 'io', 'utils', 'plot', 'bin']
