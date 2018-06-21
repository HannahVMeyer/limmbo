r"""
Parse command-line arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: limmbo.io.parser.getGWASargs

A detailed list of the command-line options for `getGWASargs` can be found in
:ref:`runAssociation`.

.. autofunction:: limmbo.io.parser.getVarianceEstimationArgs

A detailed list of the command-line options for `getVarianceEstimationArgs` can
be found in :ref:`runVarianceEstimation`.

Read data
^^^^^^^^^
.. autoclass:: limmbo.io.reader.ReadData
  :members:

Check data
^^^^^^^^^^
.. autoclass:: limmbo.io.input.InputData
  :members:
"""



from . import parser
from . import reader
from . import input

__all__ = ['parser', 'reader', 'input']
