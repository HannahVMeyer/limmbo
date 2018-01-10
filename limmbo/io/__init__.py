r"""
Data input
**********************

Parse command-line arguments
^^^^^^^^^^^^^
.. autoclass:: limmbo.io.parser.ParseData
  :members:

Read data
^^^^^^^^^^^^^^^^^^^
.. autoclass:: limmbo.io.reader.ReadData
  :members:

Check data
^^^^^^^^^^^^^^^^^^^
.. autoclass:: limmbo.io.input.InputData
  :members:
"""

import parser
import reader
import input

__all__ = ['parser', 'reader', 'input']
