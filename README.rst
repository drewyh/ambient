ambient
=======

|travis| |codecov| |readthedocs| |license| |black|

*ambient* is a building physics and simulation Python library.

The aims of *ambient* are to be:

#. Simple: Provide intuitive defaults to get started quickly.
#. Modular: Simulate only the parts you are interested in.
#. Extensible: Enable users to create and modify their own components.
#. Open: Provide an open-source building simulation library.

Features
--------

*ambient* currently offers modules for the analysis of:

- Layered constructions (including steady-state and dynamic response)
- Solar conditions (including design day calculations)
- Psychrometrics of moist air

Simulations can be read from, and written to, JSON format.

Installation
------------

Install *ambient* in a virtual environment using:

.. code-block:: python

   pip install ambient

License
-------

*ambient* is licensed under the terms of the Apache License, Version 2.0.
Refer to the `LICENSE <https://github.com/drewyh/ambient/blob/master/LICENSE>`__
file for details.

.. |travis| image:: https://travis-ci.com/drewyh/ambient.svg?branch=master
             :target: https://travis-ci.com/drewyh/ambient

.. |codecov| image:: https://codecov.io/gh/drewyh/ambient/branch/master/graph/badge.svg
              :target: https://codecov.io/gh/drewyh/ambient

.. |readthedocs| image:: https://readthedocs.org/projects/ambient/badge/?version=latest
                  :target: https://ambient.readthedocs.io/en/latest/?badge=latest
                  :alt: Documentation Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
            :target: https://github.com/psf/black

.. |license| image:: https://img.shields.io/pypi/l/ambient
              :alt: PyPI - License
