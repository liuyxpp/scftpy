scftpy
======

**scftpy** is a python package for performing polymer self-consistent field theory calculations. 

Quickstart
----------

1. Install
^^^^^^^^^^

::

    $ easy_install scftpy

or

::

    $ tar -xvf scftpy-xxx.tar.gz
    $ cd scftpy-xxx
    $ python setup.py install

Required packages:

* `numpy`: chebpy heavily depends on numpy.
* `scipy`: advanced algorithms, such as scipy.fftpack.dct.
* `chebpy`: Chebyshev collocation methods for PDEs.

2. Quick Start
^^^^^^^^^^^^^^

Here is an example of carrying out 1D unitcell calculations of A-B diblock copolymers.

::

    $ from scftpy import scft
    $ scft.run()

Note: you should modify the configuration file (scft.cfg) for different systems. 
A sample cofiguration file can be found in the package source root directory.

Ask for Help
------------

* You can directly contact me at liuyxpp@gmail.com.
* You can join the mailinglist by sending an email to scftpy@librelist.com 
  and replying to the confirmation mail. 
  To unsubscribe, send a mail to scftpy-unsubscribe@librelist.com 
  and reply to the confirmation mail.

Links
-----

* `Documentation <http://pypi.python.org/pypi/scftpy>`_
* `Website <http://ngpy.org>`_
* `Development version <http://bitbucket.org/liuyxpp/scftpy/>`_

