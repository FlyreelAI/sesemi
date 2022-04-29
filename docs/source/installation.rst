------------
Installation
------------

There are a couple ways to install and use SESEMI. The easiest and recommended way is to use a local
virtual environment that supports pip. However, in cases where you need specific libraries that cannot
be installed directly by you on the host, we also describe how you can use docker as well.

===
Pip
===

To use pip follow these instructions:

1. Configure a virtual environment of choice with at least Python 3.6 (e.g. `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_).
2. Install the Python requirements using either of the following methods:
   
   * As the master branch is generally expected to be stable, you can use it to get the newest features prior to an official release:
   
     .. code-block:: bash
          
        $ pip install git+https://github.com/FlyreelAI/sesemi.git
   
   * If instead there is a specific branch you wish to install:
   
     .. code-block:: bash
          
        $ pip install git+https://github.com/FlyreelAI/sesemi/archive/branch.zip
      
   * You can also install an officially released version from PyPI, though it may be slightly behind the master branch:
      
     .. code-block:: bash
          
        $ pip install sesemi
   
3. Check that SESEMI has been installed correctly by invoking a built-in command:
   
   .. code-block:: bash

        $ open_sesemi -h

======
Docker
======

To use docker instead, follow these instructions:

1. Ensure you have docker installed as described by the instructions `here <https://docs.docker.com/get-docker/>`_.
2. Build the docker image from the remote repository as follows:
   
   .. code-block:: bash
   
       $ USER_ID=$(id -u) SESEMI_TAG=sesemi:latest
       $ DOCKER_BUILDKIT=1 docker build \
           --build-arg USER_ID=${USER_ID} \
           -t ${SESEMI_TAG} https://github.com/FlyreelAI/sesemi.git
   
   Note that your OS user ID is obtained through the bash command `id -u`. Furthermore, this command will create an image
   with the tag `sesemi:latest`.
   
   You can build a different branch or tag by appending `\\#branch` to the end of the
   URL (the backslash escapes the pound sign in bash). A local path to the repository can also be substituted in as well.
3. Check that the image has been built correctly by invoking the following command:
   
   .. code-block:: bash

        $ docker run --rm $SESEMI_TAG open_sesemi -h