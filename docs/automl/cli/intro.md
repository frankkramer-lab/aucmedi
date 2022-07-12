For AutoML, AUCMEDI offers a Command Line Interface (CLI), which allows command line script interaction in a local environment. Recommended for research settings.

## Command Line Interface

The AutoML interface CLI allows quickly running AUCMEDI in any IT environment ranging from local laptops up to cluster infrastructure.

!!! cite "Wikipedia defines Command-line interface"
    A command-line interpreter or command-line processor uses a command-line interface (CLI) to receive commands from a user in the form of lines of text. This provides a means of setting parameters for the environment, invoking executables and providing information to them as to what actions they are to perform.

    Today, many users rely upon graphical user interfaces and menu-driven interactions. However, some programming and maintenance tasks may not have a graphical user interface and use a command line.

    [Wikipedia - Source](https://en.wikipedia.org/wiki/Command-line_interface)

### Installation

The AUCMEDI CLI is supported for console environments which can run Python: Linux, MacOS and Windows.
Core support and unittesting is done on Ubuntu (Linux) with shell & bash.

For installation, there are no specific CLI dependencies required.

For AUCMEDI, the AutoML CLI is recommended to be installed via PyPI (`pip install aucmedi`).
As dependencies, it is required to install GPU drivers (CUDA) for running Tensorflow and
Python.

Further resources:

- [TensorFlow 2 - Installation](https://www.tensorflow.org/install)
- [Python](https://www.python.org/)

!!! cite "TensorFlow FAQ: GPU Support"
    How do I install the NVIDIA driver?

    The recommended way is to use your package manager and install the cuda-drivers package (or equivalent).
    When no packages are available, you should use an official "runfile".

    Alternatively, the NVIDIA driver can be deployed through a container.
    Refer to the documentation for more information.

    [TensorFlow - Source](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
