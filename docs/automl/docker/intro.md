For AutoML, AUCMEDI offers a Docker Image, which allows command line script interaction in a secure and isolated environment. Recommended for clinical settings.

## Docker

The AutoML interface Docker allows securely running AUCMEDI in a professional IT environment, without risking protection of data privacy or dependency conflicts.

!!! cite "Wikipedia defines Docker"
    Docker implements a high-level API to provide lightweight containers that run processes in isolation. Docker containers are standard processes, so it is possible to use kernel features to monitor their execution—including for example the use of tools like strace to observe and intercede with system calls.

    Docker can package an application and its dependencies in a virtual container that can run on any Linux, Windows, or macOS computer. This enables the application to run in a variety of locations, such as on-premises, in public or private cloud.

    [Wikipedia - Source](https://en.wikipedia.org/wiki/Docker_(software))

    Platform as a service (PaaS) or application platform as a service (aPaaS) or platform-based service is a category of cloud computing services that allows customers to provision, instantiate, run, and manage a modular bundle comprising a computing platform and one or more applications, without the complexity of building and maintaining the infrastructure typically associated with developing and launching the application(s); and to allow developers to create, develop, and package such software bundles.

    [Wikipedia - Source](https://en.wikipedia.org/wiki/Platform_as_a_service)

### Installation

For running AUCMEDI Docker is required to install the Docker engine, first.

More information on How-to-install Docker can be found here:  
[https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

!!! cite "Docker Engine"
    Docker Engine is an open source containerization technology for building and containerizing your applications. Docker Engine acts as a client-server application with:

    - A server with a long-running daemon process dockerd.
    - APIs which specify interfaces that programs can use to talk to and instruct the Docker daemon.
    - A command line interface (CLI) client docker.

    The Docker Engine is available on a variety of Linux platforms, macOS and Windows 10 through Docker Desktop, and as a static binary installation.

    [Docker - Source](https://docs.docker.com/)

## Docker Image: AUCMEDI

**Image Structure:**

The building structure of the AUCMEDI Docker Image is quite simple and can be summarized as following:

- Base Image: TensorFlow environment with GPU support (CUDA)
- Install required dependencies for AUCMEDI (drivers for cv2)
- Install AUCMEDI
- Prepare AUCMEDI dataset-volume interface
- Run AUCMEDI with user-provided arguments

**Base Image:**  

The AUCMEDI Docker Image is based on the latest official Docker Image TensorFlow for GPU support (tensorflow/tensorflow:latest-gpu).

The DockerHub page for the TensorFlow Image provides in detail various information which consequently also applies to the AUCMEDI Docker Image:  
[https://hub.docker.com/r/tensorflow/tensorflow/](https://hub.docker.com/r/tensorflow/tensorflow/)

Further information can also be found in the TensorFlow documentation:  
[https://www.tensorflow.org/install/docker/](https://www.tensorflow.org/install/docker/)

**Image Hosting on GitHub Container Registry:**  

The AUCMEDI Docker Image is automatically built for each release in the GitHub repository and published in the GitHub Container Registry.

The GitHub Container Registry offers unlimited push and pull actions for Docker Images as well as allows hosting the AUCMEDI Docker Image in the same infrastructure as its source code repository.

In order to pull and run the AUCMEDI Docker Image, GitHub Container Registry provides the following Image namespace:
```bash
docker pull ghcr.io/OWNER/IMAGE_NAME
```

For AUCMEDI, which is hosted under the organization `frankkramer-lab` in the repository `aucmedi`, this results
into the following Docker Image:

```bash
# Pull the image from the Container Registry
docker pull ghcr.io/frankkramer-lab/aucmedi

# Run a container call
docker run \
  -v /home/dominik/aucmedi.data:/data \
  --rm \
  ghcr.io/frankkramer-lab/aucmedi \
  training \
  --architecture "DenseNet121"
```

Further information can be found in the GitHub documentation:  
[https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)

**GPU Support:**

!!! cite "TensorFlow FAQ: GPU Support"
    How do I install the NVIDIA driver?

    The recommended way is to use your package manager and install the cuda-drivers package (or equivalent).
    When no packages are available, you should use an official "runfile".

    Alternatively, the NVIDIA driver can be deployed through a container.
    Refer to the documentation for more information.

    [TensorFlow - Source](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
