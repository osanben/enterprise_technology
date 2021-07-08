## docker install

### Uninstall old versions

Older versions of Docker were called `docker`, `docker.io`, or `docker-engine`. If these are installed, uninstall them:

```
sudo apt-get remove docker docker-engine docker.io containerd runc
```

### Install using the repository

Before you install Docker Engine for the first time on a new host machine, you need to set up the Docker repository. Afterward, you can install and update Docker from the repository.

#### Set up the repository

1. Update the `apt` package index and install packages to allow `apt` to use a repository over HTTPS:

```
sudo apt-get update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

2. Add Docker’s official GPG key:

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

3. Use the following command to set up the **stable** repository. To add the **nightly** or **test** repository, add the word `nightly` or `test` (or both) after the word `stable` in the commands below. 

```
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

#### Install Docker Engine

1. Update the `apt` package index, and install the *latest version* of Docker Engine and containerd, or go to the next step to install a specific version:

```
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io
```

2. Verify that Docker Engine is installed correctly by running the `hello-world` image.

```
sudo docker run hello-world
```

## nvidia-docker install

### Setting up Docker

Docker-CE on Ubuntu can be setup using Docker’s official convenience script:

```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

### Setting up NVIDIA Container Toolkit

Setup the `stable` repository and the GPG key:

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

Install the `nvidia-docker2` package (and dependencies) after updating the package listing:

```
sudo apt-get update

sudo apt-get install -y nvidia-docker2
```

Restart the Docker daemon to complete the installation after setting the default runtime:

```
sudo systemctl restart docker
```

At this point, a working setup can be tested by running a base CUDA container:

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## docker command

### HOW TO CREATE A DOCKER IMAGE FROM A CONTAINER

The Docker create command will create a new container for us from the command line:

```
sudo docker create --name nginx_base -p 80:80 nginx:alpine
```

Inspect Images

```
sudo docker images -a
```

Inspect Containers

```
sudo docker ps -a
```

We will use the docker cp command to copy this file onto the running container

```
sudo docker cp index.html nginx_base:/usr/share/nginx/html/index.html
```

Create an Image From a Container

```
sudo docker commit nginx_base hi_mom_nginx
```

## python install

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```

Change the Python3 default version in Ubuntu

```py
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

sudo update-alternatives --config python
```

## python3-pip install

```
sudo apt install python3-pip
echo "alias pip=pip3" >> ~/.bash_aliases
source ~/.bash_aliases
alias pip='python3.6 -m pip'
pip install --upgrade pip
```

