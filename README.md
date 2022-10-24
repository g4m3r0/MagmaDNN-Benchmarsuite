# MagmaDNN Benchmarksuite

Due to the continuous development of hardware as well as the availability of large-scale training datasets the use of machine learning using Deep Neural Networks (DNN) has increased significantly.
This has had a major impact on a variety of scientific research areas, such as object recognition and classification, speech recognition as well as speech recognition, and speech synthesis.

The scalability of training DNNs with classical machine learning frameworks poses one big hurdle- it relies on homogeneous computing systems to train efficiently. 
The MagmaDNN framework for data analytics and machine learning uses scalable linear algebra routines of the MAGMA library which applies solutions from parallel distributed computing on current and future heterogeneous computing architectures. 

In this research practicum, a benchmark suite based on the currently available version 1.2 of MagmaDNN was developed to evaluate the performance of individual compute nodes as well as clusters of heterogeneous computing nodes of the Chair of Practical Computer Science at the Chemnitz University of Technology with and without computational accelerators (GPUs), in the domain of DNN.


## Install Dependecies

```
cd /home/user/path
wget http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.6.0.tar.gz
tar -xvpf magma-2.6.0.tar.gz
cd magma-2.6.0/
mkdir build
cd build/
cmake
-DCMAKE_INSTALL_PREFIX:PATH=/home/user/path/deps/ 
..
make -j16
make install

cd /home/user/path
wget https://bitbucket.org/icl/magmadnn/get/release-magmadnn-v1.2.tar.gz
mv icl-magmadnn-20820ee43a0e magmadnn
cd magmadnn
#cp ./make.inc-examples/make.inc-standard ./make.inc
echo "prefix = /home/user/path/deps" > ./make.inc
echo "TRY_CUDA = 1" >> ./make.inc
echo "CXX = g++-8" >> ./make.inc
echo "NVCC = nvcc" >> ./make.inc
echo "GPU_TARGET = Kepler Pascal Maxwell" >> ./make.inc
echo "CUDADIR ?= /opt/packages/cudnn-10.2-linux-x64-v7.6.5.32" >> 
./make.inc
echo "MAGMADIR ?= /home/user/path/deps" >> 
./make.inc
echo "BLASLIB ?= openblas" >> ./make.inc
cat ./make.inc-examples/make.inc-standard >> ./make.inc
make -j16
make install

cd /home/user/path
wget
https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.bz2
tar -xvjpf openmpi-4.1.1.tar.bz2
cd openmpi-4.1.1/
./configure --prefix=/home/user/path/deps
--with-cuda=/usr/include/
make -j16
make install

cd /home/user/path Code
export
LD_LIBRARY_PATH=/opt/packages/cudnn-10.2-linux-x64-v7.6.5.32/lib64:/home/user/path/deps/lib:$LD_LIBRARY_PATH
g++ -O3 -DMAGMADNN_HAVE_CUDA -DUSE_GPU -DMAGMADNN_HAVE_MPI -o
benchmarksuite benchmarksuite.cpp
-I/home/user/path/deps/include
-I/opt/packages/cudnn-10.2-linux-x64-v7.6.5.32/include
-L/home/user/path/deps/lib
-L/opt/packages/cudnn-10.2-linux-x64-v7.6.5.32/lib64 -lopenblas -lcudart
-lcudnn -lmagma -lmagmadnn -lmpi_cxx -lmpi -lpthread
```

## Compile Benchmarksuite with CUDA Support
### 0. Change the path variables
Inside the benchmarksuite.h file, the dataset_name and dataset_path variables need to be changed according to your directory structure.

```
/* CONFIGURE DATASET-PATH START */

//Human readable name for the dataset that will be displayed in the log files
std::string dataset_name = "mnist";

//Full path to the MNIST based dataset
std::string dataset_path = "/home/lucasc/lab/datasets/" + dataset_name;

/* CONFIGURE DATASET-PATHS END */
```

### 1. Add libs to path variable
```
export LD_LIBRARY_PATH=/opt/packages/cudnn-10.2-linux-x64-v7.6.5.32/lib64:/home/lucasc/magma/lib:/home/lucasc/openblas/lib:/home/lucasc/magmadnn/lib:/home/lucasc/openmpi-4.1.0/lib:/usr/lib:$LD_LIBRARY_PATH
```

### 2. Compile from source
```
g++ -O3 -DMAGMADNN_HAVE_CUDA -DUSE_GPU -DMAGMADNN_HAVE_MPI -o benchmarksuite benchmarksuite.cpp -I/home/lucasc/openblas/include -I/home/lucasc/magma/include -I/home/lucasc/magmadnn/include -I/opt/packages/cudnn-10.2-linux-x64-v7.6.5.32/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/home/lucasc/magma/lib -L/home/lucasc/openblas/lib -L/home/lucasc/magmadnn/lib -L/opt/packages/cudnn-10.2-linux-x64-v7.6.5.32/lib64 -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lopenblas -lcudart -lcudnn -lmagma -lmagmadnn -lmpi_cxx -lmpi -lpthread
```

## Compile Benchmarksuite without CUDA Support
### 0. Change the path variables
Inside the benchmarksuite.h file, the dataset_name and dataset_path variables need to be changed according to your directory structure.

```
/* CONFIGURE DATASET-PATH START */

//Human readable name for the dataset that will be displayed in the log files
std::string dataset_name = "mnist";

//Full path to the MNIST based dataset
std::string dataset_path = "/home/lucasc/lab/datasets/" + dataset_name;

/* CONFIGURE DATASET-PATHS END */
```

### 1. Add libs to path variable
```
export LD_LIBRARY_PATH=/home/lucasc/magma/lib:/home/lucasc/openblas/lib:/home/lucasc/magmadnn/lib:/home/lucasc/openmpi-4.1.0/lib:/usr/lib:$LD_LIBRARY_PATH
```

### 2. Compile from source
```
g++ -O3 -DMAGMADNN_HAVE_MPI -o benchmarksuite_nocuda benchmarksuite.cpp -I/home/lucasc/openblas/include -I/home/lucasc/magma/include -I/home/lucasc/magmadnn_nocuda/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/home/lucasc/magma/lib -L/home/lucasc/openblas/lib -L/home/lucasc/magmadnn_nocuda/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lopenblas -lmagma -lmagmadnn -lmpi_cxx -lmpi -lpthread
```

## Run Benchmarksuite without MPI
```
./benchmarksuite <use mpi y/n> <use gpu y/n> <batch size e.g. 128> <number of epochs e.g. 10> <learning rate e.g. 0.05> <path to dataset e.g. /home/lucasc/lab/datasets/mnist> <network (0 = DenseNet, 1 = AlexNet, 2 = LeNet)>
```
```
e.g. ./benchmarksuite n y 128 10 0.05 /home/lucasc/lab/datasets/mnist 0
```

```
<use MPI y/n> - Definiert, ob es sich um einen verteilten Benchmark handelt
<use GPU y/n> - Definiert, ob die GPU für den Benchmark genutzt werden soll
<batch size e.g. 128> - Definiert, die größe jedes Batchs
<number of epochs e.g. 10> - Definiert, wie viele Epochen trainiert werden soll
<learning rate e.g. 0.05> - Definiert die Lernrate
<path to dataset e.g. /home/lucasc/lab/datasets/mnist> - Definiert der Pfad des Datensatzes
<network (0 = DenseNet, 1 = AlexNet, 2 = LeNet)> - Definiert das zu verwendende Neuronale Netzwerk
```

## Run Benchmarksuite with MPI and CUDA
```
mpiexec -x LD_LIBRARY_PATH=/opt/packages/cudnn-10.2-linux-x64-v7.6.5.32/lib64:/home/lucasc/magma/lib:/home/lucasc/openblas/lib:/home/lucasc/magmadnn/lib:/home/lucasc/openmpi-4.1.0/lib:/usr/lib -n 2 -H matisse3,matisse4 --mca plm_base_verbose 1 /home/lucasc/lab/benchmarksuite_nocuda y y 128 100 0.05 /home/lucasc/lab/datasets/mnist 0
```

## Run Benchmarksuite with MPI and without CUDA
```
mpiexec -x LD_LIBRARY_PATH=/home/lucasc/magma/lib:/home/lucasc/openblas/lib:/home/lucasc/magmadnn_nocuda/lib -n 4 -H localhost,haswell2,skylake1,skylake2 --mca plm_base_verbose 1 /home/lucasc/lab/benchmarksuite_nocuda y n 128 100 0.05 /home/lucasc/lab/datasets/mnist 0
```

## Plot a Gaph with Benchmarkdata
```
FILE="<Name der Benchmark-Datei>.dat";
PLOTTITLE=$(grep -i -e header -e footer $(echo $FILE) | sed s/\"\//g | sed s/\#//g);
gnuplot -e "titel='${PLOTTITLE}'; filename='${FILE}';" <Name des gnuplot Skripts>.gnuplot
```

### Dataset Download

MNIST can be downloaded from http://yann.lecun.com/exdb/mnist/  
MNIST Fashion can be downloaded from https://github.com/zalandoresearch/fashion-mnist
