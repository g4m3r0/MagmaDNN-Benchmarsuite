# MagmaDNN Benchmarksuite

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
