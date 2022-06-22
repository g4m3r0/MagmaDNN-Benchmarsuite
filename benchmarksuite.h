#include <cstdint>
#include <cstdio>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <future>
#include <assert.h>
#include <chrono>
#include <experimental/string_view>

/* CONFIGURE DATASET-PATH START */

//Human readable name for the dataset that will be displayed in the log files
std::string dataset_name = "mnist";

//Full path to the MNIST based dataset
std::string dataset_path = "/home/lucasc/lab/datasets/" + dataset_name;

/* CONFIGURE DATASET-PATHS END */

/* include mpi if compiled with MAGMADNN_HAVE_MPI */
#if defined(MAGMADNN_HAVE_MPI)
#include "mpi.h"
#include "optimizer/distributed_momentum_sgd.h"
#endif

/* we must include magmadnn */
#include "magmadnn.h"

/* tell the compiler we're using functions from the magmadnn namespace */
using namespace magmadnn;

/* these are used for reading in the MNIST data set -- found at http://yann.lecun.com/exdb/mnist/ */
Tensor<float>* read_mnist_images(const char* file_name, uint32_t& n_images, uint32_t& n_rows, uint32_t& n_cols);
Tensor<float>* read_mnist_labels(const char* file_name, uint32_t& n_labels, uint32_t n_classes);

//write the application title to console
void write_title();

//get user input
std::string user_get_dataset_path(std::string name);
bool user_get_distributed_benchmark();
bool user_get_use_gpu();
int user_get_batch_size();
int user_get_epoches();
int user_get_network();
double user_get_learning_rate();

//get system informations
std::tuple<std::string, std::string, std::string> get_cpu_info();
std::tuple<std::string, std::string, std::string> get_gpu_info();
std::tuple<std::string, std::string, std::string, std::string> get_memory_info();
std::string get_gpu_name();
std::string get_hostname();
std::string get_cpu_name();

//system monitor
void threadMonitor(std::future<void> futureObj, int benchmark_id, std::string dataset_name, std::string dataset_path, bool user_use_gpu, bool distributed, int network);

//write result file
void write_results(int benchmark_id, std::string accuracy, std::string loss, std::string training_time, std::string score, std::string dataset_name, std::string dataset_path, bool user_use_gpu, bool distributed, int network);

//Execute Commands on Shell
std::string exec(std::string command);
