#include "benchmarksuite.h"

/* tell the compiler we're using functions from the magmadnn namespace */
using namespace magmadnn;

int main(int argc, char** argv) {

    using T = float;
    bool distributed, user_use_gpu;
    int batch_size, epochs;
    double learning_rate;
    int network = 0; // 0 = DenseNet, 1 = AlexNet, 2 = LeNet

    // Define needed variables
    Tensor<float>* images_host, * labels_host;
    uint32_t n_images, n_rows, n_cols, n_labels, n_features, n_classes = 10;
    uint32_t n_channels = 1;

    // Memory used for training (CPU or GPU)
    memory_t training_memory_type;

    // Random for benchmark_id
    std::srand(std::time(nullptr));
    int benchmark_id = std::rand();

    // Using one node by default (if there is no MPI)
    int nnodes = 1;

    // Write the application title to console
    write_title();

    if (argc != 8) {
        /* ask user for the configuration*/
#if defined(MAGMADNN_HAVE_MPI)
        distributed = user_get_distributed_benchmark();
#else
        distributed = false;
#endif

#if defined(USE_GPU)
        user_use_gpu = user_get_use_gpu();
#else
        user_use_gpu = false;
#endif
        batch_size = user_get_batch_size();
        epochs = user_get_epoches();
        learning_rate = user_get_learning_rate();
        dataset_path = user_get_dataset_path(dataset_name);
        network = user_get_network();
    }  else {
        /* use argv for user configuration */
#if defined(MAGMADNN_HAVE_MPI)
        if (std::experimental::string_view(argv[1]) == "y" || std::experimental::string_view(argv[1]) == "Y") {
            distributed = true;
		}
        else {
            distributed = false;
        }
#else
        distributed = false;
#endif

        if (std::experimental::string_view(argv[2]) == "y" || std::experimental::string_view(argv[2]) == "Y") {
            user_use_gpu = true;
        }
        else {
            user_use_gpu = false;
        }

            batch_size = std::atoi(argv[3]);
            epochs = std::atoi(argv[4]);
            learning_rate = std::atof(argv[5]);
            dataset_path = argv[6];
            network = std::atoi(argv[7]);
    }

    std::cout << "benchmark_id: " << benchmark_id << std::endl;
    std::cout << "hostname: " << get_hostname() << std::endl;
    std::cout << "distributed: " << distributed << std::endl;
    std::cout << "user_use_gpu: " << user_use_gpu << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "epochs: " << epochs << std::endl;
    std::cout << "learning_rate: " << learning_rate << std::endl;
    std::cout << "dataset_name: " << dataset_name << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "network: " << network << " (0=DenseNet,1=AlexNet,2=LeNet)" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

#if defined(MAGMADNN_HAVE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
#else
    std::cout << "Compiled without MPI support!" << std::endl;
#endif

    /* every magmadnn program must begin with magmadnn_init. This allows magmadnn to test the environment
       and initialize some GPU data. */
    std::cout << "Initializing magmaDNN." << std::endl;
    magmadnn_init();
    std::cout << "Initialized magmaDNN." << std::endl;
    
    //set dataset path
    std::string dataset_path_images = dataset_path + "/train-images-idx3-ubyte";
    std::string dataset_path_labels = dataset_path + "/train-labels-idx1-ubyte";

    // Load MNIST trainnig dataset
    std::cout << "Load " << dataset_name << " training dataset" << std::endl;
    images_host = read_mnist_images(dataset_path_images.c_str(), n_images, n_rows, n_cols);
    labels_host = read_mnist_labels(dataset_path_labels.c_str(), n_labels, n_classes);
    std::cout << "Loaded " << dataset_name << " training dataset" << std::endl;

	// preparation for other datasets
    /*
    std::cout << "Load " << dataset_name << " training dataset" << std::endl;
    magmadnn::data::MNIST<T> train_set(dataset_path, magmadnn::data::Train);
    magmadnn::data::MNIST<T> test_set(dataset_path, magmadnn::data::Test);
    std::cout << "Loaded " << dataset_name << " training dataset" << std::endl;
    */
    /*
    // Load MNIST trainnig dataset
    std::cout << "Load MNIST training dataset" << std::endl;
    magmadnn::data::MNIST<T> train_set(mnist_fashion_dir, magmadnn::data::Train);
    magmadnn::data::MNIST<T> test_set(mnist_fashion_dir, magmadnn::data::Test);
    std::cout << "Loaded MNIST trainnig dataset" << std::endl;*/

    /*
    // Load CIFAR-10 trainnig dataset   
    std::cout << "Load CIFAR10 trainig dataset" << std::endl;
    magmadnn::data::CIFAR10<T> train_set(cifar10_dir, magmadnn::data::Train);
    magmadnn::data::CIFAR10<T> test_set(cifar10_dir, magmadnn::data::Test);
    std::cout << "Loaded CIFAR10 training dataset" << std::endl;*/

    /*
    // Load CIFAR-100 trainnig dataset 
    std::cout << "Load CIFAR100 training dataset" << std::endl;
    magmadnn::data::CIFAR100<T> train_set(cifar100_dir, magmadnn::data::Train);
    magmadnn::data::CIFAR100<T> test_set(cifar100_dir, magmadnn::data::Test);
    std::cout << "Loaded CIFAR100 training dataset" << std::endl;*/


    // use gpu or cpu for training
#if defined(USE_GPU)
    
    training_memory_type = (user_use_gpu) ? DEVICE : HOST;
    
    if (user_use_gpu) {
        std::cout << "Training on GPUs" << std::endl;
    }
    else {
        std::cout << "Training on CPUs" << std::endl;
    }
#else
    if (user_use_gpu) {
        std::cout << "Compiled without GPU support!" << std::endl;
    }
    training_memory_type = HOST;
    std::cout << "Training on CPUs" << std::endl;
#endif

    //define the feature amount
    n_features = n_rows * n_cols;

    // Initialize our model parameters
    model::nn_params_t params;
    params.batch_size = batch_size; /* batch size: the number of samples to process in each mini-batch */
    params.n_epochs = epochs; /* # of epochs: the number of passes over the entire training set */
    params.learning_rate = learning_rate;
    params.momentum = 0.90;

    // Creating the neural network
    // This will serve as the input to our network
    auto x_batch = op::var<T>("x_batch", { params.batch_size, n_channels, n_rows, n_cols}, { NONE, {} }, training_memory_type);


    std::vector<layer::Layer<T>*> layers;

    if (network == 0) {
        //DenseNet
        /* initialize the layers in our network */
        auto input = layer::input(x_batch);
        auto fc1 = layer::fullyconnected(input->out(), 784, false);
        auto act1 = layer::activation(fc1->out(), layer::RELU);

        auto fc2 = layer::fullyconnected(act1->out(), 500, false);
        auto act2 = layer::activation(fc2->out(), layer::RELU);

        auto fc3 = layer::fullyconnected(act2->out(), n_classes, false);
        auto act3 = layer::activation(fc3->out(), layer::SOFTMAX);

        auto output = layer::output(act3->out());

        /* wrap each layer in a vector of layers to pass to the model */
        layers = { input, fc1, act1, fc2, act2, fc3, act3, output };
    }
    else if (network == 1) {
        //AlexNet
        /* initialize the layers in the alex network https://learnopencv.com/wp-content/uploads/2018/05/AlexNet-1.png */
        auto input = layer::input<T>(x_batch);

        auto conv2d1 = layer::conv2d<T>(input->out(), { 11, 11 }, 64, { 2, 2 }, { 4, 4 }, { 1, 1 });
        auto act1 = layer::activation<T>(conv2d1->out(), layer::RELU);
        auto pool1 = layer::pooling<T>(act1->out(), { 3, 3 }, { 0, 0 }, { 2, 2 }, AVERAGE_POOL);

        auto conv2d2 = layer::conv2d<T>(pool1->out(), { 5, 5 }, 192, layer::SAME, { 1, 1 }, { 1, 1 });
        auto act2 = layer::activation<T>(conv2d2->out(), layer::RELU);
        auto pool2 = layer::pooling<T>(act2->out(), { 3, 3 }, { 0, 0 }, { 2, 2 }, AVERAGE_POOL);

        auto conv2d3 = layer::conv2d<T>(pool2->out(), { 3, 3 }, 384, layer::SAME, { 1, 1 }, { 1, 1 });
        auto act3 = layer::activation<T>(conv2d3->out(), layer::RELU);

        auto conv2d4 = layer::conv2d<T>(act3->out(), { 3, 3 }, 384, layer::SAME, { 1, 1 }, { 1, 1 });
        auto act4 = layer::activation<T>(conv2d4->out(), layer::RELU);

        auto conv2d5 = layer::conv2d<T>(act4->out(), { 3, 3 }, 256, layer::SAME, { 1, 1 }, { 1, 1 });
        auto act5 = layer::activation<T>(conv2d5->out(), layer::RELU);

        auto pool3 = layer::pooling<T>(act5->out(), { 3, 3 }, layer::SAME, { 2, 2 }, AVERAGE_POOL);

        auto dropout1 = layer::dropout<float>(pool3->out(), 0.5);

        auto flatten = layer::flatten<T>(dropout1->out());

        auto fc1 = layer::fullyconnected<T>(flatten->out(), 4096, true);
        auto act6 = layer::activation<T>(fc1->out(), layer::RELU);

        auto fc2 = layer::fullyconnected<T>(act6->out(), 4096, true);
        auto act7 = layer::activation<T>(fc2->out(), layer::RELU);

        auto fc3 = layer::fullyconnected<T>(act7->out(), n_classes, false);
        auto act8 = layer::activation<T>(fc3->out(), layer::SOFTMAX);

        auto output = layer::output<T>(act8->out());


        /* wrap each layer in a vector of layers to pass to the model */
        layers =
        { input,
         conv2d1, act1, pool1,
         conv2d2, act2, pool2,
         conv2d3, act3,
         conv2d4, act4,
         conv2d5, act5,
         pool3,
         dropout1,
         flatten,
         fc1, act6,
         fc2, act7,
         fc3, act8,
         output };

    }
    else {
        //LeNet
        /* initialize the layers in the network */
        auto input = layer::input(x_batch);

        auto conv2d1 = layer::conv2d(input->out(), { 5, 5 }, 32, { 0, 0 }, { 1, 1 }, { 1, 1 });
        auto act1 = layer::activation(conv2d1->out(), layer::TANH);
        auto pool1 = layer::pooling(act1->out(), { 2, 2 }, { 0, 0 }, { 2, 2 }, AVERAGE_POOL);

        auto conv2d2 = layer::conv2d(pool1->out(), { 5, 5 }, 32, { 0, 0 }, { 1, 1 }, { 1, 1 });
        auto act2 = layer::activation(conv2d2->out(), layer::TANH);
        auto pool2 = layer::pooling(act2->out(), { 2, 2 }, { 0, 0 }, { 2, 2 }, AVERAGE_POOL);

        auto flatten = layer::flatten(pool2->out());

        auto fc1 = layer::fullyconnected(flatten->out(), 120, true);
        auto act3 = layer::activation(fc1->out(), layer::TANH);

        auto fc2 = layer::fullyconnected(act3->out(), 84, true);
        auto act4 = layer::activation(fc2->out(), layer::TANH);

        auto fc3 = layer::fullyconnected(act4->out(), n_classes, false);
        auto act5 = layer::activation(fc3->out(), layer::SOFTMAX);

        auto output = layer::output(act5->out());

        /* wrap each layer in a vector of layers to pass to the model */
        layers =
        { input,
         conv2d1, act1, pool1,
         conv2d2, act2, pool2,
         flatten,
         fc1, act3,
         fc2, act4,
         fc3, act5,
         output };
    }

    // This creates a Model for us. The model can train on our data
    // and perform other typical operations that a ML model can.
    //                                    
    // - layers: the previously created vector of layers containing our
    // network
    // - loss_func: use cross entropy as our loss function
    // - optimizer: use stochastic gradient descent to optimize our
    // network
    // - params: the parameter struct created earlier with our network
    // settings
#if defined(MAGMADNN_HAVE_MPI)
    optimizer::Optimizer<float>* optim = new optimizer::DistMomentumSGD<float>(
         params.learning_rate, params.momentum); 
    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optim, params);
#else
    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, params);
#endif

    // metric_t records the model metrics such as accuracy, loss, and training time
    model::metric_t metrics;

    // Create a std::promise object for stopping the monitor
    std::promise<void> exitSignal;

    //Fetch std::future object associated with promise
    std::future<void> futureObj = exitSignal.get_future();

    // Starting Thread & move the future object in lambda function by reference
    std::thread th(&threadMonitor, std::move(futureObj), benchmark_id, dataset_name, dataset_path, user_use_gpu, distributed, network);

    /* fit will train our network using the given settings.
        X: independent data
        y: ground truth
        metrics: metric struct to store run time metrics
        verbose: whether to print training info during training or not */
    model.fit(images_host, labels_host, metrics, true);

    //sleep for 1 second for initializing the monitor
    std::this_thread::sleep_for(std::chrono::seconds(1));

    //Set the value in promise
    exitSignal.set_value();

    //Wait for thread too end
    std::cout << "Waiting for the monitor to end." << std::endl;
    th.join();

    std::cout << "Accuracity: " << metrics.accuracy << " Loss: " << metrics.loss << " Training Time: " << metrics.training_time << std::endl;

     //calculate the training score
    double score = ((100 * epochs) / metrics.training_time);
    std::cout << "Training Score: " << std::to_string(score) << " (100*epochs/training_time) with " << metrics.accuracy << " accuracity." << std::endl;

    //write result to data file
    write_results(benchmark_id, std::to_string(metrics.accuracy), std::to_string(metrics.loss), std::to_string(metrics.training_time), std::to_string(score), dataset_name, dataset_path, user_use_gpu, distributed, network);

    // Clean up memory after training
    delete layers[layers.size() - 1]; //delete the last (output) layer

    //delete
    delete images_host;
    delete labels_host;

    // Every magmadnn program should call magmadnn_finalize before
    magmadnn_finalize();

#if defined(MAGMADNN_HAVE_MPI)
    //finilize MPI
    MPI_Finalize();
#endif

    return 0;
}

void write_title() {
    std::string const title = "\n==== MagmaDNN Benchmarksuit ====\n";
    std::cout << title;
}

#pragma region User_Settings

std::string user_get_dataset_path(std::string name) {
    std::string dataset_path;
    std::cout << "Enter the full path to a " + name + " compatible dataset (Enter D for Default: /home/lucasc/lab/datasets/" + name + ").\n";

    getline(std::cin, dataset_path);

    if (dataset_path == "D" || dataset_path == "d" || dataset_path.empty() || std::cin.fail()) {
        std::cin.clear();
        return "/home/lucasc/lab/datasets/" + name;
    }
    else {
        return dataset_path;
    }
}

bool user_get_use_gpu() {
    char input;

    std::cout << "Do you want to use GPUs for training? [y/n]:\n";
    std::cin >> input;

    if (input == 'y' || input == 'Y') {
        return true;
    }
    else {
        return false;
    }
}

bool user_get_distributed_benchmark() {
    char input;

    std::cout << "Do you want to run a distributed benchmark? [y/n]:\n";
    std::cin >> input;

    if (input == 'y' || input == 'Y') {
        return true;
    }
    else {
        return false;
    }
}

int user_get_batch_size() {
    int input;

    std::cout << "Define the batch size for the training as Integer (Enter D for Default: 128).\n";
    std::cin >> input;

    /* handle bad input */
    if (std::cin.fail()) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Using default batch size of 128.\n";
        return 128;
    }
    else {
        return input;
    }
}
int user_get_epoches() {
    int input;

    std::cout << "Define the number of epoches to train as Integer (Enter D for Default: 10).\n";
    std::cin >> input;

    /* handle bad input */
    if (std::cin.fail()) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Using default epoches of 10.\n";
        return 10;
    }
    else {
        return input;
    }
}
int user_get_network() {
    int input;

    std::cout << "Define the network you want to use for the benchmark (0=DenseNet, 1=AlexNet, 2=LeNet).\n";
    std::cin >> input;

    /* handle bad input */
    if (std::cin.fail()) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Using default network DenseNet.\n";
        return 0;
    }
    else {
        return input;
    }
}

double user_get_learning_rate() {
    double input;
    std::cout << "Define the Learning Rate as Double (Enter D for Default: 0.05).\n";
    std::cin >> input;

    /* handle bad input */
    if (std::cin.fail()) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Using default Learning Rate of 0.05.\n";
        return 0.05;
    }
    else {
        return input;
    }
}
#pragma endregion User_Settings

#pragma region Monitor
std::tuple<std::string, std::string, std::string> get_cpu_info() {
    std::string cpu_cores = exec("cat /proc/cpuinfo | grep 'cpu cores' | uniq | awk '{print $4}' | tr -d '\n'");
    std::string cpu_threads = exec("cat /proc/cpuinfo | grep processor | wc -l | tr -d '\n'");
    std::string cpu_all_core_clock = exec("cat /proc/cpuinfo | grep 'cpu MHz' | awk '{print $4}' | tr -d '\n'");
    std::string cpu_core0_clock = exec("cat /proc/cpuinfo | grep 'cpu MHz' | awk 'NR==3{print $4}' | tr -d '\n'");
    std::string cpu_usage = exec("top -bn1 | grep 'Cpu(s)' | awk '{print $2+$4+$6}' | tr -d '\n'");

    return { cpu_cores, cpu_core0_clock, cpu_usage };
}

std::tuple<std::string, std::string, std::string, std::string> get_memory_info() {
    std::string memory_total = exec("free -m | grep Mem | awk '{print $2}' | tr -d '\n'");
    std::string memory_used = exec("free -m | grep Mem | awk '{print $3}' | tr -d '\n'");
    std::string swap_total = exec("free -m | grep Swap | awk '{print $2}' | tr -d '\n'");
    std::string swap_used = exec("free -m | grep Swap | awk '{print $3}' | tr -d '\n'");

    return { memory_total, memory_used, swap_total, swap_used };
}

std::tuple<std::string, std::string, std::string> get_gpu_info() {
    std::string gpu_temp = exec("nvidia-smi -q | grep 'GPU Current Temp' | awk '{print $5}' | tr -d '\n'");
    std::string gpu_utilization = exec("nvidia-smi -q | grep Gpu | awk '{print $3}' | tr -d '\n'");
    std::string memory_utilization = exec("nvidia-smi -q | grep 'Memory' | awk 'NR==3{print $3}' | tr -d '\n'");

    return { gpu_temp, gpu_utilization, memory_utilization };
}

std::string get_gpu_name() {
    std::string gpu_name = exec("nvidia-smi -q | grep 'Product Name' | tr -d '\n'");
    return gpu_name;
}

std::string get_hostname() {
    std::string hostname = exec("hostname | tr -d '\n'");
    return hostname;
}

std::string get_cpu_name() {
    std::string cpu_name = exec("cat /proc/cpuinfo | grep 'model name' | uniq | tr -d '\n'");
    return cpu_name;
}

// took from https://www.tutorialspoint.com/How-to-execute-a-command-and-get-the-output-of-command-within-Cplusplus-using-POSIX
std::string exec(std::string command) {
    char buffer[128];
    std::string result = "";

    // Open pipe to file
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return "popen failed!";
    }

    // read till end of process:
    while (!feof(pipe)) {

        // use buffer to read and add to result
        if (fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }

    pclose(pipe);
    return result;
}

void threadMonitor (std::future<void> futureObj, int benchmark_id, std::string dataset_name, std::string dataset_path, bool user_use_gpu, bool distributed, int network)
{
    std::string hostname = get_hostname();
    std::string cpu_name = get_cpu_name();
    std::string gpu_name = get_gpu_name();
    int time = 0;

    std::cout << "Monitor Started" << std::endl;
    std::string file_name = std::to_string(benchmark_id) + "-" + hostname + ".dat";

    //open the filestream
    std::ofstream out(file_name.c_str());

    //write the header data
    out << "#Hostname:" << hostname << std::endl;
    out << "#CPU: " << cpu_name << std::endl;
    out << "#GPU: " << gpu_name << std::endl;
    out << "#Dataset Name: " << dataset_name << std::endl;
    out << "#Dataset Path: " << dataset_path << std::endl;
    out << "#Use GPU: " << user_use_gpu << std::endl;
    out << "#Distributed: " << distributed << std::endl;
    out << "#Network: " << network << " (0=DenseNet,1=AlexNet,2=LeNet)" << std::endl;

    //write header
    out << "time;gpu_temp;gpu_utilization;gpu_memory_utilization;cpu_cores;cpu_core0_clock;cpu_utilization;memory_utilization;memory_total;memory_used;swap_utilization;swap_total;swap_used" << std::endl;

    while (futureObj.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout)
    {
        auto start = std::chrono::high_resolution_clock::now();

        //get current system stats
        //gpu_temp, gpu_utilization, gpu_memory_utilization
        std::tuple<std::string, std::string, std::string> gpuInfo;

        if (user_use_gpu) {
            gpuInfo = get_gpu_info();
        }
        else {
            gpuInfo = std::make_tuple("-", "-", "-");
        }

        //cpu_cores, cpu_core0_clock, cpu_utilization
        std::tuple<std::string, std::string, std::string> cpuInfo = get_cpu_info();

        //memory_total, memory_used, swap_total, swap_used
        std::tuple<std::string, std::string, std::string, std::string> memoryInfo = get_memory_info();

        //get the memory utilization in percent
        float memory_utilization = (std::stod(std::get<1>(memoryInfo)) / std::stod(std::get<0>(memoryInfo))) * 100;
        float swap_utilization;

        //get the swap utilization in percent
        if (std::stod(std::get<3>(memoryInfo)) == 0) {
            swap_utilization = 0;
        } else {
            swap_utilization = (std::stod(std::get<3>(memoryInfo)) / std::stod(std::get<2>(memoryInfo))) * 100;
        }
        
        //write stats to file
        out << time << ';' << std::get<0>(gpuInfo) << ';' << std::get<1>(gpuInfo) << ';' << std::get<2>(gpuInfo);
        out << ';' << std::get<0>(cpuInfo) << ';' << std::get<1>(cpuInfo) << ';' << std::get<2>(cpuInfo);
        out << ';' << memory_utilization << ';' << std::get<0>(memoryInfo) << ';' << std::get<1>(memoryInfo) << ';' << swap_utilization << ';' << std::get<2>(memoryInfo) << ';' << std::get<3>(memoryInfo) << std::endl;

        //increment time
        time = time + 1;

        //sleep for 1s
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::this_thread::sleep_for(std::chrono::milliseconds(950 - duration.count()));
    }

    //close the filestream
    out.close();

    std::cout << "Benchmark Data saved to " << file_name << std::endl;
}

void write_results(int benchmark_id, std::string accuracy, std::string loss, std::string training_time, std::string score, std::string dataset_name, std::string dataset_path, bool user_use_gpu, bool distributed, int network) {
    std::string hostname = get_hostname();
    std::string cpu_name = get_cpu_name();
    std::string gpu_name = get_gpu_name();

    std::string file_name = std::to_string(benchmark_id) + "-" + hostname + "_result.dat";

    //open the filestream
    std::ofstream out(file_name.c_str());

    //write the header data
    out << "#Hostname:" << hostname << std::endl;
    out << "#CPU: " << cpu_name << std::endl;
    out << "#GPU: " << gpu_name << std::endl;
    out << "#Dataset Name: " << dataset_name << std::endl;
    out << "#Dataset Path: " << dataset_path << std::endl;
    out << "#Use GPU: " << user_use_gpu << std::endl;
    out << "#Distributed: " << distributed << std::endl;
    out << "#Network: " << network << " (0 = DenseNet, 1 = AlexNet, 2 = LeNet)" << std::endl;

    //write the result data
    out << "benchmark_id;accuracy;loss;training_time;score" << std::endl;
    out << std::to_string(benchmark_id) << ";" << accuracy << ";" << loss << ";" << training_time << ";" << score << std::endl;

    //close the filestream
    out.close();

    std::cout << "Results saved to " << file_name << std::endl;
}
#pragma endregion Monitor

#pragma region Read_MNIST_Distributed

#define FREAD_CHECK(res, nmemb)           \
    if ((res) != (nmemb)) {               \
        fprintf(stderr, "fread fail.\n"); \
        return NULL;                      \
    }

inline void endian_swap(uint32_t& val) {
    /* taken from https://stackoverflow.com/questions/13001183/how-to-read-little-endian-integers-from-file-in-c */
    val = (val >> 24) | ((val << 8) & 0xff0000) | ((val >> 8) & 0xff00) | (val << 24);
}

Tensor<float>* read_mnist_images(const char* file_name, uint32_t& n_images, uint32_t& n_rows, uint32_t& n_cols) {
    FILE* file;
    unsigned char magic[4];
    Tensor<float>* data;
    uint8_t val;

    file = std::fopen(file_name, "r");

    if (file == NULL) {
        std::fprintf(stderr, "Could not open %s for reading.\n", file_name);
        return NULL;
    }

    FREAD_CHECK(fread(magic, sizeof(char), 4, file), 4);
    if (magic[2] != 0x08 || magic[3] != 0x03) {
        std::fprintf(stderr, "Bad file magic.\n");
        return NULL;
    }

    FREAD_CHECK(fread(&n_images, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_images);

    FREAD_CHECK(fread(&n_rows, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_rows);

    FREAD_CHECK(fread(&n_cols, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_cols);

#if defined(MAGMADNN_HAVE_MPI)
    int rank, nnodes;
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint32_t end = n_images;
    n_images = n_images / nnodes;
    uint32_t start = rank * n_images;
    if (start + n_images < end)
        end = start + n_images;
    n_images = end - start;
    printf("MPI Node %3d: Preparing to read %5d images [%5d .. %5d] with size %u x %u ...\n",
        rank, n_images, start, end, n_rows, n_cols);
    fseek(file, sizeof(char) * start * n_rows * n_cols, SEEK_CUR);
#else
    printf("Preparing to read %u images with size %u x %u ...\n", n_images, n_rows, n_cols);
#endif

    char bytes[n_rows * n_cols];

    /* allocate tensor */
    data = new Tensor<float>({ n_images, n_rows, n_cols }, { NONE, {} }, HOST);

    for (uint32_t i = 0; i < n_images; i++) {
        FREAD_CHECK(fread(bytes, sizeof(char), n_rows * n_cols, file), n_rows * n_cols);

        for (uint32_t r = 0; r < n_rows; r++) {
            for (uint32_t c = 0; c < n_cols; c++) {
                val = bytes[r * n_cols + c];

                data->set(i * n_rows * n_cols + r * n_cols + c, (val / 128.0f) - 1.0f);
            }
        }
    }
    printf("Finished reading images.\n");

    fclose(file);

    return data;
}

Tensor<float>* read_mnist_labels(const char* file_name, uint32_t& n_labels, uint32_t n_classes) {
    FILE* file;
    unsigned char magic[4];
    Tensor<float>* labels;
    uint8_t val;

    file = std::fopen(file_name, "r");

    if (file == NULL) {
        std::fprintf(stderr, "Could not open %s for reading.\n", file_name);
        return NULL;
    }

    FREAD_CHECK(fread(magic, sizeof(char), 4, file), 4);

    if (magic[2] != 0x08 || magic[3] != 0x01) {
        std::fprintf(stderr, "Bad file magic.\n");
        return NULL;
    }

    FREAD_CHECK(fread(&n_labels, sizeof(uint32_t), 1, file), 1);
    endian_swap(n_labels);

#if defined(MAGMADNN_HAVE_MPI)
    int rank, nnodes;
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint32_t end = n_labels;
    n_labels = n_labels / nnodes;
    uint32_t start = rank * n_labels;
    if (start + n_labels < end)
        end = start + n_labels;
    n_labels = end - start;
    printf("MPI Node %3d: Preparing to read %5d labels [%5d .. %5d] with %u classes ...\n",
        rank, n_labels, start, end, n_classes);
    fseek(file, sizeof(char) * start, SEEK_CUR);
#else
    printf("Preparing to read %u labels with %u classes ...\n", n_labels, n_classes);
#endif

    /* allocate tensor */
    labels = new Tensor<float>({ n_labels, n_classes }, { ZERO, {} }, HOST);

    for (unsigned int i = 0; i < n_labels; i++) {
        FREAD_CHECK(fread(&val, sizeof(char), 1, file), 1);

        labels->set(i * n_classes + val, 1.0f);
    }

    printf("Finished reading labels.\n");
    fclose(file);

    return labels;
}

#pragma endregion Read_MNIST_Distributed

