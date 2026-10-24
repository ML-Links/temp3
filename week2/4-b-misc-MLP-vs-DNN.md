# Difference Between Multilayer Perceptron (MLP) and Deep Neural Networks (DNN)

**MLP describes an architecture; deep describes depth.** A deep MLP is both an MLP and a DNN. These are overlapping descriptions, rather than competing algorithms.

## 1. Basic Definition

**Multilayer Perceptron (MLP):** An MLP is a type of feedforward artificial neural network consisting of an input layer, one or more hidden layers, and an output layer. It uses nonlinear activations in its hidden layers and typically uses fully connected (dense) layers, in which each neuron connects to all neurons in the next layer.

**Deep Neural Network (DNN):** A DNN is a neural network with multiple hidden or processing layers. It can use different architectures, including fully connected networks, Convolutional Neural Networks (CNNs), and Transformers.

**Key difference:** MLP describes a particular network architecture, while DNN describes a broader category of networks characterized by their depth. An MLP with several hidden layers is also a DNN.

## 2. Network Architecture

- **MLP:** Primarily uses fully connected layers, with information passing from input to output.
- **DNN:** May use fully connected, convolutional, recurrent, attention-based, or other layers, depending on the architecture.

## 3. Types of Data

- **MLP:** Commonly used with structured data and numerical feature vectors. It can also process other types of data after suitable representation or preprocessing.
- **DNN:** Can process structured and unstructured data, including images, text, audio, and time-series data. The appropriate architecture depends on the task.

## 4. Feature Learning

- **MLP:** Learns relationships and useful feature representations from input data. Additional feature engineering can be helpful depending on the application.
- **DNN:** Also learns feature representations automatically. Specialized architectures can make better use of spatial, sequential, or other relationships in the data.

## 5. Number of Layers

- **MLP:** May contain a small or large number of hidden layers. A sufficiently deep MLP is a type of DNN.
- **DNN:** Contains multiple processing layers, which can support hierarchical representations; additional depth does not guarantee better accuracy. There is no universally fixed minimum number of layers that defines a network as deep.

## 6. Computational Requirements

- **MLP:** Smaller MLPs often require relatively modest computational resources. Large or deep MLPs can also be computationally expensive.
- **DNN:** Computational requirements vary with the network's size, architecture, dataset, and training method. Large models often benefit from GPUs or other accelerators.

## Summary Comparison

| Feature | MLP | DNN |
|---|---|---|
| Definition | Specific neural network architecture | Broad category of neural networks |
| Structure | Mainly fully connected layers | Various layer types and architectures |
| Network depth | Can be shallow or deep | Multiple processing layers |
| Data | Often used with feature vectors and tabular data | Can handle structured and unstructured data |
| Feature learning | Learns features automatically | Learns features automatically |
| Computation | Depends on network size | Depends on network size and architecture |
| Relationship | A deep MLP is a DNN | Includes deep MLPs and other deep architectures |

## Conclusion

An MLP is a particular type of neural network, whereas a DNN is a broader category of neural networks with multiple processing layers.

An MLP can be shallow or deep. When an MLP has sufficient depth, it is also considered a deep neural network.

**Deep Learning (DL)** refers to the broader approach of using deep neural networks to learn representations and patterns from data. It therefore includes deep MLPs, CNNs, Transformers, and other deep architectures.


## Two architectures to compare

The input dimension is shown first; it is not a trainable Dense layer. Count hidden layers separately from the input and output.

| Architecture | Hidden layers | Description | Parameters including biases |
|---|---:|---|---:|
| `2 → 4 → 1` | 1 | Shallow MLP | 17 |
| `2 → 4 → 4 → 4 → 1` | 3 | Deep MLP; also a DNN | 57 |

These examples use nonlinear hidden activations and a sigmoid output. They illustrate shallow and deep usage without defining a universal depth cutoff. A deep CNN or Transformer is another example of a DNN, with a different architecture.

**Question:** Why are nonlinear hidden activations needed? What happens if every hidden layer uses only an affine transformation?

<details>
<summary>Show answer</summary>

Composing affine transformations gives another affine transformation. Without nonlinear hidden activations, these extra layers do not add nonlinear representation power. With a sigmoid output, the result is still a sigmoid applied to an affine score.

</details>

For a Dense layer with $d$ inputs and $h$ outputs, there are $dh+h$ parameters including biases. The first model has $2(4)+4+4(1)+1=17$; the second adds two layers with $4(4)+4=20$ parameters each, giving $57$.

For a different topic, see [SGD and mini-batches](4-c-misc-SGD.md): architecture determines the computation, while batching determines which observations contribute to each update.
