
# Kaspix Pipeline: Neural Circuit Modeling 

![Status](https://img.shields.io/badge/Physics--AI-Pipeline-blueviolet)
![Framework](https://img.shields.io/badge/PyTorch-Dataset-orange)
![Simulation](https://img.shields.io/badge/Engine-NGSpice-lightgrey)

An end-to-end framework for synthetic dataset generation and training of recurrent neural networks (RNNs) designed to emulate analog filter behavior. This system converts physical SPICE topologies into tensors optimized for dynamic conditioning architectures like **FiLM (Feature-wise Linear Modulation)**.

## 🚀 Key Features

* **Multi-Topology Support**: Automated generation for multiple filters (LPF, HPF, BPF, Notch) into a single consolidated dataset.
* **Stochastic Signal Factory**: Recipe-based excitation system (Pink Noise, Chirps, Step Sequences) to capture spectral dynamics and transients.
* **Circuit ID & Metadata**: Each sample includes a unique identifier and parameter mapping to enable multi-task learning.
* **Hardware-Agnostic Processing**: Intelligent handling of *knobs* (parameters) using **Automatic Padding** for circuits of varying complexity.

## 📂 Repository Structure

```text
/
├── circuits/           # Standardized .cir files (NGSpice)
├── datasets/           # Consolidated datasets (.pt)
├── ngspice_dataset.ipynb  # Main simulation and generation pipeline
└── README.md           # Project documentation

```

## 🛠️ Netlist Format (SPICE Templates)

For the `NetlistProcessor` to extract parameters and run stochastic simulations generically, `.cir` files must follow these structural rules:

### 1. Parameter Definition (`.param`)

Components intended to be varied by the generator must be defined using the `.param` directive. The processor uses these lines to identify the control "Knobs".

```spice
.param R_gain=10k  ; Automatically detected as a control variable
.param C_cut=100n   ; Automatically detected as a control variable

```

### 2. Standard I/O Nodes

The circuit must use fixed node names so the audio pipeline knows where to inject and measure the signal automatically:

* **Input:** Voltage source `Vin` connected to the `input` node.
* **Output:** The voltage measurement point must be the `output` node.

### 3. Extraction Directive

The line `.save V(output)` must be included at the end of the file. This ensures NGSpice only exports the necessary data vector, optimizing generation speed and memory usage.

## 📊 Final Dataset Structure

The `kaspix_full_dataset.pt` file is a PyTorch container that unifies multiple topologies into a coherent structure, resolving differences in component counts through metadata and **padding**.

### Content of the `.pt` File

When loading the dataset using `torch.load(..., weights_only=False)`, you will obtain a dictionary containing:

1. **`x` (Inputs):** A list of dictionaries. Each entry contains:
* `audio_in`: Input signal tensor (excitation).
* `knobs`: Normalized parameter vector (0-1 scale) with **Zero-Padding**.
* `circuit_id`: Unique numeric index of the topology (e.g., `0` for LPF, `1` for BPF).
* `netlist_origin`: Source filename for data auditing.


2. **`y` (Outputs/Targets):** A list of tensors containing the real physical response obtained via NGSpice.
3. **`metadata`:** A global dictionary with experiment info:
* `circuit_mapping`: ID-to-Name translation dictionary (e.g., `{0: "low_pass_filter.cir"}`).
* `fs`: Sampling frequency (Sample Rate).
* `n_samples_total`: Total number of generated samples.



### The Concept of "Zero-Padding" in Knobs

Since a High-Pass filter might have 2 parameters and a Notch filter might have 4, the dataset standardizes all vectors to the size of the most complex circuit in the batch.

> **Example:** If the maximum number of parameters is 4, a simple 2-parameter filter is stored as `[val1, val2, 0, 0]`. The neural network uses the `circuit_id` (via Embeddings or FiLM layers) to learn which values in that vector are physically relevant for each specific topology.