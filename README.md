# Kaspix Pipeline: Neural Circuit Modeling (Omni-Pipeline)

![Status](https://img.shields.io/badge/Physics--AI-Pipeline-blueviolet)
![Framework](https://img.shields.io/badge/PyTorch-Dataset-orange)
![Simulation](https://img.shields.io/badge/Engine-NGSpice-lightgrey)
![Environment](https://img.shields.io/badge/Runtime-Google%20Colab-yellow)

An end-to-end framework for synthetic dataset generation and training of recurrent neural networks (RNNs/LSTMs) designed to emulate analog filter behavior. This system converts physical SPICE topologies—including active and passive circuits—into tensors optimized for dynamic conditioning architectures like **FiLM (Feature-wise Linear Modulation)**.

## 🚀 Key Features

* **Active & Passive Support**: Automated generation for 1st-order passive networks and 2nd-order active topologies (Sallen-Key, Multiple Feedback, Twin-T).
* **Omni-Pipeline V4.2**: Refactored simulation worker supporting dynamic attribute injection for Resistors, Capacitors, Inductors, and VCVS (OpAmps).
* **Circuit ID & Metadata**: Each sample includes a unique categorical identifier to enable multi-task learning and topology switching.
* **Zero-Padding Strategy**: Intelligent standardization of control *knobs* to maintain a fixed tensor dimension (Size 5) across circuits of varying complexity.

## 📂 Repository Structure

```text
/
├── activecircuits/     # Active filter netlists (Sallen-Key, MFB, Notch)
├── circuits/           # Passive filter netlists (LPF, HPF, BPF, etc.)
├── ngspice_dataset.ipynb   # Dataset generation notebook
├── ngspice_dataset_activefilters.ipynb # Active filter dataset generation
├── benchmark_&_training.ipynb  # Model training and benchmarking
├── benchmark_&_training_noise.ipynb  # Training with noisy parameters
└── README.md           # Project documentation
```

## 🛠️ Setup & Execution (Google Colab)
Due to the specific requirements of libngspice shared libraries and Linux binary links, the recommended environment is Google Colab.

Environment: Use a T4 GPU or TPU runtime for faster dataset generation.

Automated Installation: The first cell of the notebooks performs the necessary setup:

- Installs ngspice and libngspice0 via apt-get.

- Creates the symbolic link: libngspice.so.0 -> libngspice.so.

- Installs PySpice and torch.

**Note**: Local Windows installation is currently not supported due to complex .dll path mapping and NGSpice environment variables.

## 🧪 Netlist Format (SPICE Templates)
For the generator to extract parameters generically, .cir files must follow these rules:

1. Parameter Definition (.param)
Control variables must use the .param directive. These are detected as "Knobs".

    ```
    .param R_tune=10k  ; Detected as a frequency control knob
    .param C_base=10n   ; Detected as a secondary hardware parameter
    ```

2. Standard I/O Nodes
    
    - Input: Voltage source Vin connected to the input node.

    - Output: Measurement point must be the output node.

    - Ground: Standard 0 node.

## 📊 Dataset Structure
The kaspix_full_dataset.pt is a PyTorch container that unifies multiple topologies into a coherent structure.

Tensor Unpacking Protocol
The data loader yields a 4-value tuple for each batch:

1. Audio Input: Normalized excitation signal (Chirps, Noise, etc.).

2. Knobs: Normalized parameter vector (0-1 scale) padded to size 5.

3. Circuit IDs: Categorical index (e.g., 0: LPF, 1: HPF, 4: MFB_BPF).

4. Target: Real physical response obtained via NGSpice simulation.

**Technical Note**: Zero-Padding
Since an LPF might have 3 knobs and a Notch filter 5, the pipeline standardizes all vectors. The neural network uses the Circuit ID (via Embeddings or FiLM layers) to learn which values in that vector are physically relevant for each specific topology.