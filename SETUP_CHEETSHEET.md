# Setup Cheatsheet

A quick reference for setup steps, dependencies, and useful commands.

## Table of Contents
- [General Setup](#general-setup)
- [Dependencies](#dependencies)
- [Useful Commands](#useful-commands)

---

## General Setup

- Install Python 3.11 + 
    ```
    brew install python@3.11
    ```

    make sure 3.11 is installed, below command should output python 3.11 installed path
    ```
    which python3.11
    ```

- Install uv
    ```
    brew install uv
    ```


- Setup global dev environment
    ```
    uv venv --python=python3.11 ~/.venvs/dev-env
    ```
    How to activate the dev-env:
    ```
    source .venvs/dev-env/bin/activate
    ```

- Setup Dependency list for the global env


---

## Useful Dependency List
| Tool/Library              | Install Command                                 | Notes                                 |
|---------------------------|-------------------------------------------------|---------------------------------------|
| PyTorch, TorchVision, Torchaudio | `pip install torch torchvision torchaudio` | Core deep learning stack               |
| Transformers              | `pip install transformers`                      | Hugging Face models & tokenizers      |
| Datasets                  | `pip install datasets`                          | Hugging Face dataset management       |
| Accelerate                | `pip install accelerate`                        | Multi-GPU training utilities          |
| Evaluate                  | `pip install evaluate`                          | Model evaluation tools                |
| PEFT                      | `pip install peft`                              | Parameter-efficient fine-tuning       |
| BitsAndBytes              | `pip install bitsandbytes`                      | 8-bit optimizers for LLMs             |
| TRL (Transformers RL)     | `pip install trl`                               | RLHF for LLMs                         |
| Gradio                    | `pip install gradio`                            | Web UI for demos                      |
| FastAPI                   | `pip install fastapi`                           | API server for model inference        |
| vLLM                      | `pip install vllm`                              | Fast LLM inference engine             |
| mlx                       | `pip install mlx`                               | MLX core library for Apple Silicon    |
| mlx-lm                    | `pip install mlx-lm`                            | MLX LLM inference and utilities       |
| mlx-examples              | `pip install mlx-examples`                      | Example models and scripts for MLX    |
| mlx-ops                   | `pip install mlx-ops`                           | Additional MLX operators              |
| text-generation-inference | `pip install text-generation-inference`          | Hugging Face inference server         |


---

## Useful Commands

- Start development server:
    ```bash
    npm run dev
    ```
- Run tests:
    ```bash
    npm test
    ```
- Build project:
    ```bash
    npm run build
    ```

---

_Add more sections as needed for your specific setups._