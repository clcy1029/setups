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

- Install / Update dependency list for the global env
    ```
    uv pip install -r requirements.txt
    ```

- (Optional) Setup Github access
    1. https://github.com/settings/keys add ssh key to github 
        ```
        check if ssh key already exists
        ls -al ~/.ssh 

        if yes open the 
        id_ed25519.pub or id_rsa.pub file, copy the token and save it to https://github.com/settings/keys

        if not exist
        use below command to generate one
        ssh-keygen -t ed25519 -C "<your email registered in Github.com>" then do the copy

        Done! you local machine now has the permission to push/pull from Github

        (Note) Remote repo has to use git link
        like "git@github.com:clcy1029/setups.git" instead of "https://github.com/clcy1029/setups.git"
        ```
    2. set github global name
    ```
    git config --global user.name "你想用的名字"
    git config --global user.email "你的GitHub邮箱"
    ```


---

## Useful Dependency Explaination
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