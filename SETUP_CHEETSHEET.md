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
    source ~/.venvs/dev-env/bin/activate
    ```
    How to deactivate the dev-env:
    ```
    deactivate
    ```
    How to delete this env
    ```
    rm -rf ~/.venvs/dev-env/
    ```

- Install / Update dependency list for the global env
    ```
    uv pip install -r requirements.txt
    ```

- Setup Zshell config on MACOS on terminal launch with "source ~/.venvs/dev-env/bin/activate" auto enabled
    ```
    add below line to ~/.zshrc

    if [ -z "$VIRTUAL_ENV" ] && [ -f "$HOME/.venvs/dev-env/bin/activate" ]; then
        source "$HOME/.venvs/dev-env/bin/activate"
    fi
    
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
    git config --global user.name "your name"
    git config --global user.email "your github email"
    ```

## VSCODE IDE Setup

1. Force Python coding to be BLACK FORMATTED (What is black format? [Black formatter](https://www.geeksforgeeks.org/python/python-code-formatting-using-black/))

View -> Command Palette -> Preferences: Open User Settings (JSON)
Then in the json config copy and paste and insert below to the dictionary.
```
"[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
    },
```


2. Force VSCODE to use 4 space as a tab. Similarily update above user setting json

Final json setting:
```
{
    "workbench.colorTheme": "Visual Studio Light",
    "explorer.confirmDelete": false,
    "explorer.confirmDragAndDrop": false,
    "editor.fontSize": 14,
    "workbench.settings.applyToAllProfiles": [
        "editor.fontSize",
        "editor.tabSize",
        "editor.accessibilityPageSize"
    ],
    "editor.tabSize": 4,
    "editor.insertSpaces": true,  
    "editor.detectIndentation": false, 
    "editor.accessibilityPageSize": 12,
    "vs-kubernetes": {
        "vs-kubernetes.minikube-show-information-expiration": "2024-10-31T02:14:55.674Z",
        "vscode-kubernetes.helm-path-mac": "/Users/chang/.vs-kubernetes/tools/helm/darwin-amd64/helm"
    },
    "explorer.confirmPasteNative": false,
    "dataWrangler.experiments.fastCsvParsing": true,
    "git.openRepositoryInParentFolders": "never",
    "docker.extension.enableComposeLanguageServer": false,
    "security.workspace.trust.untrustedFiles": "open",
    "editor.minimap.enabled": false,
    "git.confirmSync": false,
    "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
    },
    
}
```

3. VSCODE Python debug setting

    step 1: Add a launch json configuration:
    Command Palette -> Debug:Add a configuration
    Then a new launch.json will be created under .vscode/launch.json (You can also manual create it cuz VSCODE will honor the filename)

    step 2: Copy and Paste
    ```
    {
          // Use IntelliSense to learn about possible attributes.
          // Hover to view descriptions of existing attributes.
          // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
          "version": "0.2.0",
          "configurations": [
    
                {
                      "name": "Python Debugger: Current File Without Parameters",
                      "type": "debugpy",
                      "request": "launch",
                      "program": "${file}",
                      "console": "integratedTerminal",
                      "python":"${userHome}/.venvs/dev-env/bin/python"
                }
          ]
    }
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
