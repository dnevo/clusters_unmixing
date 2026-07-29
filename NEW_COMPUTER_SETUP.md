# New Computer Setup Guide

This guide documents how to recreate a working VS Code development environment for the `clusters_unmixing` project on a fresh Windows computer.

It covers:

- installing the base tools
- cloning the repository
- recreating the Python environment
- installing the VS Code extensions used by this project
- applying the workspace settings that make imports, debugging, and notebooks work
- verifying the setup end to end

The current project remote is:

```text
https://github.com/dnevo/clusters_unmixing.git
```

The original layout used with this project was:

```text
H:\repos\clusters_unmixing      # project code
H:\envs\research_base           # Python virtual environment
```

You do not have to keep those exact paths, but doing so makes the instructions below match the repository defaults.

## Core Principle

Move the source code with Git. Recreate the Python environment locally.

Do not copy the old virtual environment folder from another computer:

```text
H:\envs\research_base
```

That folder contains machine-specific interpreter paths and should be rebuilt from scratch.

## Before Moving The Old Machine

### 1. Check The Working Tree

Open PowerShell in the project folder and inspect the repo state:

```powershell
cd H:\repos\clusters_unmixing
git status --short --branch
```

Review any modified or untracked files before you move machines. In general:

- commit source, notebook, config, and documentation changes
- do not commit temporary files such as logs, caches, or generated outputs unless you intentionally want them versioned

### 2. Push Everything You Want To Keep

Use VS Code Source Control or PowerShell.

Example:

```powershell
cd H:\repos\clusters_unmixing
git add README.md pyproject.toml main.py src notebooks experiments data .vscode NEW_COMPUTER_SETUP.md
git commit -m "Prepare setup guide for a new machine"
git push origin main
```

If you only want to save selected files, stage them individually instead of using a broad add.

After pushing, verify that the branch is clean:

```powershell
git status --short --branch
```

### 3. Back Up Anything Not In Git

GitHub only contains tracked files. Back up anything local and important that is not committed, such as:

- `.env` files
- secrets or API keys
- private data
- generated outputs you want to keep
- personal notes

Generated experiment outputs are stored under `experiments/outputs/` and are ignored by Git.

### 4. Turn On VS Code Settings Sync

VS Code settings and extensions normally live on the computer, not in the repo. Settings Sync is the easiest way to move them.

On the old computer in VS Code:

1. Click the Accounts icon in the lower-left corner.
2. Select `Turn on Settings Sync`.
3. Sign in with GitHub or Microsoft.
4. Sync at least:
   - Settings
   - Extensions
   - Keybindings
   - Snippets
   - UI State

If you prefer a backup, export the extension list too:

```powershell
code --list-extensions > vscode-extensions.txt
```

## Install The New Computer

### 1. Install Base Software

Install these first:

- Git for Windows
- Visual Studio Code
- Python
- Optional: GitHub Desktop, if you prefer a GUI for Git

The project requires Python `>=3.10`.

The previous environment was created with Python 3.14, but any Python version supported by your packages is acceptable. If a package install fails, try Python 3.12 or 3.13.

Check which Python versions are installed:

```powershell
py -0p
```

### 2. Optional: Restore Remote-SSH Settings

This project does not require SSH by default, but your old VS Code user settings include custom Remote-SSH entries. If you still use remote machines, re-add those user-level settings after installing VS Code.

The non-default entries I found were:

- `remote.SSH.remotePlatform` for `dgx03` and `dsigpu01`
- `remote.SSH.serverInstallPath` for `dgx03`

These are stored in VS Code user settings, not in the repo, and there is no checked-in `.ssh/config` or `cluster_unmixing.code-workspace` SSH block in this project.

### 3. Install VS Code

Download and install VS Code from Microsoft, then launch it once so it can initialize its user profile.

After first launch, sign in to the account you use for Settings Sync.

### 4. Install The Extensions This Repo Uses

If Settings Sync does not restore them automatically, install these extensions in VS Code:

- `ms-python.python`
- `ms-python.vscode-pylance`
- `ms-python.debugpy`
- `ms-toolsai.jupyter`
- `ms-python.vscode-python-envs`
- `github.copilot-chat`
- `github.vscode-pull-request-github`

Optional, depending on your workflow:

- `openai.chatgpt`
- `google.geminicodeassist`
- `google.colab`

You can also export and reinstall extensions from the command line if `code` is available:

```powershell
code --list-extensions
```

The project most strongly depends on Python, Pylance, Debugpy, and Jupyter.

### 5. Sign In To Cloud-Backed Extensions

After the extensions are installed, sign in again if needed:

- GitHub Copilot
- GitHub Pull Requests
- OpenAI / ChatGPT / Codex extensions
- Google Gemini Code Assist, if you use it

Do not copy authentication files or tokens from the old computer.

## Clone The Repository

If you want to keep the original drive layout:

```powershell
mkdir H:\repos
cd H:\repos
git clone https://github.com/dnevo/clusters_unmixing.git
cd clusters_unmixing
```

If you do not have an `H:` drive, clone somewhere else, such as `C:\repos`.

The repo does not depend on the absolute path `H:\repos`, but the workspace settings in `.vscode/settings.json` do assume the original virtual environment path unless you change it.

## Create The Python Environment

### Option A: Recreate The Original Layout

Create the environment outside the repo, matching the old machine:

```powershell
mkdir H:\envs
py -m venv H:\envs\research_base
H:\envs\research_base\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

### Option B: Keep The Environment Inside The Repo

If you prefer a project-local environment:

```powershell
cd H:\repos\clusters_unmixing
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you use a different environment path than the one in the repo defaults, update the workspace settings file after creating it.

## Install Project Dependencies

From the project root with the virtual environment activated:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the project in editable mode using `pyproject.toml`.

For notebook support, also install the notebook dependencies:

```powershell
python -m pip install ipython plotly ipykernel
```

If you plan to use the optional W&B logging path, install the extra too:

```powershell
python -m pip install -e .[wandb]
```

If you need the optional Mamba model, install its extra separately:

```powershell
python -m pip install -e .[mamba]
```

## Apply The Workspace Configuration

This repository already includes VS Code workspace settings in `.vscode/settings.json` and `.vscode/launch.json`.

There is currently no checked-in `cluster_unmixing.code-workspace` file. For this project, open the repo folder directly in VS Code and let `.vscode/settings.json` and `.vscode/launch.json` provide the workspace behavior.

The important settings are:

- `python.defaultInterpreterPath`: points VS Code at the project environment
- `python.terminal.activateEnvironment`: makes VS Code activate the environment in terminals
- `python.analysis.extraPaths`: adds `src` so imports resolve correctly in the editor
- `PYTHONPATH` in the debug config: ensures debug sessions also see `src`

Current workspace settings expect this interpreter path:

```text
H:\envs\research_base\Scripts\python.exe
```

If you create the environment somewhere else, edit `.vscode/settings.json` and change `python.defaultInterpreterPath` to the new interpreter path.

The debug configuration in `.vscode/launch.json` already runs the current file from the workspace root with `src` on `PYTHONPATH`.

## Open The Project In VS Code

Open the project with `cluster_unmixing.code-workspace` using `File -> Open Workspace from File`. This is the required entry point for the project because it preserves the workspace-level configuration you used before.

If you do not have the workspace file on the new computer, copy it from your old setup or recreate it before continuing.

```powershell
code cluster_unmixing.code-workspace
```

If the `code` command is unavailable, open VS Code and choose the workspace file manually:

```text
File -> Open Workspace from File -> cluster_unmixing.code-workspace
```

Then confirm the interpreter:

1. Press `Ctrl+Shift+P`.
2. Run `Python: Select Interpreter`.
3. Choose the environment you created.

If you used the original layout, select:

```text
H:\envs\research_base\Scripts\python.exe
```

If you used a local environment, select:

```text
H:\repos\clusters_unmixing\.venv\Scripts\python.exe
```

## Configure The Notebook Kernel

Open `notebooks/experiment_review.ipynb` in VS Code.

In the top-right kernel picker, choose the same environment you selected for Python.

If the environment does not appear, register it manually:

```powershell
python -m ipykernel install --user --name research_base --display-name "Python (research_base)"
```

If you use a different environment name, use that instead of `research_base`.

Reload VS Code after registering the kernel.

## Verify The Setup

From the project root with the environment activated, run:

```powershell
python -m pip check
python -c "import clusters_unmixing; print(clusters_unmixing.__file__)"
python main.py
```

Then verify the notebook helper flow:

```powershell
python notebooks\_notebook_smoke.py
```

Expected outputs are written under:

```text
experiments/outputs/
```

That folder is ignored by Git and should be created automatically when you run experiments.

## Daily Workflow On The New Computer

Before you start work, pull the latest changes:

```powershell
git pull origin main
```

After making changes:

```powershell
git status --short
git add <files-you-want-to-save>
git commit -m "Describe the change"
git push origin main
```

In VS Code Source Control, the same flow is:

1. Review the changed files.
2. Stage the files you want.
3. Commit with a clear message.
4. Push or sync the branch.

## Troubleshooting

### Imports Are Underlined In VS Code

Check the selected interpreter:

```text
Ctrl+Shift+P -> Python: Select Interpreter
```

Choose the environment you created. If the warning remains, reload the window or restart the Python language server.

### `clusters_unmixing` Cannot Be Imported

Run this from the project root:

```powershell
python -m pip install -e .
```

Then restart the VS Code Python language server.

### Notebook Kernel Is Missing

Install and register `ipykernel`:

```powershell
python -m pip install ipykernel
python -m ipykernel install --user --name research_base --display-name "Python (research_base)"
```

Reload VS Code and reselect the kernel.

### Torch Installation Fails

Use a Python version supported by the installed PyTorch build. If Python 3.14 causes trouble, recreate the environment with another Python version:

```powershell
py -3.12 -m venv H:\envs\research_base
H:\envs\research_base\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install ipython plotly ipykernel
```

### Git Push Or Pull Fails

Make sure you are signed in to GitHub in VS Code or Git Credential Manager.

Check the remote:

```powershell
git remote -v
```

Expected remote:

```text
origin  https://github.com/dnevo/clusters_unmixing.git
```

## Quick Checklist

Old computer:

- Commit the real changes.
- Push to GitHub.
- Back up ignored outputs or private files if needed.
- Turn on VS Code Settings Sync.
- Confirm GitHub has the latest project state.

New computer:

- Install Git, VS Code, and Python.
- Sign in and enable VS Code Settings Sync.
- Install the Python, Pylance, Debugpy, and Jupyter extensions.
- Clone `https://github.com/dnevo/clusters_unmixing.git`.
- Create a new virtual environment.
- Run `python -m pip install -e .`.
- Install notebook dependencies with `python -m pip install ipython plotly ipykernel`.
- Select the interpreter in VS Code.
- Open the notebook and select the same kernel.
- Sign in again to GitHub, Copilot, and any other cloud-backed extensions.