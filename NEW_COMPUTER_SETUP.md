# New Computer Setup Guide

This guide explains how to move the `clusters_unmixing` project to a new Windows computer and get it working in VS Code with Python, GitHub, notebooks, and coding extensions.

The current project GitHub remote is:

```text
https://github.com/dnevo/clusters_unmixing.git
```

The current local layout is:

```text
H:\repos\clusters_unmixing      # project code
H:\envs\research_base           # Python virtual environment
```

You do not need the exact same paths on the new computer, but using the same layout makes the move simpler.

## Important Idea

Move the project with GitHub. Recreate the Python environment on the new computer.

Do not copy the old virtual environment folder:

```text
H:\envs\research_base
```

That folder contains machine-specific Python paths and installed packages. It should be recreated.

## Before Leaving The Old Computer

### 1. Check The Git State

Open PowerShell in the project folder:

```powershell
cd H:\repos\clusters_unmixing
git status --short --branch
```

At the time this guide was created, there were local changes:

```text
modified: notebooks/00_clusters_unmixing_experiments.ipynb
modified: src/clusters_unmixing/config/schema.py
untracked: debug.log
```

Review these before moving. Usually:

- Commit real code, notebook, config, and documentation changes.
- Do not commit temporary files such as `debug.log`, unless you intentionally need them.

### 2. Commit And Push The Project

You can use VS Code Source Control, or use PowerShell.

PowerShell example:

```powershell
cd H:\repos\clusters_unmixing
git status --short
git add README.md pyproject.toml requirements.txt main.py src notebooks experiments data .vscode NEW_COMPUTER_SETUP.md
git status --short
git commit -m "Prepare project for new computer setup"
git push origin main
```

If you only want to commit selected files, stage them one by one instead of using the broad `git add` command.

After pushing, verify that the working branch is clean:

```powershell
git status --short --branch
```

You want to see no modified project files left, except files you intentionally chose not to move.

### 3. Check Files That Are Not On GitHub

GitHub only contains tracked files. Some local files may not be tracked.

Check untracked and ignored files:

```powershell
git status --short --ignored
```

In this repo, the following are important:

```text
data/6clusters_digitized.csv
data/6clusters_thomas.csv
experiments/configs/configuration.yaml
```

These are tracked by Git and should come from GitHub.

Generated experiment outputs are ignored:

```text
experiments/outputs/
```

If you need old generated outputs on the new computer, copy `experiments/outputs/` separately, for example to an external drive or cloud storage.

Also manually back up anything local that should not be committed:

- secrets
- API keys
- `.env` files
- private data
- large generated outputs
- local notes

Do not commit secrets to GitHub.

### 4. Turn On VS Code Settings Sync

VS Code settings and extensions are usually stored per computer, not inside the repo.

On the old computer:

1. Open VS Code.
2. Click the Accounts icon in the lower-left corner.
3. Choose `Turn on Settings Sync`.
4. Sign in with GitHub or Microsoft.
5. Sync at least:
   - Settings
   - Extensions
   - Keybindings
   - Snippets
   - UI State

This is the easiest way to move your VS Code preferences.

The project currently tracks:

```text
.vscode/launch.json
```

That means debug configuration moves with the repo. Most other VS Code settings depend on Settings Sync unless you explicitly save workspace settings in `.vscode/settings.json`.

### 5. Optional: Export VS Code Extensions

Settings Sync is preferred, but an extension list can be useful as a backup.

In a normal terminal where the `code` command works:

```powershell
code --list-extensions > vscode-extensions.txt
```

Current relevant extensions included:

```text
github.copilot-chat
github.vscode-pull-request-github
google.colab
google.geminicodeassist
ms-python.debugpy
ms-python.python
ms-python.vscode-pylance
ms-python.vscode-python-envs
ms-toolsai.jupyter
ms-toolsai.jupyter-keymap
ms-toolsai.jupyter-renderers
ms-toolsai.vscode-jupyter-cell-tags
ms-toolsai.vscode-jupyter-slideshow
openai.chatgpt
```

You do not need all of these for the project. The most important ones are:

```text
ms-python.python
ms-python.vscode-pylance
ms-python.debugpy
ms-toolsai.jupyter
openai.chatgpt
github.copilot-chat
github.vscode-pull-request-github
```

## Set Up The New Computer

### 1. Install Required Programs

Install:

- Git for Windows
- Visual Studio Code
- Python
- Optional: GitHub Desktop, if you prefer a GUI

The project requires Python `>=3.10`.

The old virtual environment was created with Python 3.14:

```text
D:\Program Files\Python314\python.exe -m venv H:\envs\research_base
```

If Python 3.14 works well on the new computer, you can use it again. If package installation fails, use a widely supported Python version for the packages you need, such as Python 3.12 or 3.13.

Check installed Python versions:

```powershell
py -0p
```

### 2. Sign In To VS Code

Open VS Code on the new computer.

1. Click the Accounts icon.
2. Sign in with the same account used for Settings Sync.
3. Turn on Settings Sync.
4. Wait for settings and extensions to install.

Sign in again to extensions that need accounts:

- OpenAI / ChatGPT / Codex extension
- GitHub Copilot
- GitHub Pull Requests
- Google Gemini Code Assist, if you use it

Do not manually copy extension login tokens from the old computer.

### 3. Clone The Project

If the new computer has an `H:` drive and you want the same layout:

```powershell
mkdir H:\repos
cd H:\repos
git clone https://github.com/dnevo/clusters_unmixing.git
cd clusters_unmixing
```

If there is no `H:` drive, use another location:

```powershell
mkdir C:\repos
cd C:\repos
git clone https://github.com/dnevo/clusters_unmixing.git
cd clusters_unmixing
```

The repo does not appear to depend on the absolute path `H:\repos`, so another location should work.

### 4. Create A New Python Environment

Option A: match the old layout:

```powershell
mkdir H:\envs
py -m venv H:\envs\research_base
H:\envs\research_base\Scripts\Activate.ps1
```

Option B: keep the environment inside the project:

```powershell
cd H:\repos\clusters_unmixing
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

### 5. Install Project Dependencies

From the project root, with the virtual environment activated:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

The repo has:

```text
pyproject.toml
requirements.txt
```

Prefer:

```powershell
python -m pip install -e .
```

This installs the project package in editable mode and installs dependencies from `pyproject.toml`.

For notebook use, also install notebook-only dependencies:

```powershell
python -m pip install ipython plotly ipykernel
```

If VS Code asks to install `ipykernel` when opening a notebook, allow it.

### 6. Open The Project In VS Code

Open VS Code:

```powershell
code H:\repos\clusters_unmixing
```

If the `code` command is not available, open VS Code manually and choose:

```text
File -> Open Folder -> H:\repos\clusters_unmixing
```

Then select the Python interpreter:

1. Press `Ctrl+Shift+P`.
2. Run `Python: Select Interpreter`.
3. Choose the new environment.

For the old-style layout, choose:

```text
H:\envs\research_base\Scripts\python.exe
```

For the project-local layout, choose:

```text
H:\repos\clusters_unmixing\.venv\Scripts\python.exe
```

### 7. Select The Notebook Kernel

Open:

```text
notebooks/00_clusters_unmixing_experiments.ipynb
```

In the top-right kernel selector, choose the same environment:

```text
research_base
```

or the Python path:

```text
H:\envs\research_base\Scripts\python.exe
```

If the environment is not listed, run:

```powershell
python -m ipykernel install --user --name research_base --display-name "Python (research_base)"
```

Then reload VS Code.

## Verify The Setup

From the project root with the environment activated:

```powershell
python -m pip check
python -c "import clusters_unmixing; print(clusters_unmixing.__file__)"
python main.py
```

To verify the notebook helper flow:

```powershell
python notebooks\_notebook_smoke.py
```

Expected project outputs are written under:

```text
experiments/outputs/
```

This folder is ignored by Git. That is normal.

## Git Workflow On The New Computer

Before starting work:

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

In VS Code Source Control:

1. Review changed files.
2. Stage the files you want.
3. Commit with a clear message.
4. Click `Sync Changes` or `Push`.

Remember:

- Commit saves changes locally.
- Push sends changes to GitHub.
- The other computer only sees changes after push and pull.

## Codex, OpenAI, Copilot, And Other Coding Extensions

Extensions themselves are not part of the Git repo unless listed as workspace recommendations.

Use VS Code Settings Sync for:

- installed extensions
- editor settings
- keyboard shortcuts
- snippets
- UI preferences

On the new computer, sign in again to:

- OpenAI / ChatGPT / Codex extension
- GitHub Copilot
- GitHub
- any other AI or cloud extensions

Do not copy authentication files or tokens manually.

If you use API keys or environment variables, set them again on the new computer using your secure source of truth.

## Troubleshooting

### Imports Are Underlined In VS Code

Check that VS Code is using the correct interpreter:

```text
Ctrl+Shift+P -> Python: Select Interpreter
```

Then choose the environment you created.

### `clusters_unmixing` Cannot Be Imported

Run this from the project root:

```powershell
python -m pip install -e .
```

Then restart the VS Code Python language server:

```text
Ctrl+Shift+P -> Python: Restart Language Server
```

### Notebook Kernel Is Missing

Install and register `ipykernel`:

```powershell
python -m pip install ipykernel
python -m ipykernel install --user --name research_base --display-name "Python (research_base)"
```

Reload VS Code.

### Torch Installation Fails

Use a Python version supported by your installed PyTorch build. If Python 3.14 causes trouble, create the environment again with another installed Python version:

```powershell
py -3.12 -m venv H:\envs\research_base
H:\envs\research_base\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install ipython plotly ipykernel
```

### GitHub Push Or Pull Fails

Make sure you are signed in to GitHub from VS Code or Git Credential Manager.

Check the remote:

```powershell
git remote -v
```

Expected:

```text
origin  https://github.com/dnevo/clusters_unmixing.git
```

## Quick Checklist

Old computer:

- Commit real changes.
- Push to GitHub.
- Back up ignored outputs or private files if needed.
- Turn on VS Code Settings Sync.
- Confirm GitHub has the latest project.

New computer:

- Install Git, VS Code, and Python.
- Turn on VS Code Settings Sync.
- Clone `https://github.com/dnevo/clusters_unmixing.git`.
- Create a new virtual environment.
- Run `python -m pip install -e .`.
- Install notebook dependencies with `python -m pip install ipython plotly ipykernel`.
- Select the new interpreter in VS Code.
- Sign in again to OpenAI/Codex, GitHub, Copilot, and other extensions.
- Run `python main.py` or `python notebooks\_notebook_smoke.py` to verify.
