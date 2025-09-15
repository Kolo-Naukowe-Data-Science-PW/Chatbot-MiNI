# 🤖 Chatbot MiNI

TODO: Add description

## Installation

1. Clone the repository:

   ```bash
    git clone https://github.com/Kolo-Naukowe-Data-Science-PW/Chatbot-MiNI
    cd Chatbot-MiNI
    ```

2. Create and activate a conda environment with the required packages:

   ```bash
   conda create --name chatbot_mini --file requirements.txt
   conda activate chatbot_mini
   ```

3. Install pre-commit hooks (run once after cloning):

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 3. Conventions and Usage

### 3.1. General

- **Language:** All code, comments, commit messages, PRs and documentation must be written in **English**.

### 3.2. Commits (Conventional Commits)

- Use short imperative expressions, e.g.:
  - `Add requirements.txt`
  - `Fix data loading bug`
  - `Update README`
- Keep messages concise and focused on **what** was changed (not how).

### 3.3. Branches

- Naming convention:
  - `feature/<short-description>`
  - `fix/<short-description>`
  - `hotfix/<short-description>`
- Example: `feature/create-conventions`
- **Do not push directly to `main`.** Always use a PR.

### 3.4. Pull Requests (PRs)

- Requirements:
  - **At least one approved review** before merging into `main`.
  - Keep PRs small and focused
  - Provide a clear description: *what / why / how*.
- Branches should be deleted after merge (unless long-lived).

### 3.5. Code Style

- **Formatting:** Black (runs via pre-commit).
- **Linting:** Ruff.
- **Imports:** isort (Black profile, via pre-commit if enabled).
- **Docstrings:** every function and class must have a docstring (NumPy style, consistent).
- **Comments:** minimal, only when necessary; prefer clean, self-explanatory code over redundant comments.
- **Naming:**
  - `snake_case` → functions and variables
  - `PascalCase` → classes
  - `UPPER_SNAKE_CASE` → constants
- **Logging:** use the `logging` module (instead of print statements).

### 3.6. README

Update the README whenever necessary to reflect changes in the project.

### 3.7. Jupyter Notebooks

- Use for exploration and prototyping only.
- Notebooks must be clean and well-documented.
- Avoid committing large datasets or outputs.
- Prefer scripts/modules for production code.
- Save in \`notebooks/\` directory.
