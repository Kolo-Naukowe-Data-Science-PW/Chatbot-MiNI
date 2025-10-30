# 🤖 Chatbot MiNI

TODO: Add description

## 1. Installation

### 1.1. Clone the repository

   ```bash
   git clone https://github.com/Kolo-Naukowe-Data-Science-PW/Chatbot-MiNI
   cd Chatbot-MiNI
   ```

### 1.2. Create and activate a conda environment with the required packages

   ```bash
   conda env create -f chatbot_mini.yml
   conda activate chatbot_mini
   ```

### 1.3. How to add new packages to the YAML config file, after installing them yourself (in your chatbot_mini environment)

  ```bash
  conda env export > chatbot_mini.yml
  ```

### 1.4. Install pre-commit hooks (run once after cloning)

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 2. Conventions and Usage

### 2.1. General

- **Language:** All code, comments, commit messages, PRs and documentation must be written in **English**.

### 2.2. Commits (Conventional Commits)

- Use short imperative expressions, e.g.:
  - `Add requirements.txt`
  - `Fix data loading bug`
  - `Update README`
- Keep messages concise and focused on **what** was changed (not how).

### 2.3. Branches

- Naming convention:
  - `feature/<short-description>`
  - `fix/<short-description>`
  - `hotfix/<short-description>`
- Example: `feature/create-conventions`
- **Do not push directly to `main`.** Always use a PR.

### 2.4. Pull Requests (PRs)

- Requirements:
  - **At least one approved review** before merging into `main`.
  - Keep PRs small and focused
  - Provide a clear description: *what / why / how*.
- Branches should be deleted after merge (unless long-lived).

### 2.5. Code Style

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

### 2.6. README

Update the README whenever necessary to reflect changes in the project.

### 2.7. Jupyter Notebooks

- Use for exploration and prototyping only.
- Notebooks must be clean and well-documented.
- Avoid committing large datasets or outputs.
- Prefer scripts/modules for production code.
- Save in \`notebooks/\` directory.
