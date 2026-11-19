# Installation and Setup Guide: Thyroid Disease Detection

**Project:** Thyroid Disease Detection Using Machine Learning  
**Author:** Venkatesh Devarakonda  
**Target Environment:** Windows, macOS, or Linux (Python 3.9 to 3.12)  

---

## 1. System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python Version:** Python 3.9, 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- **RAM:** Minimum 4 GB (8 GB recommended for training SVM with probabilities)
- **Disk Space:** ~500 MB for virtual environment and dataset artifacts

---

## 2. Step-by-Step Installation

### Step 1: Clone or Navigate to the Repository
Open PowerShell or your preferred terminal:
```powershell
cd "path/to/thyroid-detection"
```

### Step 2: Create an Isolated Virtual Environment
It is best practice to keep all project dependencies isolated from system-wide Python packages:

**Using standard Python:**
```powershell
python -m venv .venv
```

**Using uv (ultra-fast package manager):**
```powershell
uv venv .venv --python 3.11
```

### Step 3: Activate the Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an `ExecutionPolicy` restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again).*
- **On Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **On macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 4: Install Dependencies
Upgrade `pip` and install all required packages from `requirements.txt`:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verifying the Installation

To verify that all required packages and modules are correctly installed, run the unit test suite:
```powershell
pytest -v tests
```
*Expected Output:*
```
============================== 9 passed in 3.93s ==============================
```

---

## 4. Running the Entire Machine Learning Pipeline

### A. Generate Exploratory Visualizations:
```powershell
python src/generate_eda_figures.py
```

### B. Execute Preprocessing and Scaling:
```powershell
python -m src.data_preprocessing
```

### C. Train and Benchmark All 6 Models:
```powershell
python -m src.model_evaluation
```

### D. Launch Interactive Jupyter Notebooks:
```powershell
jupyter notebook notebooks/
```

---

## 5. Troubleshooting Common Issues

### Issue 1: Windows Store Execution Alias Error (`os error 1920`)
- **Cause:** Windows enables placeholder execution aliases for `python.exe` that point to the Microsoft Store.
- **Solution:** In Windows Settings $\rightarrow$ Search for *"Manage app execution aliases"* $\rightarrow$ Turn OFF `App Installer: python.exe` and `App Installer: python3.exe`.

### Issue 2: Matplotlib Permission Error (`[Errno 13] Permission denied`)
- **Cause:** In sandboxed or restricted corporate environments, writing to `C:\Users\<user>\.matplotlib` may be blocked.
- **Solution:** Set a local cache path before running:
  ```powershell
  $env:MPLCONFIGDIR = "$((Get-Location).Path)\.matplotlib"
  ```

### Issue 3: `ModuleNotFoundError: No module named 'src'`
- **Cause:** Python script executed directly without `src` in `sys.path`.
- **Solution:** Execute modules with `-m`:
  ```powershell
  python -m src.model_evaluation
  ```
  Or set `PYTHONPATH`:
  ```powershell
  $env:PYTHONPATH = "$((Get-Location).Path)"
  ```
