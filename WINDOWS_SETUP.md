# LogicMD — Windows setup notes

This folder contains a compatibility-cleanup of the supplied LogicMD repository.

## 1. Create and activate a virtual environment (PowerShell)

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `py` is unavailable but `python --version` works, use `python -m venv .venv`.

## 2. Install core dependencies

Install PyTorch using the official selector at https://pytorch.org/get-started/locally/ if you need a specific CUDA build. For CPU-only use:

```powershell
python -m pip install torch torchvision torchaudio
python -m pip install numpy pandas scipy scikit-learn matplotlib tqdm transformers sparsemax pillow opencv-python tensorboard
```

## 3. Verify the code imports

```powershell
python -m compileall -q .
python -c "import models; print('models import OK')"
python -c "import train_two; print('train_two import OK')"
```

## Important limitations before training

- The repository ZIP does not include the datasets or pretrained BERT checkpoints referenced by the original authors.
- Twitter/Weibo dataset defaults in `utils/datasets.py` and BERT defaults in `utils/data_utils.py` refer to the original author's machine. You must download the dataset/preprocessing files from the original README and configure paths for your computer.
- `utils/finefake_dataset_v2.py` is absent from the supplied ZIP. FineFake mode now reports a clear explanation; Twitter and Weibo imports do not require that optional module.
- This cleanup does not claim that full training works yet. Model execution still needs dataset files, compatible pretrained checkpoints, and a forward-pass test on your machine.
- Several model files contain hard-coded `.cuda()` calls. CPU-only training is not yet guaranteed; a device-portability pass and real dataset test are still required.

See the original README for the authors' dataset link and experiment instructions.
