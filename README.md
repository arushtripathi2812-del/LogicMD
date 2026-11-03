# LogicMD
The official implementation of our paper "Interpretable Multimodal Misinformation Detection with Logic Reasoning", was accepted by Finding of ACL 23 (4/4/4 for excitement and soundness). More detailed documents will be uploaded in a few weeks. If you have some questions related to this paper, please email me (liuhui3-c@my.cityu.edu.hk).

# Dataset 
You can download the dataset and the related pre-processing files in this link \url{https://portland-my.sharepoint.com/:u:/g/personal/liuhui3-c_my_cityu_edu_hk/EYR-45i16q9EivlGM1ZCe9cBrPsuOjr8O9fziihKJPLIoA?e=2KVCFW}.

Update:
This is the new dataset link: https://1drv.ms/u/c/11e65424f2029e02/ETMtRqYt7YhHsL5dcwnIZMYB6ZD05OeLID25MtZrL6SdEA?e=FGDau3
# Instruction to run our code
Our models are implemented in models/rule_detection.py. To reproduce the results of experiments, you can use train_sarcasm.py for the Sarcasm dataset and train_two.py for Weibo and Twitter datasets. 

Since I used comet package in these files, you may remove codes related to this package or just replace my api keys with yours. After that, you can run sarcasm.sh, twitter.sh, and weibo.sh directly. 

Note (2023/12/08): I found some problems in codes, try to fix them in a few weeeks.

Note (2025/2/05): I have fixed this error.

## Local synthetic smoke check

The original Twitter, Weibo, and Sarcasm experiments need the authors' processed datasets and pretrained assets. Those files are not included here. The `train.py` file is also an incomplete scaffold, so it is not a runnable training entry point.

This repository includes a small synthetic input manifest at `data/synthetic_demo.csv` and `run_synthetic_demo.py`. The script builds deterministic feature tensors and runs them through the supplied `SimilarityModule` and `DetectionModule` in `model.py`. The CSV labels are not used. This checks that the model's forward pass runs; it does **not** train the model, evaluate misinformation detection, or reproduce the paper's results.

### Run on Windows PowerShell

From the repository root, create and activate an environment, install PyTorch, then run the smoke check:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy
python .\run_synthetic_demo.py
```

Expected successful output:

```text
LogicMD synthetic forward-pass smoke check: OK
Dataset rows: 6 (synthetic; labels not used)
Text features: (6, 200, 200) | image features: (6, 512)
Similarity logits: (6, 2)
Detection logits: (6, 2)
No training, dataset evaluation, or paper result is claimed.
```

The output above is from a real local run of the script. It is recorded as text here; no screenshot is included yet. To reproduce the research experiments, obtain the original processed datasets and checkpoints, configure their paths, and use the corresponding `train_sarcasm.py` or `train_two.py` workflow described above.

See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for the fuller environment notes.



