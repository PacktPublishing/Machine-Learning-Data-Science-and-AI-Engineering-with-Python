# Run multiple experiments with Optuna & Mlflow

Perform the following steps.

### Step 1. Perform initial baseline run.

```bash
mkdir -p configs models

python train.py --config configs/baseline.yaml
```
### Step 2. Run hyperparameter optimization

```bash
python sweep_optuna.py
```

### Step 3. Identify the best model

Load the MLflow dashboard and identify the trial with best validation accuracy.

```bash
mlflow ui
```

### Step 4. Run evaluation

```bash
python evaluate.py --model-path models/fashion_cnn_baseline.pth
```
