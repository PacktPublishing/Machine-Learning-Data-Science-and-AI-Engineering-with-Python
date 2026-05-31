# Training, optimizing, monitoring models


### Simple training a solution

Run `fashion-mnist-solution-using-pytorch.ipynb`.


### Monitoring using tensorboard

#### Step 1: run training

```bash
python fashion_mnist_cnn_pytorch_tensorboard.py
```

#### Step 2: show evolution with tensorboard

Run: 
```bash
tensorboard --logdir=runs
```
and then open the link on localhost in your browser shown in the output of this command.

### Hyperparameter optimization with Optuna

#### Step 1: run optimization

```bash
python fashion_mnist_cnn_pytorch_optuna.py
```

#### Step 2: show optimization results using Optuna dashboard

```bash
optuna-dashboard sqlite:///optuna_fashion_mnist.db
```
and then open the link on localhost in your browser shown in the output of this command.

### Scaling hyperparameter optimization with Ray Tune


```bash
python fashion_mnist_cnn_pytorch_raytune.py
```

### Run multiple experiments with Optuna & Mlflow

Follow instructions from [project/README.md](project/README.md)