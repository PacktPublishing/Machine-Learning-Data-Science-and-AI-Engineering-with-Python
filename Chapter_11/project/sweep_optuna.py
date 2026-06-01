import optuna
import yaml

from train import run_training


BASE_CONFIG_PATH = "configs/baseline.yaml"


def objective(trial):
    with open(BASE_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["batch_size"] = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128],
    )

    config["learning_rate"] = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-2,
        log=True,
    )

    config["dropout"] = trial.suggest_float(
        "dropout",
        0.2,
        0.5,
    )

    config["optimizer"] = trial.suggest_categorical(
        "optimizer",
        ["Adam", "SGD"],
    )

    config["run_name"] = f"optuna_fashion_cnn_trial_{trial.number}"
    config["model_path"] = f"models/fashion_cnn_trial_{trial.number}.pth"

    val_acc = run_training(config)

    return val_acc


def main():
    study = optuna.create_study(
        study_name="fashion_mnist_cnn",
        direction="maximize",
        storage="sqlite:///optuna_fashion_mnist_cnn.db",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=10,
    )

    print("\nBest trial:")
    print(study.best_trial.number)

    print("\nBest validation accuracy:")
    print(f"{study.best_value:.4f}")

    print("\nBest hyperparameters:")
    print(study.best_params)


if __name__ == "__main__":
    main()