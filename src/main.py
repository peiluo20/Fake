import yaml
from data_loader import load_and_split_data
from train import train_model

if __name__ == "__main__":
    with open("../configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_df, val_df = load_and_split_data(config)

    train_model(train_df, val_df, config)
