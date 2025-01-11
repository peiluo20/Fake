import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_and_split_data(config):
    df = pd.read_csv(config['data']['csv_path'])
    image_dir = config['data']['image_dir']

    # Match images
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir)}

    def get_image_path(videoname):
        basename = os.path.splitext(videoname)[0]
        return os.path.join(image_dir, image_files.get(basename, "")) if basename in image_files else None

    df['image_path'] = df['videoname'].apply(get_image_path)
    df = df.dropna(subset=['image_path'])
    train_df, val_df = train_test_split(df, test_size=config['data']['test_split'], stratify=df['label'])
    print(f"训练集数量: {len(train_df)}, 验证集数量: {len(val_df)}")
    return train_df, val_df
