from tdc.multi_pred import PPI
import os

def download_huri(save_dir="data/HuRI"):
    """
    Download HuRI dataset from TDC and save train/valid/test splits locally.
    """
    os.makedirs(save_dir, exist_ok=True)

    data = PPI(name='HuRI')
    split = data.get_split()

    for split_name, df in split.items():
        file_path = os.path.join(save_dir, f"huri_{split_name}.csv")
        df.to_csv(file_path, index=False)
        print(f"✅ Saved {split_name} split to {file_path}")

    print("All splits saved successfully.")

if __name__ == "__main__":
    download_huri()
