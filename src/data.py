import argparse
from src.preprocess import preprocess_text_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for XMC")
    parser.add_argument('--dataset_name', type=str, default='eurlex-4k', help='Name of the dataset')
    data_dir = f"xmc-base/{parser.parse_args().dataset_name}"
    print("data_dir:", data_dir)
    preprocess_text_file(data_dir=data_dir, label_sep=",")
    print("Preprocessing completed for dataset:", data_dir)
