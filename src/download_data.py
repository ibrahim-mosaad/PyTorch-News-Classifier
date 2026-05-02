import os
import pandas as pd
from torchtext.datasets import AG_NEWS

def download_and_save_samples(output_dir='data', num_samples=100):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

   
    print("Downloading AG_NEWS dataset...")
    train_iter = AG_NEWS(split='train')
    
    samples = []
   
    for i, (label, text) in enumerate(train_iter):
        if i >= num_samples:
            break
        samples.append({"Label": label, "Text": text})

   
    df = pd.DataFrame(samples)
    file_path = os.path.join(output_dir, 'sample_data.csv')
    df.to_csv(file_path, index=False, encoding='utf-8')
    
    print(f"Successfully saved {num_samples} samples to {file_path}")

if __name__ == "__main__":
    download_and_save_samples()