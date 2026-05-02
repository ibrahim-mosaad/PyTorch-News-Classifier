# 📰 News-Torch: High-Performance News Classification Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)

##  Abstract
This repository implements an end-to-end automated categorization system designed to organize vast historical news archives into four strategic topics: **World, Sports, Business, and Sci/Tech**.[cite: 1] Utilizing the `AG_NEWS` dataset, the system leverages `torchtext` for efficient NLP pipelining and a custom Neural Network architecture for high-speed inference.[cite: 1]

##  Key Technical Innovations

*   **Optimized Vector Aggregation**: Implements `nn.EmbeddingBag`, which directly computes the mean of word embeddings within each document, reducing memory overhead and computational complexity.[cite: 1]
*   **Dynamic Data Pipelining**: Features a robust `collate_batch` function that handles variable-length text sequences through an automated **Offsets** calculation system.[cite: 1]
*   **Stability & Convergence**: Employs **Gradient Clipping** to mitigate exploding gradients and a **StepLR Scheduler** to ensure precise convergence during the final epochs.[cite: 1]
*   **Semantic Manifold Visualization**: Includes 3D **t-SNE** (t-Distributed Stochastic Neighbor Embedding) visualization to demonstrate how the model clusters news topics in a high-dimensional vector space.[cite: 1]

##  Architecture & Mathematics
The model follows a streamlined yet powerful architecture[cite: 1]:
1.  **Input Layer**: Text tokens are processed via a basic English tokenizer and mapped to unique indices in a vocabulary.[cite: 1]
2.  **EmbeddingBag Layer**: $64$-dimensional dense vectors are learned for each token and aggregated per document.[cite: 1]
3.  **Linear Classifier**: A fully connected layer maps aggregated embeddings to class logits.[cite: 1]
4.  **Softmax Logits**: Provides probability distribution across the 4 target classes.[cite: 1]

##  Dataset: AG_NEWS
*   **Total Training Samples**: 120,000[cite: 1]
*   **Total Testing Samples**: 7,600[cite: 1]
*   **Categories**: World (1), Sports (2), Business (3), Sci/Tec (4)[cite: 1]

##  Installation & Reproduction
1.  **Clone the Repo**:
    ```bash
      git clone https://github.com/ibrahim-mosaad/PyTorch-News-Classifier.git
    ```
2.  **Setup Environment**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Training**:
    ```bash
    python src/train.py
    ```

##  Performance Metrics
After 10 epochs of training with a Learning Rate of $0.1$, the model achieves[cite: 1]:
*   **Validation Accuracy**: >80%[cite: 1]
*   **Test Accuracy**: ~82%[cite: 1]
*   **Loss Function**: Cross-Entropy Loss[cite: 1]

##  Predictive Inference
The model can predict categories for unseen text instantly[cite: 1]:
```python
from src.predict import run_prediction
headline = "The aerospace industry is booming with new orbital satellite launches."
print(f"Prediction: {run_prediction(headline)}") 
# Output: Sci/Tec
```

##  Author
**Ibrahim Hamada Mosaad**  
*AI Engineer & Researcher | NTI Lecturer*  
*Specialized in Computer Vision, NLP, and Deep Learning Foundations.*

```
