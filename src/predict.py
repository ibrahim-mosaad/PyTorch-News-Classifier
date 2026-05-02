import torch
from model import TextClassificationModel
from utils import get_vocab, get_pipelines

def run_prediction(text):
    device = torch.device("cpu")
    vocab = get_vocab()
    text_p, _ = get_pipelines(vocab)
    
    model = TextClassificationModel(len(vocab), 64, 4)
    model.load_state_dict(torch.load('models/my_model.pth', map_location=device))
    model.eval()
    
    ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
    
    with torch.no_grad():
        processed_text = torch.tensor(text_p(text))
        output = model(processed_text, torch.tensor([0]))
        return ag_news_label[output.argmax(1).item() + 1]

if __name__ == "__main__":
    sample = "The new spacecraft reached Mars orbit successfully."
    print(f"Result: {run_prediction(sample)}")