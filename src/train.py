import torch
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
from model import TextClassificationModel
from utils import get_vocab, get_pipelines, collate_batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EMBED_DIM = 64
EPOCHS = 10
LR = 0.1

vocab = get_vocab()
text_p, label_p = get_pipelines(vocab)
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=lambda b: collate_batch(b, text_p, label_p, device))
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=lambda b: collate_batch(b, text_p, label_p, device))

model = TextClassificationModel(len(vocab), EMBED_DIM, 4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)


for epoch in range(1, EPOCHS + 1):
    model.train()
    for label, text, offsets in train_dataloader:
        optimizer.zero_grad()
        predicted = model(text, offsets)
        loss = criterion(predicted, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    
    
    torch.save(model.state_dict(), 'models/my_model.pth')
    print(f"Epoch {epoch} completed.")