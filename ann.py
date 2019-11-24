import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def mse(t1, t2):
  diff = t1 - t2
  return torch.sum(diff * diff) / diff.numel()


if __name__ == '__main__':
    dataset = pd.read_csv('data/features.csv').set_index('EGID')
    dataset = (dataset - dataset.mean()) / dataset.std()
    print(dataset)
    y = torch.from_numpy(dataset[['heating']].values).long()
    features = torch.from_numpy(dataset.drop(columns=['heating', 'cooling']).values).float()
    n_features = features.shape[1]
    n_output = y.shape[1]
    n_hidden = int((n_features + n_output) / 2)

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(n_features, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_output),
                          nn.ReLU())
    # criterionH = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    dataloader = DataLoader(list(zip(features, y)), batch_size=16, shuffle=True)

    for e in range(10):
      loss = 0
      for x, y in dataloader:
        optimizer.zero_grad()
        outputH = model(x)
        loss = mse(outputH, y.squeeze(1))
        loss.backward()
        print(loss)
        print(outputH)
