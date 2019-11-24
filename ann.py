import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dataset = pd.read_csv('data/features.csv')
    y = torch.from_numpy(dataset[['EGID', 'heating', 'cooling']].set_index('EGID').values).long()
    features = torch.from_numpy(dataset.drop(columns=['heating', 'cooling']).set_index('EGID').values).float()
    n_features = features.shape[1]
    n_output = y.shape[1]
    n_hidden = int((n_features + n_output) / 2)

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(n_features, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_output),
                          nn.ReLU())
    criterionH = nn.NLLLoss()
    criterionC = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    dataloader = DataLoader(list(zip(features, y)), batch_size=16, shuffle=True)

    for e in range(5):
      loss = 0
      for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        print(output)
        loss = criterionH(output[0], y[0]) + criterionC(output[1], y[1])
        exit(1)
    feature_tensor = torch.tensor(features)
    print(n_features, n_output, n_hidden)
    print(feature_tensor)