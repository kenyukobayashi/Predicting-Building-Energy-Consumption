import pandas as pd

if __name__ == '__main__':
    inp = pd.read_csv('data/sanitized.csv', delimiter=',')
    print(inp.shape)
    print(inp[:10])
    print(inp['GBAUP'].value_counts())
    print(inp['GKLAS'].value_counts())