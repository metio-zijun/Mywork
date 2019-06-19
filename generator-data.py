import pandas as pd
import numpy as np


all = pd.DataFrame()
for i in range(1,120):
    num = np.random.randint(150,250,size=1, dtype=np.int)
    tset = np.random.randint(24, 28, size=num, dtype=np.int).reshape(-1, 1)
    tin = np.random.randint(24, 28, size=num, dtype=np.int).reshape(-1, 1)
    id = np.array([i]).repeat(num).reshape(-1, 1)
    range = np.arange(1, num+1).reshape(-1, 1)
    ndarray = np.concatenate([id, range, tset, tin], axis=1)
    idf = pd.DataFrame(ndarray, columns=['id', 'time', 'tset', 'tin'])
    all = pd.concat([all, idf], axis=0)
