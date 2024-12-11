import os
import pandas as pd
import requests
from io import StringIO
import certifi

url = 'http://www.stat.cmu.edu/~rnugent/PUBLIC/teaching/CMU729/HW/iris.txt'
local_filename = 'iris.txt'

# Check if the file already exists locally
if not os.path.exists(local_filename):
    # Download the file and save it locally
    response = requests.get(url, verify=False)
    with open(local_filename, 'w') as f:
        f.write(response.text)

# Read the data from the local file
data = pd.read_csv(local_filename, sep=' ')
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print(data.head())
print(len(data))
print(data.species.value_counts())

features = data.columns.tolist()[:-1]
species = data.species.unique()
print(features)
