#injestion.py

# Ingest Data here

#Combine the CSV Files for the most important features we will be using

import pandas as pd
df = pd.read_csv("data.csv")
print(df)


import pandas as pd
# merging two csv files
df = pd.concat(
    map(pd.read_csv, ['mydata.csv', 'mydata1.csv']), ignore_index=True)
print(df)
