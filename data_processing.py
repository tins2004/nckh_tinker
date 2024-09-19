import numpy as np
import pandas as pd
import os


def ReadAllData(path, startFolder, endFolder, field):
  main_dataframe = pd.DataFrame()

  for i in range(startFolder, endFolder):
    folder_data = path + str(i)
    print(folder_data)
    csv_file = os.listdir(folder_data)
    for file in csv_file:
      if file.endswith('.csv'):
        df = pd.read_csv(folder_data + '/' + file)
        main_dataframe = pd.concat([main_dataframe, df])
  
  main_dataframe = main_dataframe.loc[(main_dataframe['label'] != 6.0) & (main_dataframe['label'] != -1.0)]
  main_dataframe = main_dataframe[field]

  return main_dataframe

def ReadData(path, field):
  main_dataframe = pd.DataFrame()

  csv_file = os.listdir(path)
  for file in csv_file:
      if file.endswith('.csv'):
          df = pd.read_csv(path + '/' + file)
          main_dataframe = pd.concat([main_dataframe, df])
  
  main_dataframe = main_dataframe.loc[(main_dataframe['label'] != 6.0) & (main_dataframe['label'] != -1.0)]
  main_dataframe = main_dataframe[field]

  return main_dataframe

def CheckLabel(data, labelCheck):
  data["label"] = data["label"].apply(str)

  if data["label"].str.contains(str(labelCheck)).any():
    return data[data["label"].str.contains(str(labelCheck))]
  else:
    print("no label")
    return None