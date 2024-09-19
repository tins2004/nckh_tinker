from data_prediction import *
from data_processing import *

import matplotlib.pyplot as plt


data_path = './data-user-30to39/'
label = int(4)

COL = ["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2", "MAG_0", "MAG_1", "MAG_2", "PRESS", "label"]
COL_NO_LABEL = ["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2", "MAG_0", "MAG_1", "MAG_2", "PRESS"]
   
if __name__ == '__main__':
  main_dataframe = ReadAllData(data_path, 30, 39, COL)

  main_dataframe = CheckLabel(main_dataframe, label)
#   if (main_dataframe == None):
#      exit()
  
  X_test = test_generator(main_dataframe, COL_NO_LABEL)
  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
     
  predictVector = FindLabel(X_test)

  plt.bar(range(len(predictVector)), predictVector)

  # Thêm tiêu đề và nhãn trục
  plt.title('Biểu đồ cột')
  plt.xlabel('Chỉ số')
  plt.ylabel('Giá trị')

  # Hiển thị biểu đồ
  plt.show()
