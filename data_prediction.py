import numpy as np
import tflite_runtime.interpreter as tflite

model_path = './Model_Swiming.tflite'



interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()

def find_multiples_of_180(len):
    multiples = []
    for i in range(0, len+1, 180):
        multiples.append(i)
    return multiples

def test_generator(dataframe, field):
    X_test = np.zeros((int(dataframe.shape[0] / 180), 180, dataframe[field].shape[1]))

    for i in range(int(dataframe.shape[0] / 180)):
      pos_result = find_multiples_of_180(dataframe.shape[0])
      pos = np.random.randint(len(pos_result) - 1)

      X_test[i, :, :] = dataframe[field][pos_result[pos]:pos_result[pos + 1]]

    return X_test

def run_inference(data_test):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    expected_input_shape = input_details[0]['shape']

    if not np.array_equal(data_test.shape, expected_input_shape):
        print("Data shape incompatible with model input. Please check data preprocessing or model expectations.")
        return None

    data_test = data_test.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], data_test)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def FindLabel(x_test):
  count_label_0 = 0
  count_label_1 = 0
  count_label_2 = 0
  count_label_3 = 0
  count_label_4 = 0
  
  for i in range(len(x_test)):
    data_test = x_test[i]
    data_test = data_test.reshape((1, x_test.shape[1], x_test.shape[2], x_test.shape[3]))

    result = run_inference(data_test)
    max_value, max_index = max(enumerate(result[0]), key=lambda x: x[1])

    if max_value == 0:
      count_label_0 = count_label_0 + 1
    elif max_value == 1:
      count_label_1 = count_label_1 + 1
    elif max_value == 2:
      count_label_2 = count_label_2 + 1
    elif max_value == 3:
      count_label_3 = count_label_3 + 1
    elif max_value == 4:
      count_label_4 = count_label_4 + 1

  print("count_label_0")
  print(count_label_0)

  print("count_label_1")
  print(count_label_1)

  print("count_label_2")
  print(count_label_2)

  print("count_label_3")
  print(count_label_3)

  print("count_label_4")
  print(count_label_4)
