import numpy as np
import pandas as pd
from matrix_trans import result_matrix, save_excel

loadder_array = np.load('data/new_data.npy')
result = result_matrix(loadder_array, "data/resutl_matrix.npy")
filename = "data/trans_matrix.xlsx"
save_excel(result, filename)
