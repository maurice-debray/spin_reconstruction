import numpy as np
from constants import gamma_ratio

csv_data = np.genfromtxt("spin_couplings.csv", delimiter=",", filling_values = np.nan)
raw_data = csv_data[2:,2:]

renormalized_data = np.empty(raw_data.shape)


nb_par_data =  csv_data[1,2:]*2

renormalized_data = raw_data*2


a_par_data = np.array([-6.41, 0.88, 0.24, -0.95, 2.23, 2.23, -0.61, 4.46, 22.21])*1e3 # A parallel in Hz (in fact it is not exactly that but isok)

data_header = csv_data[0,1:]