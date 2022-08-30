from utils import *

from tabulate import tabulate

knn_temps = read_pkl('./logs/knn_n_temps.pkl')

max_avg = 0
max_avg_n_temp = ''
for key, val in knn_temps.items():
    if key == 'Category':
        continue
    avg = np.mean(val)
    if avg > max_avg:
        max_avg = avg
        max_avg_n_temp = key
    # print(key, avg)

print(tabulate(knn_temps, tablefmt='html', headers='keys'))