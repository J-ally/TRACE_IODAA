from TRACE_module.preprocessing import *
import os

data_folder = "data/Data_rssi_glob_sensor_time/"
liste_fichiers=[os.path.join()]

liste_fichiers = [
"20240319-20240410_366a_glob_sensor_time_ble.parquet",
"20240319-20240410_366b_glob_sensor_time_ble.parquet",
"20240319-20240410_366c_glob_sensor_time_ble.parquet",
"20240319-20240410_366d_glob_sensor_time_ble.parquet",
"20240319-20240410_3660_glob_sensor_time_ble.parquet",
"20240319-20240410_3662_glob_sensor_time_ble.parquet",
"20240319-20240410_3663_glob_sensor_time_ble.parquet",
"20240319-20240410_3664_glob_sensor_time_ble.parquet",
"20240319-20240410_3665_glob_sensor_time_ble.parquet",
"20240319-20240410_3666_glob_sensor_time_ble.parquet",
"20240319-20240410_3668_glob_sensor_time_ble.parquet"
]
list_full_path=[data_folder+i for i in liste_fichiers]in()
data=concatenate_df(list_full_path)
data=transform_rssi_to_distance(data)
list_id=list(pd.unique(data["id_sensor"]))
stack,list_timesteps=create_matrix_stack(data,list_id)