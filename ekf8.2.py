import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

def pi_range(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotate_loc_glob(matrix, angle):
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return rotation_matrix @ matrix @ rotation_matrix.T

def compute_yaw_rate(orientation_x, orientation_y, orientation_z, orientation_w, dt, phi_ref):
    quaternions = np.column_stack([orientation_x, orientation_y, orientation_z, orientation_w])
    r = R.from_quat(quaternions)
    euler_angles = r.as_euler('xyz', degrees=False)
    yaw_global = euler_angles[:, 2]
    yaw_frenet = yaw_global - phi_ref
    yaw_rate = np.gradient(yaw_frenet, dt)
    return yaw_rate

class EKF_Frenet:
    def __init__(self, x_init, params, sensor_init, ego_init):
        self.dt_s = params["timestep_s"]
        self._n_dof = params["P_init"].shape[0]
        
        self.kf = KalmanFilter(dim_x=self._n_dof, dim_z=len(params["H_indices"][sensor_init][0]))
        self.kf.x = np.array(x_init)
        self.kf.x[2] = pi_range(self.kf.x[2])  
        
        self.kf.F = np.eye(self._n_dof)
        self.Q_glob = self.init_processnoise(params["process_noise"])
        self.update_processnoise()
        
        self._measure_covs_dict = params["measure_covs_dict"]
        self.update_measurementnoise(np.array(ego_init["state"][:2]), self.kf.x, sensor_init)
        
        self.kf.P = params["P_init"]
        self.kf.P[:2, :2] = rotate_loc_glob(self.kf.P[:2, :2], self.kf.x[2])
        
        self.H_indices = params["H_indices"]
        self.update_output_matrix(sensor_init)
    
    def update_output_matrix(self, sensor_str):
        meas_shape = (len(self.H_indices[sensor_str][0]), self._n_dof)
        if self.kf.H.shape != meas_shape:
            self.kf.H = np.zeros(meas_shape)
        self.kf.H[self.H_indices[sensor_str][0], self.H_indices[sensor_str][1]] = 1.0
    
    def predict(self):
        self.update_processnoise()
        self.kf.predict()
    
    def update(self, z_meas, sensor_type, ego_pos):
        self.update_output_matrix(sensor_type)
        self.update_measurementnoise(ego_pos, self.kf.x, sensor_type)
        self.kf.update(z_meas, R=self.kf.R, H=self.kf.H)
        self.kf.x[2] = pi_range(self.kf.x[2])
    
    def update_processnoise(self):
        velocity = np.linalg.norm(self.kf.x[3:5])
        process_scale = max(0.5, velocity)
        self.kf.Q = self.Q_glob * process_scale
        self.kf.Q[1, 1] *= 0.5
        self.kf.Q[:2, :2] = rotate_loc_glob(self.kf.Q[:2, :2], self.kf.x[2])
    
    def update_measurementnoise(self, ego_pose, x, sensor):
        dynamic_weight = max(0.1, 1.0 / (1.0 + np.linalg.norm(x[:2] - ego_pose)))
        self.kf.R = np.copy(self._measure_covs_dict[sensor]) * dynamic_weight
        self.kf.R[1, 1] *= 1.5
    
    def init_processnoise(self, process_noise):
        return np.diag(process_noise) * self.dt_s**2

if __name__ == "__main__":
    df_ref = pd.read_excel(r'C:\Users\2d\Engineering\Studium\Master\3Semester\Semesterarbeit\Kentucky_Raceline_Frenet.xlsx', sheet_name="Sheet1")
    df_lidar = pd.read_excel(r'C:\Users\2d\Engineering\Studium\Master\3Semester\Semesterarbeit\SimpleData5_Frenet.xlsx', sheet_name="Sheet1")
    df_odom = pd.read_excel(r'C:\Users\2d\Engineering\Studium\Master\3Semester\Semesterarbeit\Odometry_Relative_Heading.xlsx', sheet_name="Sheet1")
    
    dt = 0.1  
    s_ref = df_ref["s"].values
    phi_ref = df_ref["phi_ref_cl_rad"].values
    s_odom = df_odom["s_odometry"].values
    
    phi_ref_interp_func = interp1d(s_ref, phi_ref, kind="linear", fill_value="extrapolate")
    phi_ref_interpolated = phi_ref_interp_func(s_odom)
    
    df_odom["orientation_x"].interpolate(method="linear", limit_direction="both", inplace=True)
    df_odom["orientation_y"].interpolate(method="linear", limit_direction="both", inplace=True)
    df_odom["orientation_z"].interpolate(method="linear", limit_direction="both", inplace=True)
    df_odom["orientation_w"].interpolate(method="linear", limit_direction="both", inplace=True)
    
    yaw_rate = compute_yaw_rate(
        df_odom["orientation_x"].values,
        df_odom["orientation_y"].values,
        df_odom["orientation_z"].values,
        df_odom["orientation_w"].values,
        dt,
        phi_ref_interpolated
    )
    

params = {
    "timestep_s": dt,
    "P_init": np.eye(6) * 1.0,  # Reduziere Anfangskovarianz für stabilere Schätzungen
    "measure_covs_dict": {"lidar": np.diag([0.03, 0.1])},  # Geringere Messunsicherheiten für s
    "H_indices": {"lidar": ([0, 1], [0, 1])},
    "process_noise": [0.08, 0.05, 0.003, 0.003, 0.001, 0.0001]  # Erhöhtes Prozessrauschen für s
}

x_init = [df_lidar["s_global"].iloc[0], df_lidar["d_global"].iloc[0], 0, 0, yaw_rate[0], 0]
ego_init = {"state": x_init}
ekf = EKF_Frenet(x_init=x_init, params=params, sensor_init="lidar", ego_init=ego_init)

filtered_states = []
for i in range(len(df_lidar)):
    s_meas, d_meas = df_lidar.loc[i, ["s_global", "d_global"]]
    if np.isnan(s_meas) or np.isnan(d_meas):
        continue
    
    ekf.predict()
    ego_pos = [df_lidar.loc[i, "s_odometry"], df_lidar.loc[i, "d_odometry"]]
    ekf.update([s_meas, d_meas], "lidar", ego_pos)
    filtered_states.append(ekf.kf.x.copy())

filtered_states = np.array(filtered_states)
df_lidar.loc[df_lidar["s_global"].notna(), "s_filtered"] = filtered_states[:, 0]
df_lidar.loc[df_lidar["d_global"].notna(), "d_filtered"] = filtered_states[:, 1]
df_lidar.to_excel(r'C:\Users\2d\Engineering\Studium\Master\3Semester\Semesterarbeit\ekf8.2_input_sd.xlsx', index=False)
