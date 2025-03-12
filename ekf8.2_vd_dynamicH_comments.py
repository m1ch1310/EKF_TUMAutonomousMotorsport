import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# Funktion zur Normalisierung eines Winkels in den Bereich [-pi, pi]
def pi_range(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Funktion zur Rotation einer 2D-Kovarianzmatrix um einen gegebenen Winkel - Ausrichten Prozessrauschen zur Fahrzeugrichtung
def rotate_loc_glob(matrix, angle):
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return rotation_matrix @ matrix @ rotation_matrix.T

# Funktion zur Berechnung der Yaw-Rate (Drehgeschwindigkeit) aus Quaternion-Daten
def compute_yaw_rate(orientation_x, orientation_y, orientation_z, orientation_w, dt, phi_ref):
    quaternions = np.column_stack([orientation_x, orientation_y, orientation_z, orientation_w])
    r = R.from_quat(quaternions)  # Umwandlung von Quaternionen in Euler-Winkel
    euler_angles = r.as_euler('xyz', degrees=False)  # Euler-Winkel berechnen
    yaw_global = euler_angles[:, 2]  # Extrahieren des Yaw-Winkels (Drehung um Z-Achse)
    yaw_frenet = yaw_global - phi_ref  # Korrektur durch Frenet-Referenz
    yaw_rate = np.gradient(yaw_frenet, dt)  # Berechnung der Ableitung als Yaw-Rate
    return yaw_rate

# Klasse für den erweiterten Kalman-Filter (EKF) im Frenet-Koordinatensystem
class EKF_Frenet:
    def __init__(self, x_init, params, sensor_init, ego_init):
        self.dt_s = params["timestep_s"]  # Zeitschrittgröße
        self._n_dof = len(x_init)  # Anzahl der Freiheitsgrade des Zustandsvektors
        
        # Initialisierung des Kalman-Filters mit Zustands- und Messdimension
        self.kf = KalmanFilter(dim_x=self._n_dof, dim_z=len(params["H_indices"][sensor_init][0]))
        self.kf.x = np.array(x_init)  # Setzt den Startzustand
        self.kf.x[2] = pi_range(self.kf.x[2])  # Begrenzung des Yaw-Winkels auf [-pi, pi]
        
        # Systemmatrix F initialisieren (Bewegungsmodell)
        self.kf.F = np.eye(self._n_dof)  # Identitätsmatrix als Basis
        self.kf.F[0, 3] = self.dt_s  # Positionsänderung entlang s durch Geschwindigkeit v_s
        self.kf.F[1, 4] = self.dt_s  # Positionsänderung entlang d durch Geschwindigkeit v_d
        self.kf.F[3, 5] = self.dt_s  # Geschwindigkeit v_s wird durch Beschleunigung a_s beeinflusst
        self.kf.F[4, 6] = self.dt_s  # Geschwindigkeit v_d wird durch Beschleunigung a_d beeinflusst
        
        # Prozessrauschen initialisieren und anpassen
        self.Q_glob = self.init_processnoise(params["process_noise"])
        self.update_processnoise()
        
        # Messrauschen initialisieren
        self._measure_covs_dict = params["measure_covs_dict"]
        self.update_measurementnoise(np.array(ego_init["state"][:2]), self.kf.x, sensor_init)
        
        # Initiale Kovarianzmatrix für Zustandsunsicherheit
        self.kf.P = params["P_init"]
        self.kf.P[:2, :2] = rotate_loc_glob(self.kf.P[:2, :2], self.kf.x[2])  # Rotation ins globale System
        
        # Messmatrix H für Sensordaten initialisieren
        self.H_indices = params["H_indices"]
        self.update_output_matrix(sensor_init)
    
    # Aktualisiert die H-Matrix (Messmatrix), die Sensordaten mit dem Zustand verknüpft
    def update_output_matrix(self, sensor_str):
        meas_shape = (len(self.H_indices[sensor_str][0]), self._n_dof)
        if self.kf.H.shape != meas_shape:
            self.kf.H = np.zeros(meas_shape)
        self.kf.H[self.H_indices[sensor_str][0], self.H_indices[sensor_str][1]] = 1.0
    
    # Führt den Vorhersageschritt des Kalman-Filters aus
    def predict(self):
        self.update_processnoise()  # Prozessrauschen anpassen
        self.kf.predict()  # Kalman-Filter Vorhersageschritt
    
    # Führt den Update-Schritt mit Messwerten aus
    def update(self, z_meas, sensor_type, ego_pos):
        self.update_output_matrix(sensor_type)  # Messmatrix anpassen
        self.update_measurementnoise(ego_pos, self.kf.x, sensor_type)  # Messrauschen anpassen
        self.kf.update(z_meas, R=self.kf.R, H=self.kf.H)  # Messupdate des Kalman-Filters
        self.kf.x[2] = pi_range(self.kf.x[2])  # Begrenzung des Winkels auf [-pi, pi]
    
    # Aktualisiert das Prozessrauschen basierend auf der Geschwindigkeit
    def update_processnoise(self):
        velocity = np.linalg.norm(self.kf.x[3:5])  # Berechnung der Geschwindigkeit
        process_scale = max(0.5, velocity)  # Skaliert das Prozessrauschen adaptiv
        self.kf.Q = self.Q_glob * process_scale
        self.kf.Q[1, 1] *= 0.5  # Anpassung der Unsicherheit entlang d
        self.kf.Q[:2, :2] = rotate_loc_glob(self.kf.Q[:2, :2], self.kf.x[2])
    
    # Aktualisiert die Messrausch-Kovarianz basierend auf der Entfernung zum Fahrzeug
    def update_measurementnoise(self, ego_pose, x, sensor):
        velocity = np.linalg.norm(x[3:5])
        detection_distance = np.linalg.norm(x[:2] - ego_pose)
        dynamic_weight = max(0.1, 1.0 / (1.0 + detection_distance + velocity * 0.5))
        self.kf.R = np.copy(self._measure_covs_dict[sensor]) * dynamic_weight  # Dynamische Skalierung
        self.kf.R[1, 1] *= 1.5  # Erhöhte Unsicherheit in der lateralen Richtung
    
    # Initialisiert die Prozessrausch-Kovarianzmatrix
    def init_processnoise(self, process_noise):
        return np.diag(process_noise) * self.dt_s**2


# Hauptprogramm, falls das Skript direkt ausgeführt wird
if __name__ == "__main__":
    # Excel-Dateien laden
    df_ref = pd.read_excel(r'C:\Pfad\zur\Datei.xlsx', sheet_name="Sheet1")
    df_lidar = pd.read_excel(r'C:\Pfad\zur\Datei.xlsx', sheet_name="Sheet1")
    df_odom = pd.read_excel(r'C:\Pfad\zur\Datei.xlsx', sheet_name="Sheet1")
    
    dt = 0.1  # Zeitschrittgröße in Sekunden
    s_ref = df_ref["s"].values  # Referenz-Positionen
    phi_ref = df_ref["phi_ref_cl_rad"].values  # Referenz-Winkel
    s_odom = df_odom["s_odometry"].values  # Odometrie-Positionen
    
    # Interpolieren der Referenz-Winkel für Frenet-Koordinaten
    phi_ref_interp_func = interp1d(s_ref, phi_ref, kind="linear", fill_value="extrapolate")
    phi_ref_interpolated = phi_ref_interp_func(s_odom)
    
    # Berechnung der Yaw-Rate
    yaw_rate = compute_yaw_rate(
        df_odom["orientation_x"].values,
        df_odom["orientation_y"].values,
        df_odom["orientation_z"].values,
        df_odom["orientation_w"].values,
        dt,
        phi_ref_interpolated
    )
    
    # Initialisierung des Kalman-Filters
    x_init = [df_lidar["s_global"].iloc[0], df_lidar["d_global"].iloc[0], 0, 0, 0, yaw_rate[0], 0]
    ego_init = {"state": x_init}
    ekf = EKF_Frenet(x_init=x_init, params={}, sensor_init="lidar", ego_init=ego_init)
    
    # Schleife über alle Messwerte
    filtered_states = []
    for i in range(len(df_lidar)):
        s_meas, d_meas = df_lidar.loc[i, ["s_global", "d_global"]]
        if np.isnan(s_meas) or np.isnan(d_meas):
            continue  # Überspringe ungültige Messungen
        
        ekf.predict()  # Vorhersageschritt
        ego_pos = [df_lidar.loc[i, "s_odometry"], df_lidar.loc[i, "d_odometry"]]
        ekf.update([s_meas, d_meas], "lidar", ego_pos)  # Messupdate
        filtered_states.append(ekf.kf.x.copy())  # Speichern des gefilterten Zustands
    filtered_states = np.array(filtered_states)
    df_lidar.loc[df_lidar["s_global"].notna(), "s_filtered"] = filtered_states[:, 0]
    df_lidar.loc[df_lidar["d_global"].notna(), "d_filtered"] = filtered_states[:, 1]
    df_lidar.to_excel(r'C:\Users\2d\Engineering\Studium\Master\3Semester\Semesterarbeit\ekf8.2_vd_dynamicH.xlsx', index=False)