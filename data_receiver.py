import threading
import time
import numpy as np
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
from data_processor import DataProcessor


class DataReceiver(QObject):
    def __init__(self, server_ip, server_port, visualization_widget, read_frequency=200, threshold=30):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.tcp_client = TcpClient(self.server_ip, self.server_port, read_frequency=read_frequency)
        self.threshold = threshold
        self.visualization_widget = visualization_widget
        self.processor = DataProcessor(self.visualization_widget)
        self.datacycle = {}  # Dictionnaire pour stocker les données du cycle
        self.current_frame = 0  # Initialiser le compteur de frames
        self.lock = threading.Lock()  # Pour éviter les conflits de données
        self.dofcorr = {"LHip": (36, 37, 38),
                        "LKnee": (39, 40, 41),
                        "LAnkle": (42, 43, 44),
                        "RHip": (27, 28, 29),
                        "RKnee": (30, 31, 32),
                        "RAnkle": (33, 34, 35),
                        "LShoulder": (18, 19, 20),
                        "LElbow": (21, 22, 23),
                        "LWrist": (24, 25, 26),
                        "RShoulder": (9, 10, 11),
                        "RElbow": (12, 13, 14),
                        "RWrist": (15, 16, 17),
                        "Thorax": (6, 7, 8),
                        "Pelvis": (3, 4, 5)}

    def start_receiving(self):
        print("Début de la réception des données...")
        while True:
            tic = time.time()

            received_data = self.tcp_client.get_data_from_server(command=['Force', 'Markers', 'Angle', 'MarkersNames'])
            # Organisation des données reçues
            mks_data = {}
            for i, name in enumerate(received_data['MarkersNames']):
                mks_data[name] = np.array([received_data['Markers'][0][i, :], received_data['Markers'][1][i, :],
                                           received_data['Markers'][2][i, :]])

            frc_data = {1: {}, 2: {}}  # Pour deux plateformes de force
            for pfnum in [1, 2]:
                start_idx = (pfnum - 1) * 6
                for i, comp in enumerate(['Force', 'Moment']):
                    frc_data[pfnum][comp] = np.array(
                        [received_data['Force'][start_idx + 3 * i][:], received_data['Force'][start_idx + 3 * i + 1][:],
                         received_data['Force'][start_idx + 3 * i + 2][:]])

            angle_data = {}
            # Remplir le dictionnaire avec les données extraites
            for key, indices in self.dofcorr.items():
                angle_data[f'{key}'] = np.array([[received_data['Angle'][0]], [received_data['Angle'][1]],
                                                 [received_data['Angle'][2]]])

            received_data = {"Force": frc_data, "Markers": mks_data, "Angle": angle_data}
            del frc_data, mks_data, angle_data
            # Traiter les données en parallèle pour ne pas bloquer la réception
            self.process_data(received_data)

            loop_time = time.time() - tic
            #print(loop_time)
            real_time_to_sleep = max(0, int(1 / 200 - loop_time))
            time.sleep(real_time_to_sleep)

    def process_data(self, received_data):
        """Traitement des données et vérification du cycle."""

        self.check_cycle(received_data)
        self.recursive_concat(self.datacycle, received_data)

    def check_cycle(self, received_data):
        """Vérifie si un nouveau cycle doit commencer."""
        # Calcul de la force verticale moyenne pour vérifier la condition du cycle
        vertical_force_mean = np.mean(received_data['Force'][1]['Force'][2, :])

        if 'Force' in self.datacycle and len(self.datacycle['Force'][1]['Force'][2, :]) > 10:
            last_vertical_force_mean = np.mean(self.datacycle['Force'][1]['Force'][2, -10:])
        else:
            last_vertical_force_mean = 0

        # Si la force verticale dépasse le seuil, détecter le début d'un nouveau cycle
        if vertical_force_mean > self.threshold > last_vertical_force_mean != 0:
            # Détecte le début d'un nouveau cycle, envoie les données au processeur
            cycle_to_process = self.datacycle
            self.datacycle = {}
            self.current_frame = 0
            print("Les données ont été réinitialisées pour un nouveau cycle.")
            tic = time.time()
            self.processor.start_new_cycle(cycle_to_process)  # Lance le traitement du cycle
            print(time.time()-tic)

    def recursive_concat(self, datacycle, received_data):
        """Concatène récursivement les données reçues dans datacycle."""
        for key, value in received_data.items():
            if isinstance(value, dict):
                if key not in datacycle:
                    datacycle[key] = {}
                self.recursive_concat(datacycle[key], value)
            else:
                datacycle[key] = np.hstack((datacycle[key], value)) if key in datacycle else value
