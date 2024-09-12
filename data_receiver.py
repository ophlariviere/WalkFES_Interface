from PyQt5.QtCore import QObject
import numpy as np
import time
import threading
from biosiglive import TcpClient
from PyQt5.QtWidgets import QApplication
import sys
from data_processor import DataProcessor
from visualization import VisualizationWidget


class DataReceiver(QObject):
    def __init__(self, server_ip, server_port, visualization_widget, read_frequency=200, threshold=30):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.tcp_client = TcpClient(self.server_ip, self.server_port, read_frequency=read_frequency)
        self.threshold = threshold
        self.visualization_widget = visualization_widget  # Passer le widget de visualisation
        self.processor = DataProcessor(self.visualization_widget)  # Lier le DataProcessor au widget
        self.datacycle = {}  # Initialiser le dictionnaire pour stocker les données
        self.current_frame = 0  # Initialiser le compteur de frames

    def start_receiving(self):
        print("Début de la réception des données...")
        dofcorr = {"LHip": (36, 37, 38),
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
        self.current_frame = 0
        while True:
            tic=time.time()
            received_data = self.tcp_client.get_data_from_server(command=['Force', 'Markers', 'Angle', 'MarkersNames'])
            # Organisation des données reçues
            Mks_Data = {}
            for i, name in enumerate(received_data['MarkersNames']):
                Mks_Data[name] = np.array([received_data['Markers'][0][i, :], received_data['Markers'][1][i, :],
                                           received_data['Markers'][2][i, :]])

            Frc_Data = {1: {}, 2: {}}  # Pour deux plateformes de force
            for pfnum in [1, 2]:
                start_idx = (pfnum - 1) * 6
                for i, comp in enumerate(['Force', 'Moment']):
                    Frc_Data[pfnum][comp] = np.array(
                        [received_data['Force'][start_idx + 3 * i][:], received_data['Force'][start_idx + 3 * i + 1][:],
                         received_data['Force'][start_idx + 3 * i + 2][:]])

            Angle_Data = {}
            # Remplir le dictionnaire avec les données extraites
            for key, indices in dofcorr.items():
                Angle_Data[f'{key}'] = np.array([[received_data['Angle'][0]], [received_data['Angle'][1]],
                                           [received_data['Angle'][2]]])

            received_data = {"Force": Frc_Data, "Markers": Mks_Data, "Angle": Angle_Data}
            del Frc_Data, Mks_Data, Angle_Data
            if received_data:
                self.check_cycle(received_data)  # Vérifie et traite les cycles avec les données reçues
            loop_time = time.time() - tic
            real_time_to_sleep = 1 / 200 - loop_time
            print(real_time_to_sleep)
            if real_time_to_sleep > 0:
                time.sleep(real_time_to_sleep)


    def check_stimulation(self, received_data):
        """Vérifie la condition pour l'envoi de stimulation."""
        ap_force = np.mean(received_data['Force'][0])  # Force antéro-postérieure Fx
        last_ap_force = np.mean(self.datacycle['Force'][0][-10:])

        # Envoi de stimulation si la force antéro-postérieure est positive
        if ap_force > 5 >= last_ap_force:
            self.send_stimulation()

    def check_cycle(self, received_data):
        """Vérifie la condition pour démarrer un nouveau cycle."""
        vertical_force_mean = np.mean(received_data['Force'][1]['Force'][2, :])
        if 'Force' in self.datacycle and len(self.datacycle['Force'][1]['Force'][2, :]) > 10:
            last_vertical_force_mean = np.mean(self.datacycle['Force'][1]['Force'][2, -10:])
        else:
            last_vertical_force_mean = 0

        # Vérification du cycle
        if vertical_force_mean > self.threshold > last_vertical_force_mean and last_vertical_force_mean != 0:
            # Détecte le début d'un nouveau cycle, le traitement des données se déclenche
            cycletoprecess = self.datacycle
            self.reset_data()  # Réinitialise les données pour le prochain cycle
            self.processor.start_new_cycle(cycletoprecess)  # Utilise l'instance existante de DataProcessor

        self.recursive_concat(self.datacycle, received_data)
        self.current_frame += 1

    def recursive_concat(self, datacycle, received_data):
        """Concatène récursivement les données reçues dans datacycle."""
        # Parcourir tous les éléments de received_data
        for key, value in received_data.items():
            # Si l'élément est un dictionnaire, on entre dans une récursion
            if isinstance(value, dict):
                if key not in datacycle:
                    datacycle[key] = {}  # Si le sous-dictionnaire n'existe pas encore dans datacycle, on l'initialise
                self.recursive_concat(datacycle[key], value)
            else:
                # Si c'est un ndarray, on gère la concaténation (cas général)
                if key in datacycle:
                    datacycle[key] = np.hstack((datacycle[key], value))  # Concaténation horizontale
                else:
                    datacycle[key] = value  # Initialisation si première donnée

    def send_stimulation(self):
        # Appelle la méthode de stimulation (supposée être implémentée ailleurs)
        print("Stimulation envoyée.")

    def reset_data(self):
        """Réinitialise les données pour un nouveau cycle."""
        self.datacycle = {}
        self.current_frame = 0
        print("Les données ont été réinitialisées pour un nouveau cycle.")


# Exemple d'utilisation
if __name__ == "__main__":
    server_ip = "127.0.0.1"
    server_port = 50000
    app = QApplication(sys.argv)
    # Créer une instance du widget et afficher l'interface
    window = VisualizationWidget()
    window.show()
    data_receiver = DataReceiver(server_ip, server_port, visualization_widget=window)  # Passe correctement le widget
    threading.Thread(target=data_receiver.start_receiving, daemon=True).start()
    sys.exit(app.exec_())
