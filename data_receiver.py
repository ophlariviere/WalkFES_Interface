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
    def __init__(self, server_ip, server_port, visualization_widget, read_frequency=100, threshold=30):
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
        self.current_frame = 0
        while True:
            received_data = self.tcp_client.get_data_from_server(command=['Force', 'Markers', 'Angle'])
            if received_data:
                self.check_cycle(received_data)  # Vérifie et traite les cycles avec les données reçues
            time.sleep(0.01)

    def check_stimulation(self, received_data):
        """Vérifie la condition pour l'envoi de stimulation."""
        ap_force = np.mean(received_data['Force'][0])  # Force antéro-postérieure Fx
        last_ap_force = np.mean(self.datacycle['Force'][0][-10:])

        # Envoi de stimulation si la force antéro-postérieure est positive
        if ap_force > 5 >= last_ap_force:
            self.send_stimulation()

    def check_cycle(self, received_data):
        """Vérifie la condition pour démarrer un nouveau cycle."""
        vertical_force_mean = np.mean(received_data['Force'][2])
        if 'Force' in self.datacycle and len(self.datacycle['Force'][0]) > 10:
            last_vertical_force_mean = np.mean(self.datacycle['Force'][2][-10:])
        else:
            last_vertical_force_mean = 0

        # Vérification du cycle
        #print(last_vertical_force_mean, vertical_force_mean)
        if vertical_force_mean > self.threshold > last_vertical_force_mean and last_vertical_force_mean != 0:
            # Détecte le début d'un nouveau cycle, le traitement des données se déclenche
            cycletoprecess=self.datacycle
            self.reset_data()  # Réinitialise les données pour le prochain cycle
            self.processor.start_new_cycle(cycletoprecess)  # Utilise l'instance existante de DataProcessor

        
        for key, value in received_data.items():
            # Si l'élément est un dictionnaire, on entre dans une récursion
            if isinstance(value, dict):
                if key not in self.datacycle:
                    self.datacycle[
                        key] = {}  # Si le sous-dictionnaire n'existe pas encore dans self.datacycle, on l'initialise
                recursive_concat(self.datacycle[key], value)
            else:
                # Si c'est un ndarray, on gère la concaténation (cas général)
                if key in self.datacycle:
                    self.datacycle[key] = np.hstack((self.datacycle[key], value))  # Concaténation horizontale
                else:
                    self.datacycle[key] = value
        self.current_frame += 1

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
    data_receiver.start_receiving()
    sys.exit(app.exec_())
