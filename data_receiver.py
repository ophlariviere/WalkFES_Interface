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
    """Classe pour gérer la réception de données via le serveur."""

    def __init__(self, server_ip, server_port, read_frequency=100, threshold=30):
        super().__init__()
        self.server_ip = "127.0.0.1"
        self.server_port = 50000
        self.tcp_client = TcpClient(self.server_ip, self.server_port, read_frequency=100)
        self.threshold = threshold  # Seuil pour détecter un nouveau cycle
        self.current_frame = 0  # Ajout de current_frame initialisation

    def connect_to_server(self):
        """Connexion au serveur."""
        #self.tcp_client.connect()  # Connexion au serveur
        print(f"Connecté au serveur {self.server_ip}:{self.server_port}")


    def start_receiving(self):
        """Commence à recevoir les données en continu."""
        self.connect_to_server()
        print("Début de la réception des données...")
        self.current_frame = 0
        while True:
            # Récupère les données une seule fois
            received_data = self.tcp_client.get_data_from_server(command=['Force','Markers','Angle'])

            if received_data and self.current_frame != 0:
                # Crée et démarre les threads de traitement
                thread_stimulation = threading.Thread(target=self.check_stimulation, args=(received_data,))
                thread_cycle_detection = threading.Thread(target=self.check_cycle, args=(received_data,))

                thread_stimulation.start()
                thread_cycle_detection.start()

                thread_stimulation.join()
                thread_cycle_detection.join()
                
            elif received_data and self.current_frame == 0:
                self.data = received_data
                self.current_frame = self.current_frame + 1
            time.sleep(0.01)  # Pause pour contrôler la fréquence de lecture

    def check_stimulation(self, received_data):
        """Vérifie la condition pour l'envoi de stimulation."""
        ap_force = np.mean(received_data['Force'][0])  # Force antéro-postérieure Fx
        last_ap_force = np.mean(self.data['Force'][0][-10:])

        # Envoi de stimulation si la force antéro-postérieure est positive
        if ap_force > 5 >= last_ap_force:
            #self.send_stimulation()
            print("Appel fonction stim")

    def check_cycle(self, received_data):
        """Vérifie la condition pour démarrer un nouveau cycle."""
        vertical_force_mean = np.mean(received_data['Force'][2])
        last_vertical_force_mean = np.mean(self.data['Force'][2][-10:])
        # Vérification du cycle
        print(last_vertical_force_mean, vertical_force_mean)
        if vertical_force_mean > self.threshold > last_vertical_force_mean:
            # Détecte le début d'un nouveau cycle, le traitement des données se déclenche
            DataProcessor.start_new_cycle(self) # Appelle la méthode de traitement du cycle
            self.reset_data()  # Réinitialise les données pour le prochain cycle
            self.current_frame = 0
        else:
            # Ajoute les données reçues au dictionnaire `self.data`
            for key, new_arrays in received_data.items():
                if key in self.data:
                    # Pour chaque liste d'ndarray dans self.data et dico2, concaténer les arrays correspondants
                    updated_list = [
                        np.concatenate((old_array, new_array))
                        for old_array, new_array in zip(self.data[key], new_arrays)
                    ]
                    # Ajouter des nouveaux ndarray à la liste si dico2 en a plus
                    if len(new_arrays) > len(self.data[key]):
                        updated_list.extend(new_arrays[len(self.data[key]):])
                    self.data[key] = updated_list
                else:
                    self.data[key] = new_arrays
            self.current_frame = self.current_frame + 1

    def send_stimulation(self):
        self.stimulator.start_stimulation(upd_list_channels=self.list_channels, safety=True)
        print("Stimulation envoyée.")

    def reset_data(self):
        """Réinitialise les données pour un nouveau cycle."""
        self.data = {}
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
    data_receiver = DataReceiver(server_ip, server_port)
    data_receiver.start_receiving()
    sys.exit(app.exec_())