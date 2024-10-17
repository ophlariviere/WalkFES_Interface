import threading
import time
import numpy as np
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
import logging
from data_processor import DataProcessor

class DataReceiver(QObject):
    def __init__(self, server_ip, server_port, visualization_widget, read_frequency=100, threshold=30):
        super().__init__()
        self.visualization_widget = visualization_widget
        self.server_ip = server_ip
        self.server_port = server_port
        self.tcp_client = TcpClient(self.server_ip, self.server_port, read_frequency=read_frequency)
        self.threshold = threshold
        self.datacycle = {}
        self.stimulator = []
        self.current_frame = 0
        self.lock = threading.Lock()
        self.visualization_widget = visualization_widget
        self.read_frequency = read_frequency
        self.processor = DataProcessor(self.visualization_widget)  # Passez l'objet visualization_widget
        self.dofcorr = {"LHip": (36, 37, 38), "LKnee": (39, 40, 41), "LAnkle": (42, 43, 44),
                        "RHip": (27, 28, 29), "RKnee": (30, 31, 32), "RAnkle": (33, 34, 35),
                        "LShoulder": (18, 19, 20), "LElbow": (21, 22, 23), "LWrist": (24, 25, 26),
                        "RShoulder": (9, 10, 11), "RElbow": (12, 13, 14), "RWrist": (15, 16, 17),
                        "Thorax": (6, 7, 8), "Pelvis": (3, 4, 5)}
        self.marker_names = None
        self.first_time = False

    def start_receiving(self):
        logging.info("Début de la réception des données...")
        while True:
            tic = time.time()
            try:
                if self.first_time:
                    received_data = self.tcp_client.get_data_from_server(command=['Force', 'Markers', 'MarkersNames'])
                    # Stocker les noms des marqueurs et désactiver l'envoi futur des noms
                    if 'MarkersNames' in received_data:
                        self.marker_names = received_data['MarkersNames']
                    self.first_time = False  # Après le premier envoi, ne plus demander les noms des marqueurs
                else:
                    # Commande sans demander à nouveau les noms des marqueurs
                    received_data = self.tcp_client.get_data_from_server(command=['Force', 'Markers'])

                # Organisation des données reçues
                mks_data = {}
                if self.marker_names:
                    for i, name in enumerate(self.marker_names):
                        mks_data[name] = np.array([received_data['Markers'][0][i, :], received_data['Markers'][1][i, :],
                                                received_data['Markers'][2][i, :]])
                else:
                    for i in range(len(received_data['Markers'][0][:, 0])):
                        mks_data[i] = np.array([received_data['Markers'][0][i, :], received_data['Markers'][1][i, :],
                                                received_data['Markers'][2][i, :]])



                frc_data = {}
                for pfnum in [1, 2]:
                    start_idx = (pfnum - 1) * 9
                    for i, comp in enumerate(['Force', 'Moment', 'CoP']):
                        key = f"{comp}_{pfnum}"
                        frc_data[key] = np.array([
                            received_data['Force'][start_idx + 3 * i][:],
                            received_data['Force'][start_idx + 3 * i + 1][:],
                            received_data['Force'][start_idx + 3 * i + 2][:]
                        ])


                received_data = {}
                received_data = {"Force": frc_data, "Markers": mks_data}

                # Utilisation de thread pour traiter les données en parallèle
                thread_stimulation = threading.Thread(target=self.check_stimulation, args=(received_data,))
                thread_datagestion = threading.Thread(target=self.process_data, args=(received_data,))

                thread_stimulation.start()
                thread_datagestion.start()

                thread_stimulation.join()
                thread_datagestion.join()

                loop_time = time.time() - tic
                real_time_to_sleep = max(0, 1 / self.read_frequency - loop_time)
                time.sleep(real_time_to_sleep)

            except Exception as e:
                logging.error(f"Erreur lors de la réception des données : {e}")

    def check_stimulation(self, received_data):
        try:
            ap_force_mean = np.mean(received_data['Force']['Force_1'][0, :])
            long = len(received_data['Force']['Force_1'][0, :])
            if 'Force' in self.datacycle and len(self.datacycle['Force']['Force_1'][0, :]) > 0:
                last_ap_force_mean = np.mean(self.datacycle['Force']['Force_1'][0, -long:])
            else:
                last_ap_force_mean = 0

            # Accédez à stimulator_is_active via self.visualization_widget
            if ap_force_mean > 5 >= last_ap_force_mean:
                if self.visualization_widget.stimulator_is_active:  # Vérifiez si le stimulateur est actif
                    self.visualization_widget.send_stimulation()  # Appeler la méthode send_stimulation de l'instance VisualizationWidget
                    logging.info("Stimulation signal emitted")
        except Exception as e:
            logging.error(f"Erreur lors de la stimulation : {e}")

    def process_data(self, received_data):
        self.check_cycle(received_data)
        with self.lock:
            self.recursive_concat(self.datacycle, received_data)

    def check_cycle(self, received_data):
        try:
            vertical_force_mean = np.mean(received_data['Force']['Force_1'][2, :])
            long = len(received_data['Force']['Force_1'][2, :])
            if 'Force' in self.datacycle and len(self.datacycle['Force']['Force_1'][2, :]) > 0:
                last_vertical_force_mean = np.mean(self.datacycle['Force']['Force_1'][2, -long:])
            else:
                last_vertical_force_mean = 0

            if vertical_force_mean > self.threshold > last_vertical_force_mean != 0:
                cycle_to_process = self.datacycle
                self.datacycle = {}
                self.current_frame = 0
                logging.info("Les données ont été réinitialisées pour un nouveau cycle.")
                self.processor.start_new_cycle(cycle_to_process)

        except Exception as e:
            logging.error(f"Erreur lors de la vérification du cycle : {e}")

    def recursive_concat(self, datacycle, received_data):
        for key, value in received_data.items():
            if isinstance(value, dict):
                if key not in datacycle:
                    datacycle[key] = {}
                self.recursive_concat(datacycle[key], value)
            else:
                try:
                    datacycle[key] = np.hstack((datacycle[key], value)) if key in datacycle else value
                except Exception as e:
                    logging.error(f"Erreur lors de la concaténation des données pour la clé '{key}': {e}")
