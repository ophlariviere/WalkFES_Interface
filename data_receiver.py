import time
import numpy as np
from biosiglive import TcpClient
from PyQt5.QtCore import QObject
import logging
from data_processor import DataProcessor


class DataReceiver(QObject):
    def __init__(
            self,
            server_ip,
            server_port,
            visualization_widget,
            read_frequency=100,
            threshold=30,
    ):
        super().__init__()
        self.visualization_widget = visualization_widget
        self.server_ip = server_ip
        self.server_port = server_port
        self.tcp_client = TcpClient(
            self.server_ip, self.server_port, read_frequency=read_frequency
        )
        self.threshold = threshold
        self.stimulator = []
        self.datacycle = {}
        self.sendStim = {1: False, 2: False}
        self.timeStim = 0
        self.visualization_widget = visualization_widget
        self.read_frequency = read_frequency
        self.processor = DataProcessor(self.visualization_widget)  # Passez l'objet visualization_widget
        self.dofcorr = {"LHip": (36, 37, 38), "LKnee": (39, 40, 41), "LAnkle": (42, 43, 44),
                        "RHip": (27, 28, 29), "RKnee": (30, 31, 32), "RAnkle": (33, 34, 35),
                        "LShoulder": (18, 19, 20), "LElbow": (21, 22, 23), "LWrist": (24, 25, 26),
                        "RShoulder": (9, 10, 11), "RElbow": (12, 13, 14), "RWrist": (15, 16, 17),
                        "Thorax": (6, 7, 8), "Pelvis": (3, 4, 5)}
        self.marker_names = None

    def start_receiving(self):
        logging.info("Début de la réception des données...")
        while True:
            tic = time.time()
            for _ in range(3):  # Tentatives multiples
                try:
                    received_data = self.tcp_client.get_data_from_server(command=["Force", "Markers"])
                    break  # Si réussi, quittez la boucle
                except Exception as e:
                    logging.warning(f"Tentative échouée : {e}")
                    time.sleep(5)  # Attente avant la prochaine tentative
            else:
                logging.error("Impossible de se connecter après plusieurs tentatives.")
                continue

            if "Force" not in received_data or not received_data["Force"]:
                logging.warning("Aucune donnée reçue depuis le serveur.")
                continue

            # Organisation des données reçues pour PF1 et PF2
            frc_data = {}
            for pfnum in [1, 2]:
                start_idx = (pfnum - 1) * 9  # PF1: 0-8, PF2: 9-17
                for i, comp in enumerate(["Force", "Moment", "CoP"]):
                    key = f"{comp}_{pfnum}"
                    frc_data[key] = np.array(
                        [
                            received_data["Force"][start_idx + 3 * i][:],
                            received_data["Force"][start_idx + 3 * i + 1][:],
                            received_data["Force"][start_idx + 3 * i + 2][:],
                        ]
                    )
            print(np.mean(frc_data['Force_1'][0][:]),np.mean(frc_data['Force_2'][0][:]))

            # Organisation des données reçues
            mks_data = {}

            for i in range(len(received_data['Markers'][0][0, :, 0])):
                name=received_data['Markers'][1][i]
                mks_data[name] = np.array([received_data['Markers'][0][0, i, :], received_data['Markers'][0][1, i, :],
                                        received_data['Markers'][0][2, i, :]])


            received_data = {"Force": frc_data, "Markers": mks_data}

            if (self.visualization_widget.stimulator is not None) and (self.visualization_widget.dolookneedsendstim is True):
                self.check_stimulation(received_data)

            self.process_data(received_data)

            loop_time = time.time() - tic
            real_time_to_sleep = max(0, 1 / self.read_frequency - loop_time)
            time.sleep(real_time_to_sleep)

    def check_stimulation(self, received_data):
        try:
            for PFnum in range(1, 3):
                ap_force_mean, last_ap_force_mean = self._calculate_force_means(received_data, PFnum)
                if self._should_start_stimulation(ap_force_mean, last_ap_force_mean, PFnum):
                    self._start_stimulation(PFnum)
                elif self._should_stop_stimulation(ap_force_mean, last_ap_force_mean, PFnum):
                    self._stop_stimulation()
        except Exception as e:
            logging.error(f"Erreur lors de la stimulation : {e}")

    def _calculate_force_means(self, received_data, PFnum):
        """Calcule les moyennes des forces actuelles et précédentes pour PFnum."""
        print(PFnum)
        ap_force_mean = np.nanmean(received_data["Force"]["Force_" + str(PFnum)][0, :])
        ap_force_mean = -ap_force_mean
        long = len(received_data["Force"]["Force_" + str(PFnum)][0, :])
        last_ap_force_mean = (
            np.nanmean(self.datacycle["Force"]["Force_" + str(PFnum)][0, -long:])
            if "Force" in self.datacycle and len(self.datacycle["Force"]["Force_" + str(PFnum)][0, :]) > 0
            else np.NaN
        )
        if np.isnan(last_ap_force_mean):
            last_ap_force_mean = (
                np.nanmean(self.datacycle["Force"]["Force_" + str(PFnum)][0, -2*long-2:-long-2])
                if "Force" in self.datacycle and len(self.datacycle["Force"]["Force_" + str(PFnum)][0, :]) > 0
                else np.NaN
            )

        last_ap_force_mean = -last_ap_force_mean
        return ap_force_mean, last_ap_force_mean

    def _should_start_stimulation(self, ap_force_mean, last_ap_force_mean,PFnum):
        """Vérifie si la stimulation doit commencer."""
        return (
                #and (ap_force_mean - last_ap_force_mean) > 0
                self.sendStim[PFnum] is False and
                ap_force_mean > 5
                and self.visualization_widget.stimulator is not None
        )

    def _start_stimulation(self, PFnum):
        """Démarre la stimulation pour le canal spécifié."""
        channel_to_stim = [1, 2, 3, 4] if PFnum == 1 else [5, 6, 7, 8]
        self.visualization_widget.start_stimulation(channel_to_stim)
        self.sendStim[PFnum] = True
        self.timeStim = time.time()

    def _should_stop_stimulation(self, ap_force_mean, last_ap_force_mean,PFnum):
        """Vérifie si la stimulation doit s'arrêter."""
        time_since_stim = time.time() - self.timeStim
        return (
                #and  ((ap_force_mean - last_ap_force_mean) < 0
                ((self.sendStim[PFnum] is True)
                 and ap_force_mean < 10
                 and time_since_stim > 0.2)
                or (time_since_stim > 0.8 and self.sendStim[PFnum] is True)
        )

    def _stop_stimulation(self,PFnum):
        """Arrête la stimulation."""
        self.visualization_widget.pause_stimulation()
        self.sendStim[PFnum] = False

    def process_data(self, received_data):
        self.check_cycle(received_data)
        self.recursive_concat(self.datacycle, received_data)

    def check_cycle(self, received_data):
        try:
            vertical_force_mean = np.nanmean(received_data["Force"]["Force_1"][2, :])
            long = len(received_data["Force"]["Force_1"][2, :])
            if (
                    "Force" in self.datacycle
                    and len(self.datacycle["Force"]["Force_1"][2, :]) > 0
            ):
                last_vertical_force_mean = np.nanmean(
                    self.datacycle["Force"]["Force_1"][2, -long:]
                )
            else:
                last_vertical_force_mean = 0

            if vertical_force_mean > self.threshold > last_vertical_force_mean != 0:
                cycle_to_process = self.datacycle
                self.datacycle = {}
                self.current_frame = 0
                logging.info(
                    "Les données ont été réinitialisées pour un nouveau cycle."
                )
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
                    datacycle[key] = (
                        np.hstack((datacycle[key], value))
                        if key in datacycle
                        else value
                    )
                except Exception as e:
                    logging.error(
                        f"Erreur lors de la concaténation des données pour la clé '{key}': {e}"
                    )
