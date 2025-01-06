from PyQt5.Qt import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QPushButton,
    QWidget,
    QGroupBox,
    QSpinBox,
    QLabel,
    QFileDialog,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import logging
import sys
import biorbd
import pysciencemode
from pysciencemode import Device, Modes, Channel
from pysciencemode import RehastimP24 as St


# Configurer le logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class VisualizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Interface Stimulation"
        self.channels = []
        self.channel_inputs = {}
        self.DataToPlot = self.initialize_data_to_plot()
        self.stimConfigValue = 0
        self.model = None
        self.dolookneedsendstim=False
        self.stimulator = None
        self.stimconfig = {}  # Initialisation correcte ici
        self.HadAnNewStimConfig = False
        self.init_ui()

    def init_ui(self):
        """Initialisation de l'interface utilisateur."""
        self.setWindowTitle(self.title)
        layout = QVBoxLayout(self)

        # Téléchargement du modèle
        layout.addWidget(self.create_model_group())

        # Gestion de l'enregistrement des données
        layout.addWidget(self.create_save_group())

        # Configuration des canaux de stimulation
        layout.addWidget(self.create_channel_config_group())

        # Contrôles de stimulation
        layout.addLayout(self.create_stimulation_controls())
        layout.addLayout(self.create_optimization_mode())

        # Sélection des données à analyser
        layout.addWidget(self.create_analysis_group())

        # Zone graphique
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    @staticmethod
    def initialize_data_to_plot():
        """Initialise le dictionnaire des données à tracer."""
        keys = [
            "Force_1", "Force_2",
            "Tau_LHip", "Tau_LKnee", "Tau_LAnkle",
            "q_LHip", "q_LKnee", "q_LAnkle",
            "PropulsionDuration_R", 'StanceDuration_R', 'Cycleduration'
        ]
        return {key: {} for key in keys}

    """Model part"""
    def create_model_group(self):
        """Créer un groupbox pour télécharger le modèle."""
        groupbox = QGroupBox("Téléchargement du modèle")
        layout = QHBoxLayout()

        self.button = QPushButton("Choisir un fichier", self)
        self.button.clicked.connect(self.open_filename_dialog)
        self.label = QLabel("Aucun fichier sélectionné", self)

        layout.addWidget(self.button)
        layout.addWidget(self.label)
        groupbox.setLayout(layout)
        return groupbox

    """Save part"""
    def create_save_group(self):
        """Créer un groupbox pour la gestion de l'enregistrement des données."""
        groupbox = QGroupBox("Gestion enregistrement des données")
        layout = QHBoxLayout()

        self.nom_input = QLineEdit(self)
        self.save_button = QPushButton("Enregistrer", self)
        self.save_button.clicked.connect(self.save_path)

        layout.addWidget(QLabel("Nom pour enregistrer les données:"))
        layout.addWidget(self.nom_input)
        layout.addWidget(self.save_button)
        groupbox.setLayout(layout)
        return groupbox

    """Visu Stim"""
    def create_channel_config_group(self):
        """Créer un groupbox pour configurer les canaux."""
        groupbox = QGroupBox("Configurer les canaux")
        layout = QVBoxLayout()  # Ce layout contiendra les widgets pour les canaux

        # Ajouter les cases à cocher pour sélectionner les canaux
        self.checkboxes = []
        checkbox_layout = QHBoxLayout()
        for i in range(1, 9):
            checkbox = QCheckBox(f"Canal {i}")
            checkbox.stateChanged.connect(self.update_channel_inputs)
            checkbox_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        layout.addLayout(checkbox_layout)

        # Ajouter un layout vertical pour les entrées dynamiques des canaux
        self.channel_config_layout = QVBoxLayout()
        layout.addLayout(self.channel_config_layout)

        groupbox.setLayout(layout)
        return groupbox

    def create_stimulation_controls(self):
        """Créer les boutons pour contrôler la stimulation."""
        layout = QHBoxLayout()
        self.activate_button = QPushButton("Activer Stimulateur")
        self.activate_button.clicked.connect(self.activate_stimulateur)
        self.update_button = QPushButton("Actualiser Paramètre Stim")
        self.update_button.clicked.connect(self.update_stimulation)
        self.start_button = QPushButton("Démarrer Stimulation")
        self.start_button.clicked.connect(self.start_stimulation)
        self.stop_button = QPushButton("Arrêter Stimuleur")
        self.stop_button.clicked.connect(self.stop_stimulation)
        self.checkpauseStim = QCheckBox("Stop tying send stim", self)
        self.checkpauseStim.setChecked(True)
        self.checkpauseStim.stateChanged.connect(self.pausefonctiontosendstim)
        layout.addWidget(self.checkpauseStim)
        layout.addWidget(self.activate_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.update_button)
        layout.addWidget(self.stop_button)
        return layout


    def create_optimization_mode(self):
        """Créer les boutons pour choisir si la stimulation est en mode manuel ou optimisé."""
        layout = QHBoxLayout()
        label = QLabel('Stimulation parameter mode:', self)
        self.is_manual_mode = QRadioButton("Manual", self)
        self.is_manual_mode.setChecked(True)
        self.is_manual_mode.toggled.connect(self.update)
        self.is_bayesian_mode = QRadioButton("Bayesian optimization", self)
        self.is_bayesian_mode.toggled.connect(self.update)
        self.is_ilc_mode = QRadioButton("Iterative learning control", self)
        self.is_ilc_mode.toggled.connect(self.update)

        layout.addWidget(label)
        layout.addWidget(self.is_manual_mode)
        layout.addWidget(self.is_bayesian_mode)
        layout.addWidget(self.is_ilc_mode)
        return layout

    """Plot part"""
    def create_analysis_group(self):
        """Créer un groupbox pour la sélection des analyses."""
        groupbox = QGroupBox("Sélections d'Analyse")
        layout = QHBoxLayout()

        self.checkboxes_graphs = {}
        for key in self.DataToPlot.keys():
            checkbox = QCheckBox(key, self)
            checkbox.stateChanged.connect(self.update_graphs)
            layout.addWidget(checkbox)
            self.checkboxes_graphs[key] = checkbox

        groupbox.setLayout(layout)
        return groupbox

    def open_filename_dialog(self):
        # Ouvre la boîte de dialogue pour sélectionner un fichier
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier", "",
                                                   "Tous les fichiers (*);;Fichiers texte (*.txt)", options=options)
        if file_name:
            # Affiche le nom du fichier dans l'étiquette
            self.label.setText(f"Fichier sélectionné : {file_name}")
            self.model = biorbd.Model(file_name)

    def save_path(self):
        self.path_to_saveData = self.nom_input.text()
        print(f"Chemin d'enregistrement défini sur : {self.path_to_saveData}")

    def new_stim_config(self):
        self.HadAnNewStimConfig = True

    def update_channel_inputs(self):
        """Met à jour les entrées des canaux sélectionnés sous les cases à cocher."""
        selected_channels = [
            i + 1 for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()
        ]

        # Ajouter les nouveaux canaux sélectionnés
        for channel in selected_channels:
            if channel not in self.channel_inputs:
                # Layout pour les entrées du canal
                channel_layout = QHBoxLayout()

                # Création des widgets d'entrée pour le canal
                name_input = QLineEdit()
                name_input.setPlaceholderText(f"Canal {channel} - Nom")
                amplitude_input = QSpinBox()
                amplitude_input.setRange(0, 100)
                amplitude_input.setSuffix(" mA")
                pulse_width_input = QSpinBox()
                pulse_width_input.setRange(0, 1000)
                pulse_width_input.setSuffix(" µs")
                frequency_input = QSpinBox()
                frequency_input.setRange(0, 200)
                frequency_input.setSuffix(" Hz")
                mode_input = QComboBox()
                mode_input.addItems(["SINGLE", "DOUBLET", "TRIPLET"])

                # Ajouter les widgets au layout
                channel_layout.addWidget(QLabel(f"Canal {channel}:"))
                channel_layout.addWidget(name_input)
                channel_layout.addWidget(amplitude_input)
                channel_layout.addWidget(pulse_width_input)
                channel_layout.addWidget(frequency_input)
                channel_layout.addWidget(mode_input)

                # Ajouter le layout dans l'affichage des paramètres des canaux
                self.channel_config_layout.addLayout(channel_layout)

                # Enregistrer les widgets pour le canal sélectionné
                self.channel_inputs[channel] = {
                    "layout": channel_layout,
                    "name_input": name_input,
                    "amplitude_input": amplitude_input,
                    "pulse_width_input": pulse_width_input,
                    "frequency_input": frequency_input,
                    "mode_input": mode_input,
                }

        # Supprimer les canaux désélectionnés
        for channel in list(self.channel_inputs.keys()):
            if channel not in selected_channels:
                inputs = self.channel_inputs.pop(channel)
                layout = inputs["layout"]
                # Supprimer les widgets du layout
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                # Supprimer le layout lui-même
                self.channel_config_layout.removeItem(layout)

    def start_stimulation(self, channel_to_send):
        try:
            if self.stimulator is None:
                logging.warning(
                    "Stimulateur non initialisé. Veuillez le configurer avant de démarrer."
                )
                return

            self.channels = []
            for channel, inputs in self.channel_inputs.items():
                if channel in channel_to_send:
                    channel_obj = Channel(
                        no_channel=channel,
                        name=inputs["name_input"].text(),
                        amplitude=inputs["amplitude_input"].value(),
                        pulse_width=inputs["pulse_width_input"].value(),
                        frequency=inputs["frequency_input"].value(),
                        mode=Modes.SINGLE,
                        device_type=Device.Rehastimp24,
                    )
                else:
                    channel_obj = Channel(
                        no_channel=channel,
                        name=inputs["name_input"].text(),
                        amplitude=0,
                        pulse_width=inputs["pulse_width_input"].value(),
                        frequency=inputs["frequency_input"].value(),
                        mode=Modes.SINGLE,
                        device_type=Device.Rehastimp24,
                    )
                self.channels.append(channel_obj)
            if self.channels:
                self.stimulator.init_stimulation(list_channels=self.channels)
                self.stimulator.start_stimulation(
                    upd_list_channels=self.channels, safety=True
                )
            logging.info(f"Stimulation démarrée sur les canaux {channel_to_send}")
        except Exception as e:
            logging.error(f"Erreur lors du démarrage de la stimulation : {e}")

    def stop_stimulation(self):
        try:
            if self.stimulator:
                # self.pause_stimulation()
                self.stimulator.end_stimulation()
                self.stimulator.close_port()
                self.stimulator_is_active = False
                self.stimulator = None
                logging.info("Stimulateur arrêtée.")
            else:
                logging.warning("Aucun stimulateur actif à arrêter.")
        except Exception as e:
            logging.error(f"Erreur lors de l'arrêt de la stimulation : {e}")

    def pause_stimulation(self):
        try:
            if self.stimulator:
                self.stimulator.end_stimulation()
                logging.info("Stimulation arrêtée.")
            else:
                logging.warning("Aucun stimulateur actif à arrêter.")
        except Exception as e:
            logging.error(f"Erreur lors de l'arrêt de la stimulation : {e}")

    def pausefonctiontosendstim(self):
        # Met à jour self.dolookstimsend selon l'état de la checkbox
        self.dolookneedsendstim = not self.checkpauseStim.isChecked()

    def activate_stimulateur(self):
        if self.stimulator is None:
            self.stimulator = St(port="COM3", show_log="Status")
            self.stimulator_is_active = True

        self.channels = []
        for channel, inputs in self.channel_inputs.items():
            channel_obj = Channel(
                no_channel=channel,
                name=inputs["name_input"].text(),
                amplitude=inputs["amplitude_input"].value(),
                pulse_width=inputs["pulse_width_input"].value(),
                frequency=inputs["frequency_input"].value(),
                mode=Modes.SINGLE, #inputs["mode_input"].currentText(),
                device_type=Device.Rehastimp24,
            )

            self.channels.append(channel_obj)
        if self.channels:
            self.stimulator.init_stimulation(list_channels=self.channels)



    def update_stimulation(self):
        """Met à jour la stimulation."""
        self.stimConfigValue += 1

        if self.stimulator is not None:
            for channel, inputs in self.channel_inputs.items():
                # Vérifiez si le canal existe dans stimconfig, sinon, initialisez-le
                if channel not in self.stimconfig:
                    self.stimconfig[channel] = {
                        "name": "",
                        "amplitude": 0,
                        "pulse_width": 0,
                        "frequency": 0,
                        "mode": "",
                        "device_type": None,
                    }

                # Mettez à jour les valeurs de configuration
                self.stimconfig[channel]["name"] = inputs["name_input"].text()
                self.stimconfig[channel]["amplitude"] = inputs["amplitude_input"].value()
                self.stimconfig[channel]["pulse_width"] = inputs["pulse_width_input"].value()
                self.stimconfig[channel]["frequency"] = inputs["frequency_input"].value()
                self.stimconfig[channel]["mode"] = inputs["mode_input"].currentText()
                self.stimconfig[channel]["device_type"] = Device.Rehastimp24





    def update_data_and_graphs(self, new_data):
        """
        Met à jour les données et actualise les graphiques.
        - new_data: Nouvelles données du cycle actuel (peut être une valeur unitaire ou un vecteur)
        """
        """
        for key in self.DataToPlot:
            #while len(self.DataToPlot[key]) <= self.stimConfigValue:
                #self.DataToPlot[key].append([])
            self.DataToPlot[key].append(self.get_value_iterative(new_data, key))
        self.DataToPlotConfigNum.append(self.stimConfigValue)
        """
        # Parcours des clés de self.DataToPlot
        for key in self.DataToPlot.keys():
            # Initialiser la configuration de stimulation dans le dictionnaire si elle n'existe pas
            if self.stimConfigValue not in self.DataToPlot[key]:
                self.DataToPlot[key][self.stimConfigValue] = []

            # Vérifier si la nouvelle donnée contient la clé
            value = self.get_value_iterative(new_data, key)

            # Vérifier si la valeur est numérique et l'ajouter directement
            if isinstance(value, (np.ndarray, list)):  # Check for array-like objects
                value = np.array(value)  # Ensure it's a NumPy array if it's not already
                if value.ndim == 2:  # Check the number of dimensions
                    if 'Force' in key or 'Moment' in key:
                        interpolated_vector = self.interpolate_vector(value[0, :])  # TODO change to select axis
                        self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector)
                    else:
                        interpolated_vector = self.interpolate_vector(value[1, :])
                        if 'Tau' in key:
                            self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector / 58)
                        else:
                            self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector * 180 / 3.14)
                else:
                    self.DataToPlot[key][self.stimConfigValue].append(value)
            elif isinstance(value, (int, float)):  # Handle scalar numbers separately
                self.DataToPlot[key][self.stimConfigValue].append(value)

        self.update_graphs()


    def choose_parameters_bayesian_optim(self, new_data):
        """
        Utilise les données reçues pour choisir les paramètres de stimulation à l'aide d'une optimisation Bayesienne.
        - new_data: Nouvelles données du cycle actuel (peut être une valeur unitaire ou un vecteur)
        """

        # Parcours des clés de self.DataToPlot
        for key in self.DataToPlot.keys():
            # Initialiser la configuration de stimulation dans le dictionnaire si elle n'existe pas
            if self.stimConfigValue not in self.DataToPlot[key]:
                self.DataToPlot[key][self.stimConfigValue] = []

            # Vérifier si la nouvelle donnée contient la clé
            value = self.get_value_iterative(new_data, key)

            # Vérifier si la valeur est numérique et l'ajouter directement
            if isinstance(value, (np.ndarray, list)):  # Check for array-like objects
                value = np.array(value)  # Ensure it's a NumPy array if it's not already
                if value.ndim == 2:  # Check the number of dimensions
                    if 'Force' in key or 'Moment' in key:
                        interpolated_vector = self.interpolate_vector(value[0, :])  # TODO change to select axis
                        self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector)
                    else:
                        interpolated_vector = self.interpolate_vector(value[0, :])
                        if 'Tau' in key:
                            self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector / 58)
                        else:
                            self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector * 180 / 3.14)
                else:
                    self.DataToPlot[key][self.stimConfigValue].append(value)
            elif isinstance(value, (int, float)):  # Handle scalar numbers separately
                self.DataToPlot[key][self.stimConfigValue].append(value)

        self.update_graphs()


    @staticmethod
    def interpolate_vector(vector):
        # Vérification que la taille du vecteur est correcte avant d'interpoler
        if len(vector) == 0:
            logging.error("Le vecteur d'entrée est vide pour l'interpolation.")
            return np.zeros(100)  # Retourner un vecteur nul si le vecteur est vide

        x = np.linspace(0, 1, len(vector))
        x_new = np.linspace(0, 1, 100)
        function_interpolation = interp1d(x, vector, kind='linear')
        interpolated_vector = function_interpolation(x_new)
        return interpolated_vector

    def update_graphs(self):
        """Updates displayed graphs based on selected checkboxes."""
        self.figure.clear()

        # Check selected graphs
        graphs_to_display = {key: checkbox.isChecked() for key, checkbox in self.checkboxes_graphs.items()}
        count = sum(graphs_to_display.values())

        if count == 0:
            # Nothing to display
            self.canvas.draw()
            return

        # Calculate layout for subplots
        rows = (count + 1) // 2
        cols = 2 if count > 1 else 1
        subplot_index = 1

        # Affichage des graphiques en fonction des cases à cocher
        for key, is_checked in graphs_to_display.items():
            if is_checked:
                # Ajouter un sous-graphe pour chaque graphique sélectionné
                ax = self.figure.add_subplot(rows, cols, subplot_index)
                data_to_plot = self.DataToPlot[key]
                if any(len(values) > 0 for values in data_to_plot.values()):
                    if all(isinstance(values[0], (int, float)) for values in data_to_plot.values() if len(values) > 0):
                        self.plot_numeric_data(ax, key, key)
                    else:
                        self.plot_vector_data(ax, key, key)

                subplot_index += 1

        # Redessiner le canevas pour afficher les nouvelles données
        self.canvas.draw()


    @staticmethod
    def get_value_iterative(d, key_to_find):
        """Retourne la valeur associée à une clé dans un dictionnaire imbriqué, en utilisant une approche itérative."""
        stack = [d]
        while stack:
            current_dict = stack.pop()
            if not isinstance(current_dict, dict):
                continue
            if key_to_find in current_dict:
                return current_dict[key_to_find]
            for value in current_dict.values():
                if isinstance(value, dict):
                    stack.append(value)
        return None

    def plot_numeric_data(self, ax, key, ylabel):
        # Tracer les valeurs numériques (par exemple, Cycleduration, Cadence) sur le sous-graphe 'ax'
        for stim_config, values in self.DataToPlot[key].items():
            cycles = list(range(1, len(values) + 1))
            ax.plot(cycles, values, marker='o', label=f'Stim: {stim_config}')

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} Over Cycles')
        ax.legend()

    def plot_vector_data(self, ax, key, ylabel):
        # Tracer les vecteurs interpolés (par exemple, RHip, RAnkle) sur le sous-graphe 'ax'
        x_percentage = np.linspace(0, 100, 100)
        for stim_config, cycle_data in self.DataToPlot[key].items():
            cycle_data = np.array(cycle_data)  # Convert the list of cycles to a NumPy array
            mean_vector = np.mean(cycle_data, axis=0)
            std_vector = np.std(cycle_data, axis=0)

            ax.plot(x_percentage, mean_vector, label=f'Stim: {stim_config}')
            ax.fill_between(x_percentage, mean_vector - std_vector, mean_vector + std_vector, alpha=0.2)
        ax.set_xlabel('Percentage of Cycle')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} by Stimulation Configuration')
        ax.legend()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = VisualizationWidget()
    widget.show()
    sys.exit(app.exec_())
