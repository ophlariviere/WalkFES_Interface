import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QCheckBox, QGroupBox, QComboBox, QHBoxLayout, QSizePolicy, QFileDialog, QApplication
from PyQt5.Qt import Qt
import logging
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyScienceMode import Channel, Device, Modes
from pyScienceMode import RehastimP24 as St
import biorbd

# Configure le logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        # Assurez-vous que self.channel_1 est défini si vous l'utilisez.
        self.save_button = None
        self.nom_input = None
        self.button = None
        self.stimConfigValue = None
        self.label = None
        self.stimulator_is_active = False
        self.title = 'Interface Stim'
        self.channels = []  # Store the channel objects
        self.channel_inputs = []  #
        self.DataToPlot = {
            'Force_1': {},
            'LHip': {},
            'LKnee': {},
            'LAnkle': {},
            'Tau_LHip': {},
            'Tau_LKnee': {},
            'Tau_LAnkle': {},
            'q_LHip': {},
            'q_LKnee': {},
            'q_LAnkle': {},
            'qdot_LHip': {},
            'qdot_LKnee': {},
            'qdot_LAnkle': {},
            'PropulsionDuration_L': {},
        }
        self.DataToPlotConfigNum = []  # Initialiser la liste pour stocker les numéros de configuration
        self.init_ui()
    
    def init_ui(self):
        
        self.stimConfigValue = 0
        self.setWindowTitle(self.title)

        # Layout principal
        layout = QVBoxLayout()

        # Selection du model
        model_layout = QHBoxLayout()
        self.button = QPushButton('Choisir un fichier', self)
        self.button.clicked.connect(self.open_filename_dialog)
        self.label = QLabel('Aucun fichier sélectionné', self)
        model_layout.addWidget(self.button)
        model_layout.addWidget(self.label)
        # GroupBox pour l'enregistrement des données
        groupbox_model = QGroupBox("Téléchargement du modèle:")
        groupbox_model.setLayout(model_layout)
        groupbox_model.setMaximumHeight(80)
        groupbox_model.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addWidget(groupbox_model)

        # Section d'enregistrement des données
        save_layout = QHBoxLayout()
        self.nom_input = QLineEdit(self)
        save_layout.addWidget(QLabel('Nom pour enregistrer les données:'))
        save_layout.addWidget(self.nom_input)
        self.save_button = QPushButton('Validate', self)
        self.save_button.clicked.connect(self.save_path)
        save_layout.addWidget(self.save_button)
        # GroupBox pour l'enregistrement des données
        groupbox_save = QGroupBox("Gestion enregistrement données:")
        groupbox_save.setLayout(save_layout)
        groupbox_save.setMaximumHeight(80)
        groupbox_save.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addWidget(groupbox_save)

        # Section des paramètres de stimulation
        form_layout = QVBoxLayout()  # Changed to QVBoxLayout for better organization

        # Ajouter le QComboBox pour sélectionner le nombre de canaux
        stimchannum_layout = QHBoxLayout()
        self.num_channels_combo = QComboBox(self)
        self.num_channels_combo.addItems([str(i) for i in range(1, 5)])  # Allow selection between 1 and 4 channels
        self.num_channels_combo.currentIndexChanged.connect(self.update_channel_inputs)
        stimchannum_layout.addWidget(QLabel('Nombre de canaux:'))
        stimchannum_layout.addWidget(self.num_channels_combo)
        form_layout.addLayout(stimchannum_layout)
        # Initialize the groupbox_params here before calling update_channel_inputs
        self.groupbox_params = QGroupBox("Paramètres de Stimulation")
        self.groupbox_params.setLayout(form_layout)
        self.groupbox_params.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addWidget(self.groupbox_params)

        # Now call the method to populate initial channel inputs
        self.update_channel_inputs()

        self.stimulator_checkbox = QCheckBox('Activer stimulateur', self)
        self.stimulator_checkbox.stateChanged.connect(self.on_stimulator_checkbox_toggled)
        form_layout.addWidget(self.stimulator_checkbox)

        # Ajouter le bouton 'Actualiser stim' sous les paramètres de canaux
        self.ActuStim_button = QPushButton('Actualiser stim', self)
        self.ActuStim_button.clicked.connect(self.stim_actu_clicked)
        form_layout.addWidget(self.ActuStim_button)

        # Sélections d'analyse
        analysis_layout = QHBoxLayout()  # Correction ici : initialisation du layout d'analyse
        self.checkboxes = {}
        for key in self.DataToPlot.keys():
            checkbox = QCheckBox(key, self)
            checkbox.stateChanged.connect(self.update_graphs)
            analysis_layout.addWidget(checkbox)
            self.checkboxes[key] = checkbox

        analysis_groupbox = QGroupBox("Sélections d'Analyse")
        analysis_groupbox.setLayout(analysis_layout)
        analysis_groupbox.setMaximumHeight(80)
        analysis_groupbox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addWidget(analysis_groupbox)

        # Zone graphique
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def update_channel_inputs(self):
        # Si c'est la première fois, créer un layout pour les channels
        if not hasattr(self, 'channels_layout'):
            self.channels_layout = QVBoxLayout()  # Un layout séparé pour les inputs de channels
            self.groupbox_params.layout().addLayout(self.channels_layout)  # Ajouter au layout principal

        # Obtenez le nombre de canaux sélectionnés
        num_channels = int(self.num_channels_combo.currentText())

        # Si le nombre de canaux augmente, ajouter seulement les nouveaux
        while len(self.channel_inputs) < num_channels:
            channel_index = len(self.channel_inputs)  # L'indice du nouveau canal à ajouter
            channel_layout = QHBoxLayout()  # Créez un layout horizontal pour chaque canal

            inputs = {}

            # Ajout des inputs dans le layout
            inputs['name'] = QLineEdit(self)
            channel_layout.addWidget(QLabel(f'Canal {channel_index + 1} - Nom:'))
            channel_layout.addWidget(inputs['name'])

            inputs['amplitude'] = QLineEdit(self)
            channel_layout.addWidget(QLabel('Amplitude [mA]:'))
            channel_layout.addWidget(inputs['amplitude'])

            inputs['frequence'] = QLineEdit(self)
            channel_layout.addWidget(QLabel('Fréquence [Hz]:'))
            channel_layout.addWidget(inputs['frequence'])

            inputs['duree'] = QLineEdit(self)
            channel_layout.addWidget(QLabel('Durée [ms]:'))
            channel_layout.addWidget(inputs['duree'])

            inputs['largeur'] = QLineEdit(self)
            channel_layout.addWidget(QLabel('Largeur [µs]:'))
            channel_layout.addWidget(inputs['largeur'])

            inputs['mode'] = QComboBox(self)
            inputs['mode'].addItems(["SINGLE", "DOUBLET", "TRIPLET"])
            channel_layout.addWidget(QLabel('Mode:'))
            channel_layout.addWidget(inputs['mode'])

            # Ajoutez ce layout de canal au layout dédié
            self.channels_layout.addLayout(channel_layout)

            # Enregistrer les inputs pour un usage futur
            self.channel_inputs.append(inputs)

        # Si le nombre de canaux diminue, supprimer les canaux en excès
        while len(self.channel_inputs) > num_channels:
            inputs_to_remove = self.channel_inputs.pop()  # Supprime les inputs en excès
            for widget in inputs_to_remove.values():
                widget.setParent(None)  # Supprime le widget de l'interface sans le détruire immédiatement
                widget.deleteLater()  # Détruit le widget proprement

        # Nettoyage du layout : s'assurer que tous les widgets sont bien alignés
        for i in reversed(range(self.channels_layout.count())):
            layout_item = self.channels_layout.itemAt(i)
            if isinstance(layout_item, QHBoxLayout):  # Vérifier si c'est un layout de canal
                if i >= num_channels:  # S'il y a plus de layouts que nécessaire, on supprime
                    for j in reversed(range(layout_item.count())):
                        item = layout_item.itemAt(j)
                        if item.widget() is not None:
                            item.widget().deleteLater()
                    self.channels_layout.removeItem(layout_item)

        # Optionnel : Si vous voulez rafraîchir les labels des canaux après modification
        for channel_index, inputs in enumerate(self.channel_inputs):
            inputs['name'].setPlaceholderText(f'Canal {channel_index + 1} - Nom:')

    def on_stimulator_checkbox_toggled(self, state):
        if state == Qt.Checked:
            logging.info("Stimulator activated.")
            self.activer_stimulateur()
        else:
            logging.info("Stimulator deactivated.")
            self.desactiver_stimulateur()


    def activer_stimulateur(self):
        try:
            self.channels = []  # Clear previous channels
            num_channels = int(self.num_channels_combo.currentText())

            for i in range(num_channels):
                inputs = self.channel_inputs[i]
                channel = Channel(
                    mode="Single",
                    no_channel=i + 1,
                    amplitude=int(inputs['amplitude'].text()),
                    pulse_width=int(inputs['largeur'].text()),
                    frequency=int(inputs['frequence'].text()),
                    name=inputs['name'].text(),
                    device_type=Device.Rehastimp24
                )
                self.channels.append(channel)

            self.stimulator = St(port="COM3")
            self.stimulator.init_stimulation(list_channels=self.channels)
            self.stimulator_is_active = True
            print("Fonction d'activation du stimulateur appelée pour plusieurs canaux.")

        except Exception as e:
            print(f"Erreur lors de l'activation du stimulateur : {e}")
        self.stimConfigValue = 0

    def send_stimulation(self):
        if self.stimulator_is_active:
            self.stimulator.start_stimulation(upd_list_channels=self.channels,  stimulation_duration=self.channel_inputs[0]['duree'], safety=True)
            print("Stimulation envoyée.")
        else :
            print("Stim desactivé")

    def desactiver_stimulateur(self):
        if hasattr(self, 'stimulator') and self.stimulator:
            self.stimulator.end_stimulation()
            self.stimulator.close_port()
            self.stimulator_is_active = False
            print("Fonction de désactivation du stimulateur appelée.")


    def stim_actu_clicked(self):
        try:
            num_channels = int(self.num_channels_combo.currentText())

            for i in range(num_channels):
                inputs = self.channel_inputs[i]
                amplitude = int(inputs['amplitude'].text())
                frequence = int(inputs['frequence'].text())
                duree = int(inputs['duree'].text())
                largeur = int(inputs['largeur'].text())
                mode_text = inputs['mode'].currentText()
                mode = Modes.SINGLE if mode_text == "SINGLE" else Modes.DOUBLET if mode_text == "DOUBLET" else Modes.TRIPLET

                # Update channel parameters
                self.channels[i].set_amplitude(amplitude)
                self.channels[i].set_pulse_width(largeur)
                self.channels[i].set_frequency(frequence)
                self.channels[i].set_mode(mode)
                self.channel1_duree = duree

            print("Stimulation mise à jour pour tous les canaux.")

        except ValueError:
            print("Erreur : Veuillez entrer des valeurs numériques valides.")
        except Exception as e:
            print(f"Erreur lors de l'actualisation de la stimulation : {e}")
        self.stimConfigValue += 1

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
                        interpolated_vector = self.interpolate_vector(value[2, :]) #TODO change to select axis
                        self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector)
                    else:
                        interpolated_vector = self.interpolate_vector(value[1, :])
                        if 'Tau' in key:
                            self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector/58)
                        else:
                            self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector*180/3.14)
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
        fonction_interpolation = interp1d(x, vector, kind='linear')
        interpolated_vector = fonction_interpolation(x_new)
        return interpolated_vector

    def update_graphs(self):
        self.figure.clear()

        # Vérification des checkboxes pour déterminer les graphiques à afficher
        graphs_to_display = {key: checkbox.isChecked() for key, checkbox in self.checkboxes.items()}
        count = sum(graphs_to_display.values())  # Compter combien de graphes afficher

        if count == 0:
            return  # Aucun graphique à afficher, on ne fait rien

        # Calcul du nombre de lignes et de colonnes pour les subplots
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VisualizationWidget()
    ex.show()
    sys.exit(app.exec_())
