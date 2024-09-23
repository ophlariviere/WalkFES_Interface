import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
                             QCheckBox, QGroupBox, QPushButton, QComboBox, QSizePolicy,
                             QFileDialog)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pyScienceMode import Channel, Device, Modes
from pyScienceMode import RehastimP24 as St
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure le logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class VisualizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.save_button = None
        self.nom_input = None
        self.button = None
        self.stimConfigValue = None
        self.label = None
        self.title = 'Interface Stim'
        self.DataToPlot = {
            'LHip': {},
            'LKnee': {},
            'LAnkle': {},
            'StanceDuration_L': {},
            'PropulsionDuration_L': {},
            'Force_1': {}
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
        form_layout = QHBoxLayout()
        self.amplitude_input = QLineEdit(self)
        self.frequence_input = QLineEdit(self)
        self.duree_input = QLineEdit(self)
        self.largeur_input = QLineEdit(self)
        self.name_input = QLineEdit(self)
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["SINGLE", "DOUBLET", "TRIPLET"])

        # Checkbox pour activer/désactiver le stimulateur
        self.stimulator_checkbox = QCheckBox('Activer stimulateur', self)
        self.stimulator_checkbox.stateChanged.connect(self.on_stimulator_checkbox_toggled)

        # Ajouter les widgets au layout
        form_layout.addWidget(self.stimulator_checkbox)
        form_layout.addWidget(QLabel('Name:'))
        form_layout.addWidget(self.name_input)
        form_layout.addWidget(QLabel('Amplitude:'))
        form_layout.addWidget(self.amplitude_input)
        form_layout.addWidget(QLabel('Fréquence:'))
        form_layout.addWidget(self.frequence_input)
        form_layout.addWidget(QLabel('Durée:'))
        form_layout.addWidget(self.duree_input)
        form_layout.addWidget(QLabel('Largeur:'))
        form_layout.addWidget(self.largeur_input)
        form_layout.addWidget(QLabel('Mode:'))
        form_layout.addWidget(self.mode_combo)

        # Ajouter un bouton pour actualiser la stimulation
        self.ActuStim_button = QPushButton('Actualiser stim', self)
        self.ActuStim_button.clicked.connect(self.stim_actu_clicked)
        form_layout.addWidget(self.ActuStim_button)

        # GroupBox pour les paramètres
        groupbox_params = QGroupBox("Paramètres de Stimulation")
        groupbox_params.setLayout(form_layout)
        groupbox_params.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        groupbox_params.setMaximumHeight(80)
        layout.addWidget(groupbox_params)

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

    def on_stimulator_checkbox_toggled(self, state):
        """Appelée lorsque la checkbox est cochée/décochée."""
        if state == Qt.Checked:
            print("Le stimulateur est activé.")
            self.activer_stimulateur()
        else:
            print("Le stimulateur est désactivé.")
            self.desactiver_stimulateur()

    def activer_stimulateur(self):
        try:
            self.channel_1 = Channel(
                "Single", no_channel=1, amplitude=15, pulse_width=250, frequency=25, name="Gastro",
                device_type=Device.Rehastimp24
            )
            self.stimulator = St(port="COM3")
            self.stimulator.init_stimulation(list_channels=[self.channel_1])
            print("Fonction d'activation du stimulateur appelée.")
        except Exception as e:
            print(f"Erreur lors de l'activation du stimulateur : {e}")
        self.stimConfigValue = 0

    def desactiver_stimulateur(self):
        if hasattr(self, 'stimulator') and self.stimulator:
            self.stimulator.end_stimulation()
            self.stimulator.close_port()
            print("Fonction de désactivation du stimulateur appelée.")

    def stim_actu_clicked(self):
        try:
            # Récupérer les valeurs depuis l'interface utilisateur
            amplitude = int(self.amplitude_input.text())
            frequence = int(self.frequence_input.text())
            duree = int(self.duree_input.text())
            largeur = int(self.largeur_input.text())
            mode_text = self.mode_combo.currentText()

            # Définir le mode de stimulation
            mode = Modes.SINGLE if mode_text == "SINGLE" else Modes.DOUBLET if mode_text == "DOUBLET" else Modes.TRIPLET

            # Mettre à jour les paramètres du canal
            self.channel_1.set_amplitude(amplitude)
            self.channel_1.set_pulse_width(largeur)
            self.channel_1.set_frequency(frequence)
            self.channel_1.set_mode(mode)

            # Mettre à jour la stimulation
            self.stimulator.update_stimulation(upd_list_channels=[self.channel_1], stimulation_duration=duree)
            print(
                f"Stimulation mise à jour : Amplitude={amplitude}, Fréquence={frequence}, "
                f"Durée={duree}, Largeur={largeur}, Mode={mode_text}")
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
                    interpolated_vector = self.interpolate_vector(value[1,:])
                    self.DataToPlot[key][self.stimConfigValue].append(interpolated_vector*180/3.14)
                else:
                    self.DataToPlot[key][self.stimConfigValue].append(value)
            elif isinstance(value, (int, float)):  # Handle scalar numbers separately
                self.DataToPlot[key][self.stimConfigValue].append(value)

        self.update_graphs()

    def interpolate_vector(self, vector):
        # Vérification que la taille du vecteur est correcte avant d'interpoler
        if len(vector) == 0:
            logging.error("Le vecteur d'entrée est vide pour l'interpolation.")
            return np.zeros(100)  # Retourner un vecteur nul si le vecteur est vide

        x = np.linspace(0, 1, len(vector))
        x_new = np.linspace(0, 1, 100)

        try:
            interpolated_vector = np.interp(x_new, x, vector)
            return interpolated_vector
        except Exception as e:
            logging.error(f"Erreur lors de l'interpolation du vecteur : {e}")
            self.show_error_message(f"Erreur d'interpolation : {e}")
            return np.zeros(100)  # Retourner un vecteur nul en cas d'erreur

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
