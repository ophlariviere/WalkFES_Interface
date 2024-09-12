import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QCheckBox, QFormLayout, QGroupBox, QPushButton, QComboBox, QSizePolicy,QFileDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import random
from pyScienceMode import Channel, Point, Device, Modes
from pyScienceMode import RehastimP24 as St
import numpy as np


class VisualizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Interface Stim'
        self.init_ui()
        self.dataAll = {}


    def init_ui(self):
        self.stimConfigValue = 0
        self.setWindowTitle(self.title)
        #self.setGeometry(self.left, self.top, self.width, self.height)

        # Layout principal
        layout = QVBoxLayout()

        # Selection du model
        model_layout = QHBoxLayout()
        self.button = QPushButton('Choisir un fichier', self)
        self.button.clicked.connect(self.openFileNameDialog)
        self.label = QLabel('Aucun fichier sélectionné', self)
        model_layout.addWidget(self.button)
        model_layout.addWidget(self.label)
        # GroupBox pour l'enregistrement des données
        groupBox_model = QGroupBox("Téléchargement du modele:")
        groupBox_model.setLayout(model_layout)
        # Ajouter le GroupBox au layout principal
        groupBox_model.setMaximumHeight(80)
        groupBox_model.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)  # minimiser la hauteur
        layout.addWidget(groupBox_model)

        # Section d'enregistrement des données
        save_layout = QHBoxLayout()
        self.nom_input = QLineEdit(self)
        save_layout.addWidget(QLabel('Nom pour enregistrer les données:'))
        save_layout.addWidget(self.nom_input)
        self.save_button = QPushButton('Validate', self)
        self.save_button.clicked.connect(self.save_path)
        save_layout.addWidget(self.save_button)
        # GroupBox pour l'enregistrement des données
        groupBox_save = QGroupBox("Gestion enregistrement données:")
        groupBox_save.setLayout(save_layout)
        # Ajouter le GroupBox au layout principal
        groupBox_save.setMaximumHeight(80)
        groupBox_save.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum) #minimiser la hauteur
        layout.addWidget(groupBox_save)

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
        self.ActuStim_button.clicked.connect(self.Stim_Actu_clicked)
        form_layout.addWidget(self.ActuStim_button)

        # GroupBox pour les paramètres
        groupBox_params = QGroupBox("Paramètres de Stimulation")
        groupBox_params.setLayout(form_layout)
        groupBox_params.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        groupBox_params.setMaximumHeight(80)
        layout.addWidget(groupBox_params)

        # Sélections d'analyse
        self.StanceDuration_checkbox = QCheckBox('StanceDuration', self)
        self.cinetique_checkbox = QCheckBox('Moment cinétique', self)
        self.articulaire_checkbox = QCheckBox('Moment articulaire', self)
        self.travail_checkbox = QCheckBox('Travail mécanique', self)

        analysis_layout = QHBoxLayout()
        analysis_layout.addWidget(self.StanceDuration_checkbox)
        analysis_layout.addWidget(self.cinetique_checkbox)
        analysis_layout.addWidget(self.articulaire_checkbox)
        analysis_layout.addWidget(self.travail_checkbox)

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
        self.stimConfigValue=0

    def desactiver_stimulateur(self):
        if self.stimulator:
            self.stimulator.end_stimulation()
            self.stimulator.close_port()
            print("Fonction de désactivation du stimulateur appelée.")

    def Stim_Actu_clicked(self):
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
            print(f"Stimulation mise à jour : Amplitude={amplitude}, Fréquence={frequence}, Durée={duree}, Largeur={largeur}, Mode={mode_text}")
        except ValueError:
            print("Erreur : Veuillez entrer des valeurs numériques valides.")
        except Exception as e:
            print(f"Erreur lors de l'actualisation de la stimulation : {e}")
        self.stimConfigValue = self.stimConfigValue + 1

    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data_and_graphs)
        self.timer.start(1000)  # Mise à jour toutes les 1000 ms (1 seconde)

    def update_data_and_graphs(self, new_data):
        """
        Met à jour les données et actualise les graphiques.
        - new_data: Nouvelles données du cycle actuel (peut être une valeur unitaire ou un vecteur)
        """
        # Mise à jour des données vectorielles pour self.stimConfigValue
        if self.stimConfigValue not in self.dataAll:
            self.dataAll[self.stimConfigValue] = []
        self.dataAll[self.stimConfigValue].append(new_data)

        # Après mise à jour, on redessine les graphes
        self.update_graphs()

    def update_graphs(self):
        self.figure.clear()

        # Vérification des checkboxes pour déterminer les graphiques à afficher
        graphs_to_display = [
            self.StanceDuration_checkbox.isChecked(),
            self.cinetique_checkbox.isChecked(),
            self.articulaire_checkbox.isChecked(),
            self.travail_checkbox.isChecked()
        ]
        count = sum(graphs_to_display)  # Compter combien de graphes afficher

        if count == 0:
            return  # Aucun graphique à afficher, on ne fait rien

        # Calcul du nombre de lignes et de colonnes pour les subplots
        rows = (count + 1) // 2
        cols = 2 if count > 1 else 1
        subplot_index = 1

        # Affichage des valeurs unitaires : StanceDuration
        if self.StanceDuration_checkbox.isChecked():
            ax = self.figure.add_subplot(rows, cols, subplot_index)
            configstim_values = []
            cadence_values = []

            # Parcourir les configurations et cycles pour extraire les valeurs de cadence
            for numeroconfigstim in range(len(self.dataAll)):  # Itérer sur les configurations
                for numerocycle in range(len(self.dataAll[numeroconfigstim])):  # Itérer sur les cycles
                    # Extraire la valeur de cadence pour chaque cycle et configuration
                    cadence = self.dataAll[numeroconfigstim][numerocycle]['GaitParameter']['Cadence']

                    # Ajouter les données aux listes
                    configstim_values.append(numeroconfigstim)
                    cadence_values.append(cadence)

            ax.plot(configstim_values, cadence_values, 'o')
            ax.set_title("StanceDuration par Cycle")
            ax.legend()
            subplot_index += 1

        # Affichage des valeurs unitaires : Moment cinétique
        if self.cinetique_checkbox.isChecked():
            ax = self.figure.add_subplot(rows, cols, subplot_index)
            for stim_value, moment_data in self.dataAll.get('Moment cinétique', {}).items():
                ax.plot([stim_value] * len(moment_data), moment_data, 'o', label=f"Moment cinétique {stim_value}")
            ax.set_title("Moment Cinétique par Cycle")
            ax.legend()
            subplot_index += 1

        # Affichage des vecteurs : Tau
        if self.articulaire_checkbox.isChecked():
            ax = self.figure.add_subplot(rows, cols, subplot_index)
            for stim_value, tau_data in self.dataAll.get('Tau', {}).items():
                stacked_tau = np.vstack(tau_data)  # Empiler les vecteurs pour calculer
                mean_tau = np.mean(stacked_tau, axis=0)
                std_tau = np.std(stacked_tau, axis=0)
                x_values = np.arange(len(mean_tau))
                ax.plot(x_values, mean_tau, label=f"Moyenne Tau {stim_value}")
                ax.fill_between(x_values, mean_tau - std_tau, mean_tau + std_tau, alpha=0.3,
                                label=f"Écart-Type Tau {stim_value}")
            ax.set_title("Moyenne et Écart-Type Tau")
            ax.legend()
            subplot_index += 1

        # Affichage des vecteurs : Work
        if self.travail_checkbox.isChecked():
            ax = self.figure.add_subplot(rows, cols, subplot_index)
            for stim_value, work_data in self.dataAll.get('Work', {}).items():
                stacked_work = np.vstack(work_data)  # Empiler les vecteurs pour calculer
                mean_work = np.mean(stacked_work, axis=0)
                std_work = np.std(stacked_work, axis=0)
                x_values = np.arange(len(mean_work))
                ax.plot(x_values, mean_work, label=f"Moyenne Work {stim_value}")
                ax.fill_between(x_values, mean_work - std_work, mean_work + std_work, alpha=0.3,
                                label=f"Écart-Type Work {stim_value}")
            ax.set_title("Moyenne et Écart-Type Work")
            ax.legend()
            subplot_index += 1

        # Redessiner le canevas pour afficher les nouvelles données
        self.canvas.draw()

    def openFileNameDialog(self):
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VisualizationWidget()
    ex.show()
    sys.exit(app.exec_())
