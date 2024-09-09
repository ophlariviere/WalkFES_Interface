import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QCheckBox, QFormLayout, QGroupBox, QPushButton, QComboBox, QSizePolicy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import random
from pyScienceMode import Channel, Point, Device, Modes
from pyScienceMode import RehastimP24 as St


class VisualizationWidget(QWidget):
    """Widget PyQt5 pour afficher les graphiques et les données en temps réel."""
    def __init__(self):
        super().__init__()
        self.title = 'Interface de Stimulation et d\'Analyse'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600

        # Initialisation des données
        self.data = {
            'Lyapunov': [],
            'Moment cinétique': [],
            'Tau': [],
            'Work': []
        }

        self.stimulator = None  # Variable pour le stimulateur
        self.channel_1 = None  # Stockage du canal de stimulation
        self.init_ui()
        self.start_timer()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Layout principal
        layout = QVBoxLayout()

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
        self.lyapunov_checkbox = QCheckBox('Lyapunov', self)
        self.cinetique_checkbox = QCheckBox('Moment cinétique', self)
        self.articulaire_checkbox = QCheckBox('Moment articulaire', self)
        self.travail_checkbox = QCheckBox('Travail mécanique', self)

        analysis_layout = QHBoxLayout()
        analysis_layout.addWidget(self.lyapunov_checkbox)
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

    def update_data_and_graphs(self):
        data_keys = ['Lyapunov', 'Moment cinétique', 'Tau', 'Work']

        for key in data_keys:
            if len(self.data[key]) >= 100:
                self.data[key].pop(0)
            self.data[key].append(random.randint(0, 50))

        self.update_graphs()

    def update_graphs(self):
        self.figure.clear()

        graphs_to_display = [
            self.lyapunov_checkbox.isChecked(),
            self.cinetique_checkbox.isChecked(),
            self.articulaire_checkbox.isChecked(),
            self.travail_checkbox.isChecked()
        ]
        count = sum(graphs_to_display)

        if count == 0:
            return  # Aucun graphique à afficher

        rows = (count + 1) // 2
        cols = 2 if count > 1 else 1
        subplot_index = 1

        if self.lyapunov_checkbox.isChecked():
            ax = self.figure.add_subplot(rows, cols, subplot_index)
            ax.plot(self.data['Lyapunov'], label="Lyapunov")
            ax.legend()
            subplot_index += 1

        if self.cinetique_checkbox.isChecked():
            ax = self.figure.add_subplot(rows, cols, subplot_index)
            ax.plot(self.data['Moment cinétique'], label="Moment cinétique")
            ax.legend()
            subplot_index += 1

        if self.articulaire_checkbox.isChecked():
            ax = self.figure.add_subplot(rows, cols, subplot_index)
            ax.plot(self.data['Tau'], label="Tau [Nm/kg]")
            ax.legend()
            subplot_index += 1

        if self.travail_checkbox.isChecked():
            ax = self.figure.add_subplot(rows, cols, subplot_index)
            ax.plot(self.data['Work'], label="Work [W/kg]")
            ax.legend()
            subplot_index += 1

        self.canvas.draw()

    def save_path(self):
        self.path_to_saveData = self.nom_input.text()
        print(f"Chemin d'enregistrement défini sur : {self.path_to_saveData}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VisualizationWidget()
    ex.show()
    sys.exit(app.exec_())
