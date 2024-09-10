import biorbd
import numpy as np

from visualization import VisualizationWidget
from PyQt5.QtWidgets import QApplication
import sys

class DataProcessor:
    def __init__(self, visualization_widget):
        self.current_cycle_data = []
        self.is_in_cycle = False
        self.visualization_widget = visualization_widget  # Référence au widget de visualisation

    def start_new_cycle(self,cycledata):
        print("Début d'un nouveau cycle détecté.")
        self.calculate_kinematic_dynamic(cycledata['Force'], cycledata['Angle'])
        self.calculate_gait_parameters(cycledata['Force'], cycledata['Markers'])
        print("Mise à jour des graphiques...")
        # Utilisation des données traitées pour mettre à jour le widget
        self.visualization_widget.update_data_and_graphs(self.current_cycle_data)
        self.save_cycle_data()
        self.reset_cycle_data()

    def calculate_kinematic_dynamic(self,forcedata,angle):
        self.model=biorbd.Model('C:\\Users\\felie\\PycharmProjects\\test\\examples\\LAO.bioMod')
        self.q
        """Calcule la cinématique et la dynamique inverse pour le cycle actuel."""
        print("Calcul de la cinématique inverse (IK) et dynamique inverse (ID)...")
        # Ajoutez ici la logique réelle de calcul si nécessaire.

    def calculate_gait_parameters(self,forcedata,mksdata):
        Fs_PF=1000
        Fz_PF1 = forcedata[0][2, :]  # Fz pour PF1 (pied gauche)
        Fz_PF2 = forcedata[1][2, :]  # Fz pour PF2 (pied droit)
        Fy_PF1 = forcedata[0][1, :]  # Fy pour PF1 (pied gauche)
        Fy_PF2 = forcedata[1][1, :]  # Fy pour PF2 (pied droit)
        Mks1 =maksdata['LCAL']
        Mks2 = maksdata['RCAL']
        Rheel_strikes = np.where(Fz_PF2 > self.threshold)[0]
        Ltoe_off = np.where(Fz_PF1 < self.threshold)[0]
        Rtoe_off = np.where(Fz_PF2 < self.threshold)[0]
        GaitParam['StanceDuration']['L'] = Ltoe_off/Fs_PF
        GaitParam['StanceDuration']['R'] = (len(Fz_PF2)-(Rheel_strikes-Rtoe_off))/Fs_PF
        GaitParam['Cycleduration'] = (len(Fz_PF2))/Fs_PF
        GaitParam['StepWidth'] = np.abs(np.mean(Mks1[1, :]) - np.mean(Mks2[1, :]))
        GaitParam['StepLength']['L'] = Mks1[0, :-1] - Mks1[0, 0]
        GaitParam['StepLength']['R'] = Mks2[0, :-1] - Mks2[0, 0]
        GaitParam['PropulsionDuration']['L'] = len(np.where(Fy_PF2 < 4))/Fs_PF
        GaitParam['PropulsionDuration']['R'] = len(np.where(Fy_PF1 < 4))/Fs_PF
        GaitParam['Cadence'] = (60 / GaitParam['Cycleduration']) * 2



    def save_cycle_data(self):
        """Sauvegarde les données du cycle actuel."""
        print("Sauvegarde des données du cycle...")
        # Ajoutez ici la logique réelle de sauvegarde si nécessaire.

    def reset_cycle_data(self):
        """Réinitialise les données pour un nouveau cycle."""
        self.current_cycle_data = []
        self.is_in_cycle = False


# Exemple d'utilisation
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizationWidget()
    window.show()

    # Passer le widget de visualisation au DataProcessor
    processor = DataProcessor(visualization_widget=window)
    processor.start_new_cycle()

    sys.exit(app.exec_())
