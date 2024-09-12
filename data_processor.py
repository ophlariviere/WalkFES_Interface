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
        GaitParameter=self.calculate_gait_parameters(cycledata['Force'], cycledata['Markers'])
        cycledata['GaitParameter'] = GaitParameter
        print("Mise à jour des graphiques...")
        # Utilisation des données traitées pour mettre à jour le widget
        self.visualization_widget.update_data_and_graphs(cycledata)
        self.save_cycle_data(cycledata)
        self.reset_cycle_data()

    def calculate_kinematic_dynamic(self,forcedata,angle):
        self.model=biorbd.Model('C:\\Users\\felie\\PycharmProjects\\test\\examples\\LAO.bioMod')

        """Calcule la cinématique et la dynamique inverse pour le cycle actuel."""
        print("Calcul de la cinématique inverse (IK) et dynamique inverse (ID)...")
        # Ajoutez ici la logique réelle de calcul si nécessaire.

    def calculate_gait_parameters(self,forcedata,mksdata):
        Fs_PF = 2000
        Fs_Mks = 200
        Fz_PF1 = forcedata[1]['Force'][2, :]  # Fz pour PF1 (pied gauche)
        Fz_PF2 = forcedata[2]['Force'][2, :]  # Fz pour PF2 (pied droit)
        Fx_PF1 = forcedata[1]['Force'][0, :]  # Fy pour PF1 (pied gauche)
        Fx_PF2 = forcedata[2]['Force'][0, :]  # Fy pour PF2 (pied droit)
        Mks1 = mksdata['LCAL']
        Mks2 = mksdata['RCAL']
        bool_array = Fz_PF2 > 30
        Rheel_strikes = np.where((bool_array[1:] == True) & (bool_array[:-1] == False))[0][0]+1
        Ltoe_off = np.where(Fz_PF1 < 30)[0][0]
        Rtoe_off = np.where(Fz_PF2 < 30)[0][0]
        GaitParam = {
            'StanceDuration': {'L': 0, 'R': 0},
            'Cycleduration': 0,
            'StepWidth': 0,
            'StepLength': {'L': 0, 'R': 0},
            'PropulsionDuration': {'L': 0, 'R': 0},
            'Cadence': 0
        }

        GaitParam['Cycleduration'] = (len(Fz_PF2))/Fs_PF
        GaitParam['StanceDuration']['L'] = 100*(Ltoe_off / Fs_PF) / GaitParam['Cycleduration']
        GaitParam['StanceDuration']['R'] = 100*((len(Fz_PF2) - (Rheel_strikes - Rtoe_off)) / Fs_PF) / GaitParam['Cycleduration']
        GaitParam['StepWidth'] = np.abs(np.mean(Mks1[1, :]) - np.mean(Mks2[1, :]))
        GaitParam['StepLength']['L'] = Mks1[0, 0] - Mks2[0, 0]
        GaitParam['StepLength']['R'] = Mks2[0, int(Fs_Mks*Rheel_strikes/Fs_PF)] - Mks1[0, int(Fs_Mks*Rheel_strikes/Fs_PF)]
        GaitParam['PropulsionDuration']['L'] = len(np.where(Fx_PF2 < -6)[0])/Fs_PF
        GaitParam['PropulsionDuration']['R'] = len(np.where(Fx_PF1 < -6)[0])/Fs_PF
        GaitParam['Cadence'] = 2*(60 / GaitParam['Cycleduration'])
        return GaitParam


    def save_cycle_data(self):
        """Sauvegarde les données du cycle actuel."""
        print("Sauvegarde des données du cycle...")
        # Ajoutez ici la logique réelle de sauvegarde si nécessaire.

    def reset_cycle_data(self):
        """Réinitialise les données pour un nouveau cycle."""
        self.current_cycle_data = []


# Exemple d'utilisation
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizationWidget()
    window.show()

    # Passer le widget de visualisation au DataProcessor
    processor = DataProcessor(visualization_widget=window)
    processor.start_new_cycle()

    sys.exit(app.exec_())
