import biorbd
import numpy as np
import concurrent.futures
import os
import csv


class DataProcessor:
    def __init__(self, visualization_widget):
        self.is_in_cycle = False
        self.visualization_widget = visualization_widget
        self.model = biorbd.Model(
            'C:\\Users\\felie\\PycharmProjects\\test\\examples\\LAO.bioMod')  # Précharger le modèle
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)  # Gestion des threads

    def start_new_cycle(self, cycledata):
        print("Début d'un nouveau cycle détecté.")
        # Lancer les calculs cinématiques/dynamiques et des paramètres de marche en parallèle
        futures = [
            self.executor.submit(self.calculate_kinematic_dynamic, cycledata['Force'], cycledata['Angle']),
            self.executor.submit(self.calculate_gait_parameters, cycledata['Force'], cycledata['Markers'])
        ]

        # Attendre la fin des calculs (récupérer les deux résultats si nécessaire)
        results = [future.result() for future in futures]

        # Stocker les résultats dans cycledata (si on veut aussi stocker le résultat de calculate_kinematic_dynamic)
        kinematic_dynamic_result = results[0]  # Résultat de calculate_kinematic_dynamic (si nécessaire)
        gait_parameters = results[1]  # Résultat de calculate_gait_parameters

        # Mise à jour des paramètres de marche dans cycledata
        cycledata['gait_parameter'] = gait_parameters

        # Lancer la mise à jour des graphiques de manière asynchrone
        self.executor.submit(self.visualization_widget.update_data_and_graphs, cycledata)

        # Lancer la sauvegarde des données de manière asynchrone
        self.executor.submit(self.save_cycle_data, cycledata)

        # L'interface utilisateur reste réactive pendant que les opérations ci-dessus sont effectuées en parallèle

    @staticmethod
    def calculate_kinematic_dynamic(forcedata, angle):
        """Calcule la cinématique inverse (IK) et dynamique inverse (ID)."""
        print("Calcul de la dynamique inverse...")
        # Ajoutez ici la logique de calcul

    @staticmethod
    def calculate_gait_parameters(forcedata, mksdata):
        """Calcul des paramètres de marche."""
        fs_pf = 2000  # TODO pas en brut
        fz_pf1 = forcedata[1]['Force'][2, :]
        fz_pf2 = forcedata[2]['Force'][2, :]
        fx_pf1 = forcedata[1]['Force'][0, :]
        fx_pf2 = forcedata[2]['Force'][0, :]
        mks1 = mksdata['LCAL']
        mks2 = mksdata['RCAL']

        rheel_strikes = np.where((fz_pf2[1:] > 30) & (fz_pf2[:-1] <= 30))[0][0] + 1
        ltoe_off = np.where(fz_pf1 < 20)[0][0]

        gait_param = {
            'StanceDuration': {'L': 100 * (ltoe_off / fs_pf), 'R': 100 * (rheel_strikes / fs_pf)},
            'Cycleduration': len(fz_pf2) / fs_pf,
            'StepWidth': np.abs(np.mean(mks1[1, :]) - np.mean(mks2[1, :])),
            'StepLength': {'L': mks1[0, -1] - mks2[0, -1], 'R': mks2[0, int(rheel_strikes/10)] - mks1[0, int(rheel_strikes/10)]},
            'PropulsionDuration': {'L': len(np.where(fx_pf1 < -6)[0]) / fs_pf,
                                   'R': len(np.where(fx_pf2 < -6)[0]) / fs_pf},
            'Cadence': 2 * (60 / (len(fz_pf2) / fs_pf)),
        }
        return gait_param

    def save_cycle_data(self, cycledata):
        """Sauvegarde des données du cycle."""
        print("Sauvegarde des données...")

        # Récupérer le nom du fichier depuis self.nom_input de VisualizationWidget
        save_dir = "saved_cycles"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Utiliser le nom défini par l'utilisateur dans self.nom_input
        filename = os.path.join(save_dir, f"{self.visualization_widget.nom_input.text()}.csv")

        # Ajouter les configurations de stimulation aux données du cycle
        cycledata['StimulationConfig'] = self.get_stimulation_config()

        # Convertir les données en un format sérialisable (e.g. convertir les ndarray en listes)
        for key, value in cycledata.items():
            if isinstance(value, np.ndarray):
                cycledata[key] = value.tolist()  # Convertir ndarray en liste

        # Écriture dans un fichier CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Écrire les clés comme en-têtes
            writer.writerow(cycledata.keys())

            # Transposer les valeurs pour qu'elles correspondent à leurs clés respectives
            rows = zip(*cycledata.values())

            # Écrire les lignes des données
            for row in rows:
                writer.writerow(row)
        print(f"Données du cycle et configuration de stimulation sauvegardées dans {filename}")

    def get_stimulation_config(self):
        """Récupère les configurations de stimulation depuis le widget."""
        return {
            "Amplitude": self.visualization_widget.amplitude_input.text(),
            "Fréquence": self.visualization_widget.frequence_input.text(),
            "Durée": self.visualization_widget.duree_input.text(),
            "Largeur": self.visualization_widget.largeur_input.text(),
            "Mode": self.visualization_widget.mode_combo.currentText(),
        }
