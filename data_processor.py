import biorbd
import numpy as np
import concurrent.futures
import os
import csv
import logging

# Configure le logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class DataProcessor:
    def __init__(self, visualization_widget):
        self.is_in_cycle = False
        self.visualization_widget = visualization_widget
        self.model = biorbd.Model()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.cycle_num = 0

    def start_new_cycle(self, cycledata):
        logging.info("Début d'un nouveau cycle détecté.")
        try:
            futures = [
                self.executor.submit(self.calculate_kinematic_dynamic, cycledata['Force'], cycledata['Angle']),
                self.executor.submit(self.calculate_gait_parameters, cycledata['Force'], cycledata['Markers'])
            ]

            results = [future.result() for future in futures]
            kinematic_dynamic_result = results[0]
            gait_parameters = results[1]
            cycledata['gait_parameter'] = gait_parameters
            self.cycle_num = self.cycle_num+1

            self.executor.submit(self.visualization_widget.update_data_and_graphs, cycledata)
            self.executor.submit(self.save_cycle_data, cycledata)

        except Exception as e:
            logging.error(f"Erreur lors du traitement du nouveau cycle : {e}")

    @staticmethod
    def calculate_kinematic_dynamic(forcedata, angle):
        logging.info("Calcul de la dynamique inverse...")
        # Ajoutez ici la logique de calcul

    @staticmethod
    def calculate_gait_parameters(forcedata, mksdata):
        logging.info("Calcul des paramètres de marche...")
        """Calcul des paramètres de marche."""
        fs_pf = 2000  # TODO pas en brut
        fz_pf1 = forcedata['Force_1'][2, :]
        fz_pf2 = forcedata['Force_2'][2, :]
        fx_pf1 = forcedata['Force_1'][0, :]
        fx_pf2 = forcedata['Force_2'][0, :]
        mks1 = mksdata['LCAL']
        mks2 = mksdata['RCAL']
        RapportFs=len(mks1[0])/len(fx_pf2)
        rheel_strikes = np.where((fz_pf2[1:] > 30) & (fz_pf2[:-1] <= 30))[0][0] + 1
        ltoe_off = np.where(fz_pf1 < 20)[0][0]

        gait_param = {
            'StanceDuration_L': 100 * (ltoe_off / fs_pf),
            'StanceDuration_R': 100 * (rheel_strikes / fs_pf),
            'Cycleduration': len(fz_pf2) / fs_pf,
            'StepWidth': np.abs(np.mean(mks1[1, :]) - np.mean(mks2[1, :])),
            'StepLength_L': mks1[0, -1] - mks2[0, -1],
            'StepLength_R': mks2[0, int(rheel_strikes * RapportFs)] - mks1[0, int(rheel_strikes * RapportFs )],
            'PropulsionDuration_L': len(np.where(fx_pf1 < -6)[0]) / fs_pf,
            'PropulsionDuration_R': len(np.where(fx_pf2 < -6)[0]) / fs_pf,
            'Cadence': 2 * (60 / (len(fz_pf2) / fs_pf)),
        }
        return gait_param

    def extract_headers_and_values(self, data, parent_key='', sep='_'):
        headers = []
        values = []

        # Si le data est un dictionnaire, on parcourt chaque clé
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f'{parent_key}{sep}{key}' if parent_key else key
                sub_headers, sub_values = self.extract_headers_and_values(value, full_key, sep=sep)
                headers.extend(sub_headers)
                values.extend(sub_values)
        # Si le data est un tableau numpy, on l'aplatit et on génère des en-têtes pour chaque élément
        elif isinstance(data, np.ndarray):
            for i in range(data.size):
                headers.append(f'{parent_key}{sep}{i + 1}')
            values.extend(data.flatten().tolist())
        # Si le data est une valeur simple (str, int, float), on l'ajoute directement
        else:
            headers.append(parent_key)
            values.append(data)

        return headers, values

    def save_cycle_data(self, cycledata):
        logging.info("Saving cycle data into separate CSV files, with rows per frame...")
        try:
            save_dir = "saved_cycles"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Include StimulationConfig in cycledata
            cycledata['StimulationConfig'] = self.get_stimulation_config()

            # Iterate through each primary key in cycledata
            for key, data in cycledata.items():
                if isinstance(data, dict) and any(isinstance(v, (np.ndarray, list, float, int)) for v in data.values()):
                    # Flatten the data structure for saving
                    headers, flattened_data = self.flatten_single_entry_with_frames(data, key)
                else:
                    # Handle the single-entry data (e.g., 'StimulationConfig')
                    headers, flattened_data = self.flatten_single_entry_without_frames(data, key)

                # Construct the output file name
                output_file = os.path.join(save_dir, f"{self.visualization_widget.nom_input.text()}_{key}.csv")

                # Check if the file exists to avoid writing headers multiple times
                file_exists = os.path.isfile(output_file)

                # Save the data to a CSV file
                with open(output_file, mode='a', newline='') as file:
                    writer = csv.writer(file)

                    # Write headers only if the file is newly created or empty
                    if not file_exists or self.cycle_num == 0:
                        writer.writerow(headers)

                    # Write flattened data rows
                    writer.writerows(flattened_data)

                logging.info(f"Data for key '{key}' saved to {output_file}.")

        except Exception as e:
            logging.error(f"Error while saving data: {e}")

    def flatten_single_entry_with_frames(self, data, key):
        flattened_data = []
        headers = ['Cycle', 'Frame']  # Initialize headers with 'Cycle' and 'Frame'

        # Check if 'data' is a dictionary
        if not isinstance(data, dict):
            raise ValueError(f"Expected data for key '{key}' to be a dictionary, got {type(data)}")

        # Determine the number of frames from data structure
        num_frames = None
        example_key = None
        for subkey, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 3:
                num_frames = value.shape[1]
                example_key = subkey
                break
            elif isinstance(value, list):
                num_frames = len(value)
                example_key = subkey
                break
            elif isinstance(value, float) or isinstance(value, int):
                num_frames = 1
                break

        if num_frames is None:
            raise ValueError(f"Could not determine frame count for key '{key}'")

        # Extend headers based on the data type and content
        for subkey, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 3:
                headers.extend([f"{subkey}_Fx", f"{subkey}_Fy", f"{subkey}_Fz"])
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                headers.append(f"{subkey}")
            else:
                headers.append(f"{subkey}")

        # Iterate over each frame to construct rows
        for frame_idx in range(num_frames):
            row = [self.cycle_num, frame_idx + 1]  # Add 'Cycle' and 'Frame' number

            # Populate the row based on data type
            for subkey, value in data.items():
                if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 3:
                    if frame_idx < value.shape[1]:
                        fx, fy, fz = value[:, frame_idx]
                        row.extend([fx, fy, fz])
                    else:
                        row.extend([None, None, None])
                elif isinstance(value, np.ndarray) or isinstance(value, list):
                    row.append(value[frame_idx] if frame_idx < len(value) else None)
                elif isinstance(value, (float, int)):
                    row.append(value)
                else:
                    row.append(None)  # Handle unexpected data types as None

            flattened_data.append(row)

        return headers, flattened_data

    # Handle single-entry data (e.g., 'StimulationConfig') which is not frame-based
    def flatten_single_entry_without_frames(self, data, key):
        flattened_data = []
        headers = ['Cycle']  # Initialize headers with 'Cycle'

        # Check if 'data' is a dictionary
        if not isinstance(data, dict):
            raise ValueError(f"Expected data for key '{key}' to be a dictionary, got {type(data)}")

        # Extend headers based on the keys in the dictionary
        headers.extend(data.keys())

        # Create a single row with 'Cycle' value and data values
        row = [self.cycle_num]
        for subkey in data.keys():
            row.append(data[subkey])

        flattened_data.append(row)

        return headers, flattened_data

    def get_stimulation_config(self):
        try:
            return {
                "Amplitude": self.visualization_widget.amplitude_input.text(),
                "Fréquence": self.visualization_widget.frequence_input.text(),
                "Durée": self.visualization_widget.duree_input.text(),
                "Largeur": self.visualization_widget.largeur_input.text(),
                "Mode": self.visualization_widget.mode_combo.currentText(),
            }
        except Exception as e:
            logging.error(f"Erreur lors de la récupération de la configuration de stimulation : {e}")
            return {}
