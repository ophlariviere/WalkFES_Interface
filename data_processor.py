import biorbd
import numpy as np
import concurrent.futures
import os
import csv
import logging
from scipy.signal import filtfilt, butter, savgol_filter


class DataProcessor:
    def __init__(self, visualization_widget, buffer):
        self.visualization_widget = visualization_widget
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.cycle_num = 0
        self.dofcorr = {"LHip": (36, 37, 38), "LKnee": (39, 40, 41), "LAnkle": (42, 43, 44),
                        "RHip": (27, 28, 29), "RKnee": (30, 31, 32), "RAnkle": (33, 34, 35),
                        "LShoulder": (18, 19, 20), "LElbow": (21, 22, 23), "LWrist": (24, 25, 26),
                        "RShoulder": (9, 10, 11), "RElbow": (12, 13, 14), "RWrist": (15, 16, 17),
                        "Thorax": (6, 7, 8), "Pelvis": (3, 4, 5)}
        self.buffer = buffer

    def start_new_cycle(self, cycledata):
        logging.info("Début d'un nouveau cycle détecté.")
        try:
            futures = []
            # Check if self.model exists before adding the kinematic dynamic calculation
            if self.visualization_widget.model is not None:
                futures.append(
                    self.executor.submit(self.calculate_kinematic_dynamic, cycledata['Force'], cycledata['Markers']))

            # Always add the gait parameters calculation

            futures.append(
                self.executor.submit(self.calculate_gait_parameters, cycledata['Force'], cycledata['Markers']))

            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logging.error(f"Erreur lors de l'exécution du thread : {e}")

            # Extract results
            if self.visualization_widget.model is not None:
                kinematic_dynamic_result = results[0][0]
                q = results[0][1]
                qdot = results[0][2]
                qddot = results[0][3]

                gait_parameters = results[1]
                cycledata['gait_parameter'] = gait_parameters

                moment = {}
                Ang = {}
                VitAng = {}
                for key, indices in self.dofcorr.items():
                    keyt = f"Tau_{key}"
                    moment[keyt] = np.array([kinematic_dynamic_result[indices[0]],
                                             kinematic_dynamic_result[indices[1]],
                                             kinematic_dynamic_result[indices[2]]])

                    keyq = f"q_{key}"
                    Ang[keyq] = np.array([q[indices[0]], q[indices[1]], q[indices[2]]])

                    keyqdot = f"qdot_{key}"
                    VitAng[keyqdot] = np.array([qdot[indices[0]], qdot[indices[1]], qdot[indices[2]]])

                cycledata['Tau'] = moment      # Tau
                cycledata['VitAng'] = VitAng   # Qdot
                cycledata['Angle'] = Ang       # Q
                cycledata['AccAng'] = qddot    # Qddot

            else:
                # If self.model does not exist, handle only the gait parameters
                gait_parameters = results[0]
                cycledata['gait_parameter'] = gait_parameters
                logging.info("Kinematic dynamic calculation skipped as self.model is not defined.")

            # TODO: Charbie -> check if the gait parameters are stable (if yes update params, else keep stimulating)
            self.executor.submit(self.visualization_widget.update_data_and_graphs, cycledata)
            self.executor.submit(self.save_cycle_data, cycledata)
            self.cycle_num += 1
            self.buffer.put(cycledata)

        except Exception as e:
            logging.error(f"Erreur lors du traitement du nouveau cycle : {e}")

        return cycledata

    @staticmethod
    def forcedatafilter(data, order, sampling_rate, cutoff_freq):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = np.empty([len(data[:, 0]), len(data[0, :])])
        for ii in range(3):
            # filtered_data[ii, :] = medfilt(data[ii, :], kernel_size=5)
            filtered_data[ii, :] = filtfilt(b, a, data[ii, :], axis=0)
        return filtered_data

    def calculate_kinematic_dynamic(self, forcedata, mks):
        if self.visualization_widget.model:
            logging.info("Calcul de la dynamique inverse...")
            # Récupération des informations de base

            n_frames = next(iter(mks.values())).shape[1]
            marker_names = tuple(n.to_string() for n in self.visualization_widget.model.technicalMarkerNames())

            # Créer un array de NaN pour stocker les données des marqueurs
            markers_in_c3d = np.full((3, len(marker_names), n_frames), np.nan)

            # Remplir le tableau avec les données des marqueurs à partir du dictionnaire 'mks'
            for i, name in enumerate(marker_names):
                if name in mks:
                    filtmks = self.forcedatafilter(mks[name], 4, 100, 10)
                    markers_in_c3d[:, i, :] = filtmks  # Conversion en mètres si les données sont en mm

            # Créer une structure de filtre de Kalman
            freq = 100
            params = biorbd.KalmanParam(freq)
            kalman = biorbd.KalmanReconsMarkers(self.visualization_widget.model, params)

            # Configuration de la boucle d'exécution
            q = biorbd.GeneralizedCoordinates(self.visualization_widget.model)
            qdot = biorbd.GeneralizedVelocity(self.visualization_widget.model)
            qddot = biorbd.GeneralizedAcceleration(self.visualization_widget.model)
            frame_rate = 100
            first_frame = 0  # Commence à la première frame
            last_frame = n_frames - 1  # Termine à la dernière frame
            # t = np.linspace(first_frame / frame_rate, last_frame / frame_rate, n_frames)

            # Initialiser les tableaux de sortie pour toutes les frames
            q_out = np.ndarray((self.visualization_widget.model.nbQ(), n_frames))
            qdot_out = np.ndarray((self.visualization_widget.model.nbQdot(), n_frames))
            qddot_out = np.ndarray((self.visualization_widget.model.nbQddot(), n_frames))

            # Application du filtre de Kalman pour chaque frame
            for i in range(n_frames):
                kalman.reconstructFrame(self.visualization_widget.model,
                                        np.reshape(markers_in_c3d[:, :, i].T, -1), q, qdot, qddot)
                q_out[:, i] = q.to_array()
                qdot_out[:, i] = qdot.to_array()
                qddot_out[:, i] = qddot.to_array()

            angle = q_out

            contact_names = ["LFoot", "RFoot"]
            num_contacts = len(contact_names)
            num_frames = len(angle[0, :])
            sampling_factor = 20

            # Initialize arrays for storing external forces and moments
            force_filtered = np.zeros((num_contacts, 3, len(forcedata['Force_1'][0])))
            moment_filtered = np.zeros((num_contacts, 3, len(forcedata['Force_1'][0])))
            cop_filtered = np.zeros((num_contacts, 3, len(forcedata['Force_1'][0])))
            platform_origin = np.array([[[0.79165588], [0.77004227], [0.00782072]],
                                        [[0.7856461], [0.2547548], [0.00760771]]])

            # Process force platform data
            for contact_idx, contact_name in enumerate(contact_names):
                force_filtered[contact_idx] = self.forcedatafilter(forcedata[f"Force_{contact_idx + 1}"], 4,
                                                                   2000, 10)
                moment_filtered[contact_idx] = self.forcedatafilter(forcedata[f"Moment_{contact_idx + 1}"]/1000, 4,
                                                                    2000, 10)
                # cop_filtered[contact_idx] = self.forcedatafilter(forcedata[f"CoP_{contact_idx + 1}"] / 1000,
                # 4, 2000, 10)

            # Initialize arrays for storing forces and points of application
            self.force = np.empty((num_contacts, 9, num_frames))
            point_application = np.zeros((num_contacts, 3, num_frames))

            # Appliquer le filtre Savitzky-Golay à chaque signal
            q = savgol_filter(angle, 31, 3, axis=1)

            # Initialize arrays for angular velocity and acceleration
            qdot = np.gradient(q, axis=1, edge_order=2) / (1 / 100)
            qddot = np.gradient(qdot, axis=1, edge_order=2) / (1 / 100)

            tau_data = []

            # Perform inverse dynamics frame-by-frame
            for i in range(num_frames):
                self.ext_load = self.visualization_widget.model.externalForceSet()

                for contact_idx, contact_name in enumerate(contact_names):
                    name = biorbd.String(contact_name)

                    # Extract the spatial vector (moment and force) in the global frame
                    spatial_vector = np.concatenate((moment_filtered[contact_idx, :, sampling_factor * i],
                                                     force_filtered[contact_idx, :, sampling_factor * i]))

                    # Define the application point in the global frame
                    point_app_global = platform_origin[contact_idx, :, 0]
                    # cop_filtered[contact_idx, :, sampling_factor * i]#

                    # Check if force magnitude is significant
                    if np.linalg.norm(spatial_vector[3:6]) > 40:
                        self.ext_load.add(name, spatial_vector, point_app_global)

                # Calculate the inverse dynamics tau using the model's function
                tau = self.visualization_widget.model.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i], self.ext_load)
                tau_data.append(tau.to_array())
                self.ext_load = []  # Reset external load for the next iteration

            # Convert tau_data to numpy array and store the results
            tau_data = np.array(tau_data)
            tau = tau_data.T  # Transpose to match expected output format
            return tau, q, qdot, qddot
        else:
            print('ID was not performed please add a model')

            tau = []
            return tau

    @staticmethod
    def calculate_gait_parameters(forcedata, mksdata):
        logging.info("Calcul des paramètres de marche...")
        """Calcul des paramètres de marche."""
        fs_pf = 2000  # TODO pas en brut
        fz_pf1 = forcedata['Force_1'][2, :]
        fz_pf2 = forcedata['Force_2'][2, :]
        fx_pf1 = forcedata['Force_1'][0, :]
        fx_pf2 = forcedata['Force_2'][0, :]
        #fs_mks=len(mks1[0])
        fs_mks = 100
        RapportFs = fs_mks/fs_pf
        rheel_strikes = np.where((fz_pf2[1:] > 30) & (fz_pf2[:-1] <= 30))[0][0] + 1
        ltoe_off = np.where(fz_pf1[:-20] > 30)[0][-2]

        if ('LCAL' in mksdata) and ('RCAL' in mksdata):
            mks1 = mksdata['LCAL']
            mks2 = mksdata['RCAL']
            gait_param = {
                'StanceDuration_L': 100 * (ltoe_off / fs_pf),
                'StanceDuration_R': 100 * (rheel_strikes / fs_pf),
                'Cycleduration': len(fz_pf2) / fs_pf,
                'StepWidth': np.abs(np.mean(mks1[1, :]) - np.mean(mks2[1, :])),
                'StepLength_L': mks1[0, -1] - mks2[0, -1],
                'StepLength_R': mks2[0, int(rheel_strikes * RapportFs)] - mks1[0, int(rheel_strikes * RapportFs)],
                'PropulsionDuration_L': len(np.where(fx_pf1 > 6)[0]) / fs_pf,
                'PropulsionDuration_R': len(np.where(fx_pf2 > 6)[0]) / fs_pf,
                'Cadence': 2 * (60 / (len(fz_pf2) / fs_pf)),  # 60s/min * nb_step/s = nb_step/min ?
            }
        else:
            gait_param = {
                'StanceDuration_L': 100 * (ltoe_off / fs_pf),
                'StanceDuration_R': 100 * (rheel_strikes / fs_pf),
                'Cycleduration': len(fz_pf2) / fs_pf,
                'PropulsionDuration_L': len(np.where(fx_pf1 > 6)[0]) / fs_pf,
                'PropulsionDuration_R': len(np.where(fx_pf2 > 6)[0]) / fs_pf,
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
            # Create an empty dictionary to store all channel configurations
            stimulation_config = {}

            # Iterate over all channels stored in visualization_widget.channel_inputs
            for channel, config in self.visualization_widget.stimconfig.items():
                channel_number = channel  # Ou si vous préférez utiliser un autre identifiant pour les canaux, ajustez ici

                # Récupérer les valeurs pour chaque paramètre de ce canal
                name = config.get("name", "")
                amplitude = config.get("amplitude", "")
                frequence = config.get("frequency", "")
                duree = config.get("pulse_width", "")  # Pulse width semble être la durée
                largeur = config.get("pulse_width",
                                     "")  # Largeur de l'impulsion, en fonction de la terminologie utilisée
                mode = config.get("mode", "")

                # Stocker dans le dictionnaire de stimulation
                stimulation_config[f"channel{channel_number}_name"] = name
                stimulation_config[f"channel{channel_number}_Amplitude"] = amplitude
                stimulation_config[f"channel{channel_number}_Fréquence"] = frequence
                stimulation_config[f"channel{channel_number}_Durée"] = duree
                stimulation_config[f"channel{channel_number}_Largeur"] = largeur
                stimulation_config[f"channel{channel_number}_Mode"] = mode

            # Retourner la configuration complète
            return stimulation_config

        except Exception as e:
            logging.error(f"Erreur lors de la récupération de la configuration de stimulation : {e}")
            return {}
