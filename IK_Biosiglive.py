from custom_interface import MyInterface
from biosiglive import load, RealTimeProcessingMethod, InterfaceType, DeviceType, Server, InverseKinematicsMethods
import numpy as np
import time


# Fonction pour détecter si Fz dépasse le seuil
def detect_start(previous_f_z, current_f_z, threshold=30):
    # Détection du passage de inférieur à supérieur au seuil
    return previous_f_z <= threshold < current_f_z


if __name__ == "__main__":
    server_ip = "127.0.0.1"
    port = 50000
    server = Server(server_ip, port)
    server.start()

    # Charger une seule fois les noms des marqueurs
    tmp = load("walkAll_LAO01_Cond10.bio")
    mks_name = tmp['markers_names'].data[0:49].tolist()

    # Interface setup
    interface = MyInterface(system_rate=100, data_path="walkAll_LAO01_Cond10.bio")
    nb_second = 1
    interface.add_marker_set(
        nb_markers=49,
        data_buffer_size=100 * nb_second,
        processing_window=100 * nb_second,
        marker_data_file_key="markers",
        name="markers",
        rate=100,
        kinematics_method=InverseKinematicsMethods.BiorbdKalman,
        model_path="C:\\Users\\felie\\PycharmProjects\\walkerKinematicReconstruction\\walker\\LAO.bioMod",
        unit="mm",
    )

    interface.add_device(
        12,
        name="Treadmill",
        device_type=DeviceType.Generic,
        rate=2000,
        data_buffer_size=int(3000 * nb_second),
        processing_window=int(3000 * nb_second),
        device_data_file_key="treadmill",
    )

    sending_started = False
    threshold = 30  # Seuil de force verticale
    Fs_PF = 2000

    previous_fz = 0  # Valeur initiale de Fz (inférieur au seuil)

    while True:
        tic = time.perf_counter()
        dataforce = interface.get_device_data(device_name="Treadmill")

        # Calcul de la force verticale moyenne actuelle
        current_fz = np.mean(dataforce[2])

        if not sending_started:
            # Détecter le passage de inférieur à supérieur au seuil
            if detect_start(previous_fz, current_fz, threshold):
                sending_started = True
                print("Démarrage de l'envoi des données.")
        else:
            connection, message = server.client_listening()  # Non-bloquant
            Q, _ = interface.get_kinematics_from_markers(marker_set_name="markers", get_markers_data=True)
            mark_tmp, _ = interface.get_marker_set_data()

            if connection:
                dataAll = {"Force": dataforce, "Markers": mark_tmp, "Angle": Q[:, -1], "MarkersNames": mks_name}
                server.send_data(dataAll, connection, message)

        # Mettre à jour la valeur précédente de Fz
        previous_fz = current_fz
        dataforce=[]
        loop_time = time.perf_counter() - tic
        real_time_to_sleep = max(0, 0.01 - loop_time)  # Fréquence de 100 Hz
        if real_time_to_sleep > 0:
            time.sleep(real_time_to_sleep)
