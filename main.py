import sys
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import QApplication
from visualization import VisualizationWidget
from data_receiver import DataReceiver
import logging


# Configure le logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Définition des paramètres du serveur
    server_ip = "192.168.0.1"  # Utiliser l'adresse IP du serveur
    server_port = 7  # Port à utiliser

    # Créer une application PyQt5
    app = QApplication(sys.argv)

    # Créez une instance du widget de visualisation et affichez-la
    visualization_widget = VisualizationWidget()
    visualization_widget.show()

    # Créez une instance de DataReceiver et liez le widget de visualisation à celui-ci
    data_receiver = DataReceiver(server_ip, server_port, visualization_widget)

    # Utilisez ThreadPoolExecutor pour exécuter les tâches en parallèle
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(data_receiver.start_receiving)
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
