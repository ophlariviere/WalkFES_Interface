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
    server_ip = "127.0.0.1"  # Utiliser l'adresse IP du serveur
    server_port = 50000  # Port à utiliser

    # Créer une application PyQt5
    app = QApplication(sys.argv)

    # Créer une instance du widget de visualisation et l'afficher
    window = VisualizationWidget()
    window.show()

    # Créer une instance du récepteur de données et lier le widget à celui-ci
    data_receiver = DataReceiver(server_ip, server_port, window)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(data_receiver.start_receiving)
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
