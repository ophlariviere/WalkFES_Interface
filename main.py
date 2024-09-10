import sys
import threading
from PyQt5.QtWidgets import QApplication
from visualization import VisualizationWidget
from data_receiver import DataReceiver


def main():
    # Définition des paramètres du serveur
    server_ip = "127.0.0.1"
    server_port = 50000

    # Créer une application PyQt5
    app = QApplication(sys.argv)

    # Créer une instance du widget de visualisation et l'afficher
    window = VisualizationWidget()
    window.show()

    # Créer une instance du récepteur de données et lier le widget à celui-ci
    data_receiver = DataReceiver(server_ip, server_port, window)

    # Démarrer la réception des données dans un thread séparé pour éviter de bloquer l'interface
    receiving_thread = threading.Thread(target=data_receiver.start_receiving)
    receiving_thread.daemon = True  # Marquer le thread comme un thread de démon pour qu'il se ferme avec l'interface
    receiving_thread.start()

    # Lancer la boucle d'événements de l'interface PyQt5
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
