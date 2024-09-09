from data_receiver import DataReceiver
from data_processor import DataProcessor

class MainApplication:
    """Classe principale de l'application pour gérer l'interface utilisateur et la logique de traitement."""

    def __init__(self, server_ip, server_port, read_frequency=100):
        self.server_ip = server_ip
        self.server_port = server_port
        self.read_frequency = read_frequency
        self.data_receiver = DataReceiver(server_ip, server_port, read_frequency)
        self.data_processor = DataProcessor()

        # Connecter le signal de réception des données au traitement des données
        self.data_receiver.data_received.connect(self.data_processor.process_data)

        self.data_receiver.connect_to_server()

    def start(self):
        """Démarre la boucle principale d'acquisition et de traitement."""
        self.data_receiver.start_receiving()


if __name__ == "__main__":
    # Exemple d'utilisation
    server_ip = "127.0.0.1"
    server_port = 50000
    read_frequency = 100

    app = MainApplication(server_ip, server_port, read_frequency)
    app.start()
