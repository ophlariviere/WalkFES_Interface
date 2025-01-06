import sys
import time
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import pyqtSignal, QObject

from data_receiver import DataReceiver



class SignalEmitter(QObject):
    signal_triggered = pyqtSignal(float)

class ComputationThread(threading.Thread):
    def __init__(self, data_queue, user_input, signal_emitter):
        super().__init__()
        self.data_queue = data_queue
        self.user_input = user_input
        self.signal_emitter = signal_emitter
        self.running = True

    def run(self):
        while self.running:
            if self.data_queue:
                stream_data = self.data_queue.pop(0)
                try:
                    user_value = float(self.user_input.get())
                except ValueError:
                    user_value = 0  # Default to 0 if user input is invalid

                derived_metric = stream_data + user_value  # Example computation
                print(f"Derived metric: {derived_metric}")

                if derived_metric > 15:  # Example condition to send signal
                    self.signal_emitter.signal_triggered.emit(derived_metric)

            time.sleep(0.1)

    def stop(self):
        self.running = False

class MyApp(QWidget):
    def __init__(self, signal_emitter):
        super().__init__()
        self.initUI()
        self.signal_emitter = signal_emitter
        self.signal_emitter.signal_triggered.connect(self.handle_signal)

    def initUI(self):
        self.layout = QVBoxLayout()

        self.label = QLabel("Enter a number:")
        self.layout.addWidget(self.label)

        self.input_field = QLineEdit()
        self.layout.addWidget(self.input_field)

        self.output_label = QLabel("Signal: None")
        self.layout.addWidget(self.output_label)

        self.setLayout(self.layout)
        self.setWindowTitle("PyQt Multithreading Example")

    def handle_signal(self, value):
        self.output_label.setText(f"Signal: Triggered with value {value}")

    def get_user_input(self):
        return self.input_field.text()

def main():

    # Définition des paramètres du serveur
    server_ip = "192.168.0.1" #   # "192.168.0.1" 127.0.0.1# Adresse IP du serveur
    server_port = 7  # 7  # 50000 Port à utiliser

    # Créez une application PyQt5
    app = QApplication(sys.argv)




    # Créez une instance de DataReceiver
    data_receiver = DataReceiver(server_ip, server_port, visualization_widget)


    signal_emitter = SignalEmitter()
    gui = MyApp(signal_emitter)
    gui.show()


    computation_thread = ComputationThread(data_queue, user_input, signal_emitter)

    data_receiver.start_receiving()
    computation_thread.start()

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        data_receiver.stop_receiving()
        computation_thread.stop()
        data_receiver.join()
        computation_thread.join()

if __name__ == "__main__":
    main()
