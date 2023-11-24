import tkinter as tk
from tkinter import filedialog
import pandas as pd
import joblib
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class ModelRunnerApp:

    def __init__(self, master):
        self.master = master
        self.master.title("FinPulse: AI-Powered Finance")
        self.dataset = None
        self.model = None
        self.data = None

        # Create GUI components

        image = Image.open("FinPulse Logo.jpg")  # Change the path to your image
        image = image.resize((280, 85))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(master, image=self.photo)
        self.image_label.pack(pady=10)

        self.load_data_button = tk.Button(master, text="Load Dataset", command=self.load_dataset)
        self.load_data_button.pack(pady=10)

        self.load_model_button = tk.Button(master, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.run_model_button = tk.Button(master, text="Run Model", command=self.run_model)
        self.run_model_button.pack(pady=10)

        self.dataset_text = tk.Text(master, height=10, width=100)
        self.dataset_text.pack(pady=10)

        self.display_graph_button = tk.Button(master, text="Display Graph", command=self.display_graph)
        self.display_graph_button.pack(pady=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.dataset = pd.read_csv(file_path)
            print("Dataset loaded successfully.")
            print(self.dataset.head())
            self.dataset_text.delete(1.0, tk.END)
            self.dataset_text.insert(tk.END, self.dataset.head().to_string())

    def load_model(self):
        file_path = filedialog.askopenfilename(title="Select Model", filetypes=[("Model files", "*.pkl")])
        if file_path:
            self.model = joblib.load(file_path)
            print("Model loaded successfully.")

    def run_model(self):
        if self.dataset is None or self.model is None:
            print("Please load both dataset and model.")
            return

        # Make predictions using the loaded model
        predictions = self.model.predict(self.dataset)
        predictions.tolist()

        vals = []
        dates = []

        for i in range(len(self.dataset)):
            st = str(self.dataset['Month'].loc[i]) + "/" + str(self.dataset['Day'].loc[i]) + "/" + str(
                self.dataset['Year'].loc[i])
            dates.append(st)

        feature_columns = ['Store', 'Month', 'Day', 'Year', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
                           'Unemployment']

        for index, row in self.dataset.iterrows():
            # Extract features from the current row, replace this with your actual feature extraction logic
            # For example, assuming 'feature_columns' is a list of column names used as features
            features = row[feature_columns].values.reshape(1, -1)

            # Make a prediction using the model
            vals.append(self.model.predict(features))

        vals = ["%.2f" % elem for elem in vals]

        self.data = pd.DataFrame({'Date': dates,
                                 'Predictions': vals})

        self.data['Predictions'] = pd.to_numeric(self.data['Predictions'])

        print(self.data)

        self.dataset_text.delete(1.0, tk.END)
        self.dataset_text.insert(tk.END, self.data)

    def display_graph(self):
        if self.dataset is None:
            print("Please load a dataset.")
            return

        # Assume the dataset has a numeric column named 'numeric_column'
        plt.figure(figsize=(6, 4))
        plt.title("Weekly Sales Predictions")
        plt.xlabel("Date")
        plt.ylabel("Weekly Sales ($USD)")
        dates = self.data['Date']
        vals = self.data['Predictions']
        plt.plot(dates, vals)

        # Display the graph in the Tkinter window
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.master)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('800x600')

    app = ModelRunnerApp(root)

    root.mainloop()
