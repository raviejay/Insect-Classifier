import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from backend.prediction import model, class_names, predict_image

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the image
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        panel.configure(image=img_tk)
        panel.image = img_tk
        
        # Predict the image
        predicted_class = predict_image(model, file_path, class_names)
        result_label.config(text=f'Predicted Class: {predicted_class}')

def start_tkinter_app():
    global panel, result_label

    root = tk.Tk()
    root.title("Insect Detector")

    upload_button = tk.Button(root, text="Upload Image", command=upload_and_predict)
    upload_button.pack()

    panel = tk.Label(root)
    panel.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    root.mainloop()
