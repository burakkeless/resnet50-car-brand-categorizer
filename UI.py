import tkinter as tk

import main
from main import load_model_and_predict
from tkinter import filedialog, font
from tkinter import messagebox

img_filepath = ""
h5_filepath = ""


def select_imgfile():
    global img_filepath
    img_filepath = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg")])
    if img_filepath:
        selected_imgfile_label.config(text="Selected Image File: " + img_filepath)


def select_h5file():
    global h5_filepath
    # opens file selector with given path for easy to use
    h5_filepath = filedialog.askopenfilename(initialdir="/Users/burakkeles/PycharmProjects/car_brand_predictor/working",
                                             filetypes=[("h5 files", "*.h5")])
    if h5_filepath:
        selected_modfile_label.config(text="Selected Model File: " + h5_filepath)


def UI_solver():
    def path_sender(img_filepath, h5_filepath):
        load_model_and_predict(img_filepath, h5_filepath)

    global img_filepath, h5_filepath
    if not img_filepath or not h5_filepath:
        messagebox.showwarning("Error", "Please select both image and model files !")
    else:
        path_sender(img_filepath, h5_filepath)


root = tk.Tk()
root.title("Brand Selector")
root.geometry("800x800")

# Program name
prog_label = tk.Label(root, text="Car Brand Categorizer",font=("Helvetica", 24))
prog_label.place(relx=0.5, rely=0.1, anchor="center")
#sign text
sign_font = font.Font(family="Arial", size=11, slant="italic")
sign_label = tk.Label(root,text="by Burak Keles",font=sign_font)
sign_label.place(relx=0.71, rely=0.106,anchor="center")

machine_button = tk.Button(root,
                           text="Create machine learning model with net50 ",
                           font=("Arial",12),
                           command= main.learner_control_train)
machine_button.place(relx=0.5, rely=0.2, anchor="center")

# image text
label_image = tk.Label(root, text="Select Image:", font=("Helvetica", 16))
label_image.place(relx=0.12, rely=0.4)
# image button
select_img_button = tk.Button(root, text="Browse image", font=("Helvetica", 14), command=select_imgfile)
select_img_button.place(relx=0.26, rely=0.399)
# image path
selected_imgfile_label = tk.Label(root, text="", font=("Helvetica", 12), fg="red")
selected_imgfile_label.place(relx=0.12, rely=0.45)

# model text
label_model = tk.Label(root, text="Select model:", font=("Helvetica", 16))
label_model.place(relx=0.12, rely=0.52)
# model button
select_mod_button = tk.Button(root, text="Browse model", font=("Helvetica", 14), command=select_h5file)
select_mod_button.place(relx=0.258, rely=0.52)
# model path
selected_modfile_label = tk.Label(root, text="", font=("Helvetica", 12), fg="red")
selected_modfile_label.place(relx=0.12, rely=0.56)

# solver button
solver_button = tk.Button(root, text="Find the brand !", font=("Helvetica", 16), command=UI_solver)
solver_button.place(relx=0.5, rely=0.75, anchor="center")

root.mainloop()
