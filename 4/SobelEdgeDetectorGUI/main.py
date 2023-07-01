import tkinter
from tkinter import Tk, ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import tempfile
import subprocess
import shutil

PROGRAM_NAME = "SobelEdgeDetector"
input_filename = ""

def view_image(path: str):
    print("Viewing", path)
    img = Image.open(path)
    img.thumbnail((800, 800), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    image_viewer_label.configure(image=img)
    image_viewer_label.image = img

def run_edge_detector() -> str:
    working_dir = tempfile.mkdtemp()
    print("Running in", working_dir)
    shutil.copy(PROGRAM_NAME, working_dir)
    command = ["./" + PROGRAM_NAME, input_filename, image_alpha_var.get(), image_beta_var.get(), image_threshold_var.get()]
    subprocess.Popen(command, cwd=working_dir).wait()
    return working_dir

def run_and_get_result(result_filename: str):
    if input_filename == "":
        return
    running_path = run_edge_detector()
    view_image(running_path + "/" + result_filename)
    shutil.rmtree(running_path)

def grayscale_image():
    run_and_get_result("grayscale.png")

def brighten_image():
    run_and_get_result("brighten.png")

def edge_detect_image():
    run_and_get_result("G.png")

def select_input():
    """
    The main entry point of choose file click;
    This function gets the input file, gets the output filename and does the conversion
    """
    global input_filename
    filetypes = (
        ('JPG', '*.jpg'),
        ('PNG', '*.png'),
        ('WEBP', '*.webp'),
        ('All files', '*.*'),
    )
    input_file = fd.askopenfilename(
        title='Open input',
        filetypes=filetypes)
    if input_file == "":
        return  # Do nothing
    input_filename = input_file
    view_image(input_file)
    

root = Tk()
root.resizable(False, False)
root.title('Sobel Edge detector')
frm = ttk.Frame(root)
frm.grid(pady=10)
# Buttons
ttk.Button(frm, text="Select File", command=select_input).grid(column=0, row=0, padx=10, pady=10)
ttk.Button(frm, text="Grayscale", command=grayscale_image).grid(column=1, row=0, padx=10, pady=10)
ttk.Button(frm, text="Change Brightness", command=brighten_image).grid(column=2, row=0, padx=10, pady=10)
ttk.Button(frm, text="Edge Detect", command=edge_detect_image).grid(column=3, row=0, padx=10, pady=10)
# Parameters
image_alpha_var = tkinter.StringVar(root)
image_alpha_var.set("1")
ttk.Label(frm, text="Alpha").grid(column=0, row=1)
ttk.Spinbox(frm, from_=0, to=2, textvariable=image_alpha_var).grid(column=0, row=2, padx=10)
image_beta_var = tkinter.StringVar(root)
image_beta_var.set("0")
ttk.Label(frm, text="Beta").grid(column=1, row=1)
ttk.Spinbox(frm, from_=-255, to=255, textvariable=image_beta_var).grid(column=1, row=2, padx=10)
image_threshold_var = tkinter.StringVar(root)
image_threshold_var.set("70")
ttk.Label(frm, text="Threshold").grid(column=2, row=1)
ttk.Spinbox(frm, from_=0, to=255, textvariable=image_threshold_var).grid(column=2, row=2, padx=10)
# Viewer
image_viewer_label = ttk.Label(frm)
image_viewer_label.grid(column=0, row=3, padx=10, pady=10, columnspan=4)
root.mainloop()