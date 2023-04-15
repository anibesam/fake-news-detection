import numpy as np
import pandas as pd
import json
import csv
import random
import tkinter as tk
from tkinter import ttk

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

import pprint
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
tf.disable_eager_execution()

# Reading the data
data = pd.read_csv("dataset.csv")

# Creating the GUI window
root = tk.Tk()
root.title("Detection of fake video news using CNN and XGBoost")

# Setting the window size to match the screen size
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (width, height))

# Creating the menu bar
menu_bar = tk.Menu(root)

# Creating the "About" menu
about_menu = tk.Menu(menu_bar, tearoff=0)
about_menu.add_command(label="About")

# Creating the "Developer" submenu
developer_menu = tk.Menu(about_menu, tearoff=0)
developer_menu.add_command(label="Developed by Anibe Samuel")
about_menu.add_cascade(label="Developer", menu=developer_menu)

# Adding the menus to the menu bar
menu_bar.add_cascade(label="About Project", menu=about_menu)

# Attaching the menu bar to the window
root.config(menu=menu_bar)

# Creating a tkinter Table to display the data
table = ttk.Treeview(root)
table.pack(expand=True, fill=tk.BOTH)

# Defining the columns of the table
table["columns"] = list(data.columns)

# Formatting the columns
for col in table["columns"]:
    table.column(col, width=100)
    table.heading(col, text=col)

# Inserting the data into the table
for i, row in data.iterrows():
    # Set the row color to blue for real news and white for fake news
    if row["label"] == "real":
        tags = ("blue",)
    else:
        tags = ()
    table.insert("", "end", text=str(i), values=list(row), tags=tags)

# Configure the row colors
table.tag_configure("blue", background="blue", foreground="white")

# Running the GUI loop
root.mainloop()
