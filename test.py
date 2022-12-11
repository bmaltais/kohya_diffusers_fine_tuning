import tkinter as tk
from tkinter import ttk

# create a window
window = tk.Tk()

# prevent the user from resizing the window
window.resizable(False, False)

# set the default padding for all columns to 5 pixels on the left and right sides
for i in range(2):
    window.columnconfigure(i, minsize=5, pad=5)

# set the default padding for all rows to 5 pixels on the top and bottom sides
for i in range(3):
    window.rowconfigure(i, minsize=5, pad=5)

# create a notebook widget to hold the tabs
notebook = ttk.Notebook(window, style="my.TNotebook")

# create a style for the tabs
style = ttk.Style()

# set the background color of the tabs to dark gray
style.configure("my.TNotebook.Tab", background="dark gray")

# set the border width of the tabs to 1 pixel
style.configure("my.TNotebook.Tab", borderwidth=1)

# give the tabs a 3D appearance by setting the relief to "raised"
style.configure("my.TNotebook.Tab", relief="raised")

# create a tab for example 1
example1_tab = ttk.Frame(notebook)

# create a label for the word field
word_label = tk.Label(example1_tab, text="Word:")
word_label.grid(row=0, column=0)

# create a text field where the user can type a word
word_field = tk.Entry(example1_tab)
word_field.grid(row=0, column=1)

# create a label for the result field
result_label = tk.Label(example1_tab, text="Result:")
result_label.grid(row=1, column=0)

# create a text field where the result will be displayed
result_field = tk.Entry(example1_tab)
result_field.grid(row=1, column=1)

# create a button that the user can click to run the program
# use the lambda keyword to create a function that will be called when the user clicks the "Run" button
run_button = tk.Button(example1_tab, text="Run", command=lambda: run(word_field, result_field))
run_button.grid(row=2, column=0, columnspan=2)

# add the example 1 tab to the notebook
notebook.add(example1_tab, text="Example 1")

# create a tab for example 1
example2_tab = ttk.Frame(notebook)

# add the example 2 tab to the notebook
notebook.add(example2_tab, text="Example 2")

# add the notebook widget to the window
notebook.grid(row=0, column=0)

# define the function that will be called when the user clicks the "Run" button
def run(word_field, result_field):
    # get the word from the word field
    word = word_field.get()

    # reverse the letters in the word
    reversed_word = word[::-1]

    # display the reversed word in the result field
    result_field.insert(0, reversed_word)

# style the widgets

# set the font for all widgets to a sans-serif font with size 12
font = ("Arial", 12)

# set the background color of the window to light gray
window.configure(background="light gray")

# set the font and background color of the labels
word_label.configure(font=font, background="light gray")
result_label.configure(font=font, background="light gray")

# set the font, background color, and border width of the text fields
word_field.configure(font=font, background="red", bd=2)
result_field.configure(font=font, background="blue", bd=2)

# set the font and background color of the button
run_button.configure(font=font, background="light gray")

# start the event loop
window.mainloop()
