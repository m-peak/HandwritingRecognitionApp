import sys
import os
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import recognizer 
import utils

title = 'My Handwriting Recognition'
app_width = 1200
app_height = 560
canvas_width = 1200
canvas_height = 400

class App(Frame):

    r = 10 # line thickness
    model_names = ['model_pad_fill.h5', 'model.h5']
    elms = {
        'mac': {
            'label_results': 14,
            'labels': 16,
        },
        'win': {
            'label_results': 12,
            'labels': 14,
        },
    }
    os = ''
    results = []
    accuracies = []
    recogns = []

    ##################################################
    # Initialization                                 #
    ##################################################
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.os = utils.get_os()
        self.create_ui()
        self.results = [self.result0, self.result1]
        self.accuracies = [self.accuracy0, self.accuracy1]
        for i in range(len(self.results)):
            recogn = recognizer.recognizer(self.model_names[i])
            self.recogns.append(recogn)

    ##################################################
    # Create UI                                      #
    ##################################################
    def create_ui(self):
        self.x = self.y = 0

        label_results = self.elms[self.os]['label_results']
        labels = self.elms[self.os]['labels']

        style = Style() 
        style.configure('TButton', font=('Helvetica', label_results), borderwidth='1')
        style.map('TButton', foreground=[('active', '!disabled', '#999999')], background=[('active', '#CCCCCC')])

        # UI Elements
        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, bg='white', cursor='cross')
        self.label_result0 = tk.Label(self, text='Data Augmentation (rotation, shift, shear, zoom) + Fill & Pad', font=('Helvetica', label_results), foreground='#999999')
        self.label_result1 = tk.Label(self, text='Data Augmentation (rotation, shift, shear, zoom)', font=('Helvetica', label_results), foreground='#999999')
        self.label_result = tk.Label(self, text='Result: ', font=('Helvetica', labels), foreground='#999999')
        self.result0 = tk.Entry(self, width=40, textvariable='', font=('Helvetica', labels))
        self.result1 = tk.Entry(self, width=40, textvariable='', font=('Helvetica', labels))
        self.button_recognize = tk.ttk.Button(self, text='Recognize', style='TButton', command=self.recognize)
        self.label_accuracy = tk.Label(self, text='Accuracy: ', font=('Helvetica', labels), foreground='#999999')
        self.accuracy0 = tk.Entry(self, width=40, textvariable='', font=('Helvetica', labels))
        self.accuracy1 = tk.Entry(self, width=40, textvariable='', font=('Helvetica', labels))
        self.button_clear = tk.ttk.Button(self, text='Clear', style='TButton', command=self.clear)
        
        # Grid Structure
        self.canvas.grid(row=0, column=0, pady=2, padx=2, columnspan=6)
        self.label_result0.grid(row=1, column=1, pady=5, padx=5, sticky=W)
        self.label_result1.grid(row=1, column=2, pady=5, padx=5, sticky=W)
        self.label_result.grid(row=2, column=0, pady=5, padx=5, sticky=E)
        self.result0.grid(row=2, column=1, pady=5, padx=0, sticky=W)
        self.result1.grid(row=2, column=2, pady=5, padx=0, sticky=W)
        self.button_recognize.grid(row=2, column=3, pady=5, padx=0, sticky=W)
        self.label_accuracy.grid(row=3, column=0, pady=5, padx=5, sticky=E)
        self.accuracy0.grid(row=3, column=1, pady=5, padx=0, sticky=W)
        self.accuracy1.grid(row=3, column=2, pady=5, padx=0, sticky=W)
        self.button_clear.grid(row=3, column=3, pady=5, padx=0, sticky=W)
        self.canvas.bind('<B1-Motion>', self.draw)

    ##################################################
    # Draw Characters                                #
    ##################################################
    def draw(self, event):
        self.x = event.x
        self.y = event.y
        self.canvas.create_oval(self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r, fill='black')

    ##################################################
    # Recognize Handwriting                          #
    ##################################################
    def recognize(self):
        self.reset()
        app_title = app.master.title()
        winfo_id = self.canvas.winfo_id()
        im = utils.capture_handwriting(app_title, canvas_width, canvas_height, winfo_id)
        
        # Compare methods
        for i in range(len(self.recogns)):
            result,accuracies = self.recogns[i].recognize(im)
            result = result if result is not None else 'Unknown'
            accuracy = ''
            if len(accuracies) > 0:
                for acc in accuracies:
                    accuracy += (', ' if accuracy != '' else '') + (str(round(acc*100, 2)) + '%')
            else:
                accuracy = 'Unknown'

            '''
            # debugging
            print(result)
            '''

            self.results[i].insert(0, result)
            self.accuracies[i].insert(0, accuracy)

    ##################################################
    # Clear Characters                               #
    ##################################################
    def clear(self):
        self.canvas.delete('all')
        self.reset()

    ##################################################
    # Reset Result & Accuracy                        #
    ##################################################
    def reset(self):
        for result,accuracy in zip(self.results, self.accuracies):
            result.delete(0, END)
            accuracy.delete(0, END)

root = Tk()
app = App(master=root)
app.master.title(title)
app.master.maxsize(app_width, app_height)
app.mainloop()
root.destroy()
