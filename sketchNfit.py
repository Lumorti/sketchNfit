from tkinter import *
import numpy as np
from tkinter import filedialog
import os

# The main class
class Paint(object):

    # Called on start
    def __init__(self):
        self.root = Tk()

        # Vars
        self.points = []
        self.data = np.array([])
        self.monomialNames = []
        self.line_width = 5.0
        self.paint_color = 'black'

        # Create the file load button
        self.load_button = Button(self.root, text='load from monomials file', command=self.load_file)
        self.load_button.grid(row=0, column=2, padx=10, pady=5)

        # Create the fit button
        self.load_button = Button(self.root, text='fit', command=self.fit)
        self.load_button.grid(row=0, column=3, padx=10, pady=5)

        # Create the canvas
        self.c = Canvas(self.root, bg='white', width=1000, height=500)
        self.c.grid(row=1, columnspan=6)

        # Start everything
        self.old_x = None
        self.old_y = None
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.root.mainloop()

    # Load the file
    def load_file(self):

        # The list of monomials
        self.monomialFile = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
        print("Loading monomials file: ", self.monomialFile)
        self.monomialNames = np.loadtxt(self.monomialFile, dtype=str)
        for i in range(len(self.monomialNames)):
            self.monomialNames[i] = self.monomialNames[i].replace("s_[", "")
            self.monomialNames[i] = self.monomialNames[i].replace("]", "")
            self.monomialNames[i] = self.monomialNames[i].replace("x,", "x")
            self.monomialNames[i] = self.monomialNames[i].replace("y,", "y")
            self.monomialNames[i] = self.monomialNames[i].replace("z,", "z")
        print("Monomial list has shape: ", self.monomialNames.shape)

        # Then load all numpy files in the same directory
        self.data = []
        lastSlash = self.monomialFile.rfind("/")
        path = self.monomialFile[:lastSlash+1]
        print("Loading numpy files from: ", path)
        for filename in os.listdir(path):
            if filename.endswith(".npy"):
                newData = np.load(path + filename)
                numInFilename = ""
                for char in filename.replace(".npy", ""):
                    if char.isdigit() or char == ".":
                        numInFilename += char
                numInFilename = float(numInFilename)
                newData = np.insert(newData, 0, numInFilename)
                print("Data from file: ", filename, " has shape: ", newData.shape)
                self.data.append(newData)
        self.data = np.array(self.data)
        print("Overall data has shape: ", self.data.shape)

    # Paint the line
    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    # Reset the line drawing
    def reset(self, event):
        self.old_x, self.old_y = None, None

    # Fit the data
    def fit(self):

        # For now just plot the first value
        self.fittedData = []
        for i in range(len(self.data)):
            self.fittedData.append([self.data[i][0], self.data[i][1]])
        self.fittedData = np.array(self.fittedData)

        # Determine the range of the data
        minX = np.min(self.fittedData[:,0])
        maxX = np.max(self.fittedData[:,0])
        minY = np.min(self.fittedData[:,1])
        maxY = np.max(self.fittedData[:,1])

        # A bit bigger than the canvas to leave some space
        height = 400
        width = 900
        offsetX = 50
        offsetY = 50

        # Plot the points
        for i in range(len(self.fittedData)):
            self.fittedData[i][0] = offsetX + width * (self.fittedData[i][0] - minX) / (maxX - minX)
            self.fittedData[i][1] = offsetY + height * (self.fittedData[i][1] - minY) / (maxY - minY)
            self.c.create_oval(self.fittedData[i][0], self.fittedData[i][1], self.fittedData[i][0], self.fittedData[i][1], fill='red', width=5)

# Run the program
if __name__ == '__main__':
    Paint()

