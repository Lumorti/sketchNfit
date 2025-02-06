from tkinter import *
import numpy as np
from tkinter import filedialog
import os
import math

# The main class
class Paint(object):

    # Called on start
    def __init__(self):
        self.root = Tk()

        # Vars
        self.points = []
        self.data = np.array([])
        self.monomialNames = []
        self.line_width = 3.0
        self.paint_color = 'black'
        self.monomialFile = None

        # Create the file load button
        self.load_button = Button(self.root, text='load from monomials.txt', command=self.load_file)
        self.load_button.grid(row=0, column=0, padx=10)

        # Create the reload button
        self.reload_button = Button(self.root, text='reload', command=self.reload)
        self.reload_button.grid(row=1, column=0, padx=10)

        # Create the fit button
        self.fit_button = Button(self.root, text='fit', command=self.fit)
        self.fit_button.grid(row=0, column=2, padx=10)

        # Create the clear button
        self.clear_button = Button(self.root, text='clear', command=self.clear)
        self.clear_button.grid(row=0, column=3, padx=10)

        # The checkbox to decide if we should use kernel or not
        self.kernel = IntVar()
        self.kernel.set(1)
        self.kernel_checkbox = Checkbutton(self.root, text="Use kernel", variable=self.kernel)
        self.kernel_checkbox.grid(row=0, column=4, padx=10)

        # Checkbox to decide if we do invidual normalization
        self.individual = IntVar()
        self.individual.set(0)
        self.individual_checkbox = Checkbutton(self.root, text="Individual normalization", variable=self.individual)
        self.individual_checkbox.grid(row=1, column=4, padx=10)

        # Create the random button
        self.random_button = Button(self.root, text='random', command=self.random)
        self.random_button.grid(row=0, column=5, padx=10)

        # Create the many random button
        self.many_random_button = Button(self.root, text='many random', command=self.randomMany)
        self.many_random_button.grid(row=1, column=5, padx=10, pady=10)

        # The slider for the monomial limit
        self.monomialLimit = Scale(self.root, from_=1, to=10, orient=HORIZONTAL, label="max monom")
        self.monomialLimit.set(10)
        self.monomialLimit.bind("<ButtonRelease-1>", lambda event: self.reload())
        self.monomialLimit.grid(row=0, column=7, padx=10)

        # Create the canvas
        self.c = Canvas(self.root, bg='white', width=1000, height=500)
        self.c.grid(row=2, columnspan=8)

        # Set the title
        self.root.title('Phase Transition Detection')

        # Start everything
        self.old_x = None
        self.old_y = None
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.root.mainloop()

    # Clear the canvas
    def clear(self):
        self.c.delete("all")
        self.drawAxes()

    # Reload the file
    def reload(self):
        self.load_file(self.monomialFile)

    # Load the file
    def load_file(self, filename=None):
        self.points = []

        # The list of monomials
        if filename is None:
            self.monomialFile = filedialog.askopenfilename(filetypes=[('Text files', 'monomials.txt')], title="Select the monomials file", initialdir=os.getcwd()+"/data/")
            print("Loading monomials file: ", self.monomialFile)
            if self.monomialFile == "":
                return
        else:
            self.monomialFile = filename
        self.monomialNames = np.loadtxt(self.monomialFile, dtype=str)
        for i in range(len(self.monomialNames)):
            self.monomialNames[i] = self.monomialNames[i].replace("s_[", "")
            self.monomialNames[i] = self.monomialNames[i].replace("]", "")
            self.monomialNames[i] = self.monomialNames[i].replace("x,", "x")
            self.monomialNames[i] = self.monomialNames[i].replace("y,", "y")
            self.monomialNames[i] = self.monomialNames[i].replace("z,", "z")
        print("Monomial list has shape: ", self.monomialNames.shape)

        # Make sure we have a monomial file
        if len(self.monomialFile) == 0:
            return

        # Get the file name without the path and set the title
        justFile = self.monomialFile.replace("/monomials.txt", "")
        justFile = justFile[justFile.rfind("/")+1:]
        self.root.title(justFile)

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

        for i in range(10):
            print(self.monomialNames[i], len(self.monomialNames[i]))

        # Only keep monomials with a limited size
        removed = 0
        limit = self.monomialLimit.get() * 16
        toKeep = [True for i in range(len(self.monomialNames))]
        for i in range(len(self.monomialNames)):
            if len(self.monomialNames[i]) > limit:
                toKeep[i] = False
                removed += 1
        self.monomialNames = self.monomialNames[toKeep]
        toKeep.insert(0, True)
        self.data = self.data[:,toKeep]
        print("Removed ", removed, " monomials with size > ", limit)

        # Initial fit
        self.fit()

    # Paint the line
    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
        self.points.append([event.x, event.y])

    # Reset the line drawing
    def reset(self, event):
        self.old_x, self.old_y = None, None

    # Draw the axes
    def drawAxes(self):

        # A bit bigger than the canvas to leave some space
        self.offsetX = 100
        self.offsetY = 50
        self.height = 500 - 2 * self.offsetY
        self.width = 1000 - 2 * self.offsetX

        # Determine the x range of the data
        self.minX = np.min(self.data[:,0])
        self.maxX = np.max(self.data[:,0])
        self.minY = 0
        self.maxY = 1

        # Draw the axes
        self.c.create_line(self.offsetX, self.offsetY, self.offsetX, self.offsetY + self.height, width=2)
        self.c.create_line(self.offsetX, self.offsetY + self.height, self.offsetX + self.width, self.offsetY + self.height, width=2)

        # Label the axes
        self.c.create_text(self.offsetX - 10, self.offsetY, text=str(round(self.maxY, 2)), anchor=E)
        self.c.create_text(self.offsetX - 10, self.offsetY + self.height - 5, text=str(round(self.minY, 2)), anchor=E)

        # Add labels every so often on the X
        numLabels = 10
        for i in range(numLabels+1):
            xLoc = 10 + self.offsetX + self.width * i / numLabels
            xVal = self.minX + (self.maxX - self.minX) * i / numLabels
            self.c.create_text(xLoc, self.offsetY + self.height + 10, text=str(round(xVal, 2)), anchor=E)

    # Plot a random polynomial of the data
    def random(self):

        # Same setup as the regression
        numMonomials = len(self.monomialNames)
        numCoeffs = 1 + numMonomials
        A = np.zeros((len(self.data), numCoeffs))
        b = np.zeros((len(self.data), 1))
        for i in range(len(self.data)):
            A[i,0] = 1
            for j in range(numMonomials):
                A[i,j+1] = self.data[i,j+1]

        # Random polynomial in the moments
        randRange = 2
        x = randRange * np.random.rand(numCoeffs, 1) - randRange/2.0

        # Get the y values
        yVals = A @ x
        minY = np.min(yVals)
        maxY = np.max(yVals)
        yVals = (yVals - minY) / (maxY - minY)
        self.fittedData = []
        xVals = self.data[:,0]
        self.fittedData = np.column_stack((xVals, yVals))

        # Sort the data by X
        self.fittedData = self.fittedData[self.fittedData[:,0].argsort()]

        # If told to use kernel
        if self.kernel.get():

            # Apply the "kernel" to the data
            self.newData = np.zeros((len(self.fittedData), 2))
            self.newData[:,0] = self.fittedData[:,0]
            self.kernelPower = 4
            for i in range(0, len(self.fittedData)-1):
                self.newData[i,1] = np.abs(self.fittedData[i,1] - self.fittedData[i+1,1])**self.kernelPower
            self.fittedData = self.newData

            # Renormalize
            minY = np.min(self.fittedData[:,1])
            maxY = np.max(self.fittedData[:,1])
            for i in range(len(self.fittedData)):
                self.fittedData[i][1] = (self.fittedData[i][1] - minY) / (maxY - minY)

            # Plot again
            prevX = None
            prevY = None
            for i in range(len(self.fittedData)):
                xLoc = self.offsetX + self.width * (self.fittedData[i][0] - self.minX) / (self.maxX - self.minX)
                yLoc = self.height * (1.0-self.fittedData[i][1]) + self.offsetY
                if prevX is not None and prevY is not None:
                    self.c.create_line(xLoc, yLoc, prevX, prevY, width=2, fill='blue')
                self.c.create_oval(xLoc, yLoc, xLoc, yLoc, fill='blue', width=5, outline='blue')
                prevX = xLoc
                prevY = yLoc

        # Otherwise plot the normal data
        else:

            # Plot the fitted points
            prevX = None
            prevY = None
            for i in range(len(self.fittedData)):
                xLoc = self.offsetX + self.width * (self.fittedData[i][0] - self.minX) / (self.maxX - self.minX)
                yLoc = self.height * (1.0-self.fittedData[i][1]) + self.offsetY
                if prevX is not None and prevY is not None:
                    self.c.create_line(xLoc, yLoc, prevX, prevY, width=2, fill='red')
                self.c.create_oval(xLoc, yLoc, xLoc, yLoc, fill='red', width=5, outline='red')
                prevX = xLoc
                prevY = yLoc

    # Plot many random polynomials
    def randomMany(self):

        # Same setup as the regression
        numMonomials = len(self.monomialNames)
        numCoeffs = 1 + numMonomials
        A = np.zeros((len(self.data), numCoeffs))
        b = np.zeros((len(self.data), 1))
        for i in range(len(self.data)):
            A[i,0] = 1
            for j in range(numMonomials):
                A[i,j+1] = self.data[i,j+1]

        xVals = self.data[:,0]
        numRandom = 200
        self.fittedData = np.column_stack((xVals, np.zeros((len(xVals), numRandom))))
        for i in range(numRandom):

            # Random polynomial in the moments
            randRange = 2
            x = randRange * np.random.rand(numCoeffs, 1) - randRange/2.0
            x = x / np.linalg.norm(x)

            # Get the y values
            yVals = A @ x
            self.fittedData[:,i+1] = yVals.flatten()

        # Sort the data by X
        self.fittedData = self.fittedData[self.fittedData[:,0].argsort()]

        # If told to use kernel
        if self.kernel.get():

            # Apply the "kernel" to the data
            self.newData = np.zeros(self.fittedData.shape)
            self.newData[:,0] = self.fittedData[:,0]
            self.kernelPower = 2
            for i in range(self.fittedData.shape[1]-1):
                self.newData[:,i+1] = np.abs(self.fittedData[:,i+1] - np.roll(self.fittedData[:,i+1], -1))**self.kernelPower
                self.newData[-1,i+1] = 0
            self.fittedData = self.newData

            # Renormalize
            if self.individual.get():
                for i in range(1, self.fittedData.shape[1]):
                    minY = np.min(self.fittedData[:,i])
                    maxY = np.max(self.fittedData[:,i])
                    self.fittedData[:,i] = (self.fittedData[:,i] - minY) / (maxY - minY)
            else:
                minY = np.min(self.fittedData[:,1:])
                maxY = np.max(self.fittedData[:,1:])
                self.fittedData[:,1:] = (self.fittedData[:,1:] - minY) / (maxY - minY)

            # Plot again
            for j in range(1, self.fittedData.shape[1]):
                prevX = None
                prevY = None
                for i in range(len(self.fittedData)):
                    xLoc = self.offsetX + self.width * (self.fittedData[i,0] - self.minX) / (self.maxX - self.minX)
                    yLoc = self.height * (1.0-self.fittedData[i,j]) + self.offsetY
                    if prevX is not None and prevY is not None:
                        self.c.create_line(xLoc, yLoc, prevX, prevY, width=2, fill='blue')
                    self.c.create_oval(xLoc, yLoc, xLoc, yLoc, fill='blue', width=5, outline='blue')
                    prevX = xLoc
                    prevY = yLoc

        # Otherwise plot the normal data
        else:

            # Renormalize
            if self.individual.get():
                for i in range(1, self.fittedData.shape[1]):
                    minY = np.min(self.fittedData[:,i])
                    maxY = np.max(self.fittedData[:,i])
                    self.fittedData[:,i] = (self.fittedData[:,i] - minY) / (maxY - minY)
            else:
                minY = np.min(self.fittedData[:,1:])
                maxY = np.max(self.fittedData[:,1:])
                self.fittedData[:,1:] = (self.fittedData[:,1:] - minY) / (maxY - minY)

            # Plot the fitted points
            for j in range(1, self.fittedData.shape[1]):
                prevX = None
                prevY = None
                for i in range(len(self.fittedData)):
                    xLoc = self.offsetX + self.width * (self.fittedData[i,0] - self.minX) / (self.maxX - self.minX)
                    yLoc = self.height * (1.0-self.fittedData[i,j]) + self.offsetY
                    if prevX is not None and prevY is not None:
                        self.c.create_line(xLoc, yLoc, prevX, prevY, width=2, fill='red')
                    self.c.create_oval(xLoc, yLoc, xLoc, yLoc, fill='red', width=5, outline='red')
                    prevX = xLoc
                    prevY = yLoc

    # Fit the data
    def fit(self):

        # Refresh the canvas
        self.clear()

        # Determine the x range of the data
        if len(self.data) == 0:
            self.minX = 0
            self.maxX = 1
        else:
            self.minX = np.min(self.data[:,0])
            self.maxX = np.max(self.data[:,0])

        # Scale the drawing X values to the same range
        for i in range(len(self.points)):
            self.points[i][0] = self.minX + (self.points[i][0] - self.offsetX) * (self.maxX - self.minX) / self.width

        # Sample the user's drawing
        self.drawingSamples = []
        for i in range(len(self.data)):

            # Find the closest point in X
            xToFind = self.data[i,0]
            closestPoint = None
            closestDist = 1000000
            for j in range(len(self.points)):
                dist = abs(self.points[j][0] - xToFind)
                if dist < closestDist:
                    closestDist = dist
                    closestPoint = self.points[j]

            # Add the closest point
            if closestPoint is not None:
                self.drawingSamples.append([xToFind, closestPoint[1] - self.offsetY])

        # Make sure we have drawing samples
        if len(self.drawingSamples) == 0:
            return

        # Convert to numpy
        self.drawingSamples = np.array(self.drawingSamples)

        # Normalize the drawing Y values
        for i in range(len(self.drawingSamples)):
            self.drawingSamples[i][1] = self.drawingSamples[i][1] / self.height

        # Invert the Y values
        for i in range(len(self.drawingSamples)):
            self.drawingSamples[i][1] = 1.0 - self.drawingSamples[i][1]

        # Now we do polynomial regression
        # We have Ax = b
        # A is the matrix of monomials and their values for each data point (rows)
        # b is the vector of the ideal y values from the drawing samples
        # x is the vector of coefficients for each monomial
        numMonomials = len(self.monomialNames)
        numCoeffs = 1 + numMonomials
        A = np.zeros((len(self.drawingSamples), numCoeffs))
        b = np.zeros((len(self.drawingSamples), 1))
        for i in range(len(self.data)):
            A[i,0] = 1
            for j in range(numMonomials):
                A[i,j+1] = self.data[i,j+1]

            # Find the corresponding drawing sample
            for k in range(len(self.drawingSamples)):
                if self.drawingSamples[k][0] == self.data[i][0]:
                    b[i] = self.drawingSamples[k][1]
                    break

        # Solve the system
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        # Output mean squared error
        yVals = A @ x
        error = np.sum((yVals - b) ** 2) / len(b)
        print("Mean squared error: ", error)

        # Now we plot the fitted data
        self.fittedData = []
        xVals = self.data[:,0]
        self.fittedData = np.column_stack((xVals, yVals))

        # At this point everything is normalized
        self.minY = 0
        self.maxY = 1

        print("Drawing samples: ", self.drawingSamples.shape)
        print("Fitted data: ", self.fittedData.shape)

        # Sort the data by X
        self.fittedData = self.fittedData[self.fittedData[:,0].argsort()]
        if len(self.drawingSamples) > 0:
            self.drawingSamples = self.drawingSamples[self.drawingSamples[:,0].argsort()]

        # Plot the fitted points
        for i in range(len(self.fittedData)):
            xLoc = self.offsetX + self.width * (self.fittedData[i][0] - self.minX) / (self.maxX - self.minX)
            yLoc = self.height * (1.0 - self.fittedData[i][1]) + self.offsetY
            self.c.create_oval(xLoc, yLoc, xLoc, yLoc, fill='red', width=5, outline='red')

        # Plot the drawing samples
        for i in range(len(self.drawingSamples)):
            xLoc = self.offsetX + self.width * (self.drawingSamples[i][0] - self.minX) / (self.maxX - self.minX)
            yLoc = self.height * (1.0 - self.drawingSamples[i][1]) + self.offsetY
            self.c.create_oval(xLoc, yLoc, xLoc, yLoc, fill='blue', width=5, outline='blue')

# Run the program
if __name__ == '__main__':
    Paint()

