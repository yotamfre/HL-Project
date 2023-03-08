from tkinter import *
import numpy as np
from LogisticRegression import *

def makeMainMenu():
    frm = Tk()

    canvas_width = 200
    canvas_height = 400

    buttonHeight = 2
    buttonWidth = 10

    message = Label(frm, text = "Your number is:",font = "Bold", bg = "#b6d7a8", width = 200, height = 15)
    message.pack()

    canvas = Canvas(frm,
            width=canvas_width,
            height=canvas_height)
    canvas.pack(expand=YES, fill=BOTH)

    def resetClicked():
        print("reset has been clicked")

    def AddDigitClicked():
        frm.destroy()
        makeCanvas()

    ResetButton = Button(frm, text = "Rest", font = "Bold", bg = "Blue", command = resetClicked, width = buttonWidth, height = buttonHeight)
    ResetButton.place(x = 25,y = 600)

    AddDigitButton = Button(frm, text = "Add Digit", font = "Bold", bg = "Blue", command = AddDigitClicked, width = buttonWidth, height = buttonHeight)
    AddDigitButton.place(x = 1675,y = 600)

def makeCanvas():
    pixelList = []
    frm = Tk()

    canvas_width = 200
    canvas_height = 400

    buttonHeight = 2
    buttonWidth = 10

    message = Label(frm, text="write the digit you wish to add", font = "Bold", bg = "#b6d7a8", width = 200, height=15)
    message.pack(side=TOP)

    canvas = Canvas(frm,
            width=canvas_width,
            height=canvas_height)
    canvas.pack(expand=YES, fill=BOTH)

    def paint(event):
        blackHex = "#000000"
        x1, y1 = (event.x -4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        canvas.create_oval(x1, y1, x2, y2, fill=blackHex)

        if (event.x > 600 and event.x <= 1800):
            pixelList.append((event.x, event.y))

    def cancelClicked():
        frm.destroy()
        makeMainMenu()

    def AddClicked():
        makeImage(pixelList)
        vec = model.toVector(image)
        print(model.predict(vec))
        

    cancelButton = Button(frm, text = "Cancel", font = "Bold", bg = "Blue", command = cancelClicked, width = buttonWidth, height = buttonHeight)
    cancelButton.place(x = 25,y = 600)

    AddButton = Button(frm, text = "Add", font = "Bold", bg = "Blue", command = AddClicked, width = buttonWidth, height = buttonHeight)
    AddButton.place(x = 1675,y = 600)

    canvas.bind("<B1-Motion>", paint)

#compresses the images
def makeImage(lst):
    image = np.zeros((28,28))
    image = image - 1

    for i in range(len(lst)):
        tup = lst.pop()
        image[int((tup[1] + 200) / 28)][int((tup[0] - 1200) / 28)] = 1


#code starts here
model = DigitRecognition()
model.train()
image = np.zeros((28,28))

makeMainMenu()
mainloop()