# Import necessary libraries
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load the MNIST dataset
dat = pd.read_csv('#add your location of your dataset')
dat = np.array(dat)
m, n = dat.shape
np.random.shuffle(dat)  # Shuffle data to prevent training bias

# Split into validation and training sets
val = dat[0:1000].T
v_y = val[0]
v_x = val[1:n] / 255.0  # Normalize to [0,1]

trn = dat[1000:m].T
t_y = trn[0]
t_x = trn[1:n] / 255.0

def init_params():
    # Initialize NN with 784->10->10 architecture
    w1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)

def fwd(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    ohy = np.zeros((y.max() + 1, y.size))
    ohy[y, np.arange(y.size)] = 1
    return ohy

def d_relu(z):
    return (z > 0).astype(float)

def bwd(z1, a1, z2, a2, w2, x, y):
    m = y.size
    ohy = one_hot(y)
    dz2 = a2 - ohy
    dw2 = (1 / m) * dz2.dot(a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = w2.T.dot(dz2) * d_relu(z1)
    dw1 = (1 / m) * dz1.dot(x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2

def upd(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    return w1, b1, w2, b2

def get_pred(a2):
    return np.argmax(a2, axis=0)

def get_acc(p, y):
    return np.mean(p == y)

def gd(x, y, itr, lr):
    w1, b1, w2, b2 = init_params()
    for i in range(itr):
        z1, a1, z2, a2 = fwd(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = bwd(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = upd(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)
        if i % 50 == 0:
            p = get_pred(a2)
            acc = get_acc(p, y)
            print(f"Iteration: {i}, Accuracy: {acc*100:.2f}%")
    return w1, b1, w2, b2

# Train the neural network
w1, b1, w2, b2 = gd(t_x, t_y, 500, 0.1)

def pred(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = np.maximum(0, z1)
    z2 = w2.dot(a1) + b2
    exp = np.exp(z2 - np.max(z2, axis=0, keepdims=True))
    a2 = exp / np.sum(exp, axis=0, keepdims=True)
    p = np.argmax(a2, axis=0)
    return p[0]

class DrawApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Digit and Predict")

        self.cw = 280
        self.ch = 280
        self.canvas = tk.Canvas(master, width=self.cw, height=self.ch, bg='black')
        self.canvas.pack()

        self.clr_btn = tk.Button(master, text="Clear", command=self.clr_canvas)
        self.clr_btn.pack()

        self.pred_btn = tk.Button(master, text="Predict", command=self.pred_digit)
        self.pred_btn.pack()

        self.res_lbl = tk.Label(master, text="Draw a digit and click Predict")
        self.res_lbl.pack()

        self.img = Image.new("L", (self.cw, self.ch), color=0)
        self.draw = ImageDraw.Draw(self.img)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.lx, self.ly = None, None

    def paint(self, event):
        x, y = event.x, event.y
        r = 8  
        if self.lx and self.ly:
            self.canvas.create_line(self.lx, self.ly, x, y, fill='white', width=r*2)
            self.draw.line([self.lx, self.ly, x, y], fill=255, width=r*2)
        else:
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
            self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        self.lx, self.ly = x, y

    def clr_canvas(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (self.cw, self.ch), color=0)
        self.draw = ImageDraw.Draw(self.img)
        self.res_lbl.config(text="Draw a digit and click Predict")
        self.lx, self.ly = None, None

    def prep_img(self):
        img = self.img.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.array(img) / 255.0
        arr = arr.reshape(784, 1)
        return arr

    def pred_digit(self):
        x = self.prep_img()
        p = pred(w1, b1, w2, b2, x)
        self.res_lbl.config(text=f"Predicted Digit: {p}")
        self.lx, self.ly = None, None


if __name__ == "__main__":
    import numpy as np
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()

