# main.py
import tkinter as tk
from interface_gui import App

def main():
    root = tk.Tk()
    app = App(root)
    root.geometry("1980x1080")
    root.mainloop()

if __name__ == "__main__":
    main()
