# main.py
import tkinter as tk
from interface_gui import App

def main():
    root = tk.Tk()
    app = App(root)
    root.attributes('-fullscreen', True)
    root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
    root.bind('<F11>', lambda e: root.attributes('-fullscreen', not root.attributes('-fullscreen')))
    root.state('zoomed')
    root.mainloop()

if __name__ == "__main__":
    main()
