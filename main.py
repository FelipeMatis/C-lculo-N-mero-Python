# main.py
import tkinter as tk
from interface_gui import App

def main():
    root = tk.Tk()
    app = App(root)

    try:
        root.state('zoomed')
    except Exception:
        try:
            root.attributes('-zoomed', True)
        except Exception:
            # fallback: definir tamanho padrão
            root.geometry("1200x760")

    # F11 alterna fullscreen
    def toggle_fullscreen(event=None):
        is_full = bool(root.attributes('-fullscreen'))
        root.attributes('-fullscreen', not is_full)
    root.bind('<F11>', toggle_fullscreen)

    # Esc desliga fullscreen (se estiver on)
    root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))

    root.mainloop()

if __name__ == "__main__":
    main()
