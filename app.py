from tkinter import *

BG_VIOET = "#5D2A42"
BG_COLOR = "#BB8F89"  # Corrected color code
TEXT_COLOR = "#FFF9EC"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place()
if __name__ == "__main__":
    app = ChatApplication()
    app.run()
