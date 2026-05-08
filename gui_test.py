import tkinter as tk
root = tk.Tk()
root.title("System Test")
label = tk.Label(root, text="If you see this, the GUI is working!", font=("Arial", 20))
label.pack(padx=20, pady=20)
print("TEST_STARTED")
root.mainloop()
