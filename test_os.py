import os

path = './save/k/demo.json.npz'
dir = os.path.dirname(path)
print(dir)
os.makedirs(dir,exist_ok=True)