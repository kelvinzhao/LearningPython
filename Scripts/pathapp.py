from pathlib import Path

path = Path()
files = path.glob("*")
for filename in files:
    print(filename)
