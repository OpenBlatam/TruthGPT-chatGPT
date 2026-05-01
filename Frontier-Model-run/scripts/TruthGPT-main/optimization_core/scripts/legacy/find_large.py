import os

files = []
for r, d, f in os.walk('.'):
    for file in f:
        if file.endswith('.py'):
            path = os.path.join(r, file)
            files.append({'path': path, 'size': os.path.getsize(path)})

files.sort(key=lambda x: x['size'], reverse=True)
for f in files[:20]:
    print(f"{f['size']} {f['path']}")

