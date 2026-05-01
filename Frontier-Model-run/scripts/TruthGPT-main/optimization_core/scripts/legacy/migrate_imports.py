
import os

root_dir = r"c:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run-polyglot\scripts\TruthGPT-main\optimization_core"

replacements = {
    "from optimization_core.modules.optimizers": "from optimization_core.modules.optimizers",
    "from optimization_core.modules.models": "from optimization_core.modules.models",
    "from optimization_core.modules.learning": "from optimization_core.modules.learning",
}

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(subdir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = content
            modified = False
            for old, new in replacements.items():
                if old in new_content:
                    new_content = new_content.replace(old, new)
                    modified = True
            
            if modified:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated: {file_path}")

