"""
Deployment script for TruthGPT Hugging Face Space
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

class HuggingFaceSpaceDeployer:
    """Deploy TruthGPT models to Hugging Face Spaces."""
    
    def __init__(self, space_name: str = "truthgpt-models-demo", username: str = "OpenBlatam-Origen"):
        self.space_name = space_name
        self.username = username
        self.space_url = f"https://huggingface.co/spaces/{username}/{space_name}"
        self.repo_url = f"https://huggingface.co/spaces/{username}/{space_name}"
    
    def check_hf_cli(self) -> bool:
        """Check if Hugging Face CLI is installed."""
        try:
            result = subprocess.run(['huggingface-cli', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_hf_cli(self):
        """Install Hugging Face CLI if not present."""
        print("📦 Installing Hugging Face CLI...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub[cli]'], 
                      check=True)
    
    def login_to_hf(self, token: Optional[str] = None):
        """Login to Hugging Face."""
        if token:
            print("🔐 Logging in to Hugging Face...")
            subprocess.run(['huggingface-cli', 'login', '--token', token], check=True)
        else:
            print("🔐 Please login to Hugging Face manually:")
            subprocess.run(['huggingface-cli', 'login'], check=True)
    
    def create_space(self):
        """Create a new Hugging Face Space."""
        print(f"🚀 Creating Hugging Face Space: {self.space_name}")
        
        cmd = [
            'huggingface-cli', 'repo', 'create',
            f'{self.username}/{self.space_name}',
            '--type', 'space',
            '--space_sdk', 'gradio'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Space created successfully: {self.space_url}")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e):
                print(f"ℹ️  Space already exists: {self.space_url}")
            else:
                raise e
    
    def clone_space(self, target_dir: str = "./hf_space_repo"):
        """Clone the Hugging Face Space repository."""
        print(f"📥 Cloning space repository...")
        
        if Path(target_dir).exists():
            print(f"🗂️  Directory {target_dir} already exists, removing...")
            subprocess.run(['rm', '-rf', target_dir], check=True)
        
        subprocess.run(['git', 'clone', self.repo_url, target_dir], check=True)
        return target_dir
    
    def copy_files_to_space(self, space_repo_dir: str):
        """Copy Space files to the cloned repository."""
        print("📋 Copying files to space repository...")
        
        space_files = [
            'app.py',
            'requirements.txt',
            'README.md',
            '.gitignore',
            'model_export.py'
        ]
        
        for file in space_files:
            src = Path(file)
            dst = Path(space_repo_dir) / file
            
            if src.exists():
                subprocess.run(['cp', str(src), str(dst)], check=True)
                print(f"✅ Copied {file}")
            else:
                print(f"⚠️  File {file} not found, skipping...")
    
    def commit_and_push(self, space_repo_dir: str):
        """Commit and push changes to the Space repository."""
        print("📤 Committing and pushing changes...")
        
        os.chdir(space_repo_dir)
        
        subprocess.run(['git', 'config', 'user.email', 'devin-ai-integration[bot]@users.noreply.github.com'])
        subprocess.run(['git', 'config', 'user.name', 'Devin AI'])
        
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', '🚀 Deploy TruthGPT Models Interactive Demo'], check=True)
        subprocess.run(['git', 'push'], check=True)
        
        print(f"✅ Successfully deployed to: {self.space_url}")
    
    def deploy(self, hf_token: Optional[str] = None):
        """Complete deployment process."""
        print("🚀 Starting TruthGPT Hugging Face Space Deployment")
        print("=" * 60)
        
        if not self.check_hf_cli():
            self.install_hf_cli()
        
        self.login_to_hf(hf_token)
        
        self.create_space()
        
        space_repo_dir = self.clone_space()
        self.copy_files_to_space(space_repo_dir)
        self.commit_and_push(space_repo_dir)
        
        print("\n" + "=" * 60)
        print("🎉 Deployment Complete!")
        print(f"🌐 Space URL: {self.space_url}")
        print(f"📱 Interactive Demo: {self.space_url}")
        print("🔧 You can now test the models through the web interface")

def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy TruthGPT to Hugging Face Spaces')
    parser.add_argument('--token', type=str, help='Hugging Face token')
    parser.add_argument('--space-name', type=str, default='truthgpt-models-demo', 
                       help='Name of the Hugging Face Space')
    parser.add_argument('--username', type=str, default='OpenBlatam-Origen',
                       help='Hugging Face username/organization')
    
    args = parser.parse_args()
    
    deployer = HuggingFaceSpaceDeployer(args.space_name, args.username)
    deployer.deploy(args.token)

if __name__ == "__main__":
    main()
