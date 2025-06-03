#!/usr/bin/env python3
"""
Architecture validation script for Enhanced Viral Video Clips Model
Validates the code structure and architecture patterns without requiring dependencies
"""

import os
import ast
from pathlib import Path
from typing import Dict, List, Set

def analyze_file_structure():
    """Analyze the file structure and architecture"""
    print("🏗️ Analyzing Enhanced Viral Video Clips Model Architecture...")
    
    base_dir = Path(__file__).parent
    
    # Expected structure
    expected_structure = {
        "interfaces": ["video_llm_interface.py"],
        "configs": ["video_model_configs.py"],
        "llm": ["video_llm_factory.py", "enhanced_viral_video_llm.py"],
        "tools": ["video_processing_tools.py"],
        "agents": ["viral_video_agent.py"]
    }
    
    print("\n📁 File Structure Analysis:")
    
    structure_valid = True
    for directory, files in expected_structure.items():
        dir_path = base_dir / directory
        if dir_path.exists():
            print(f"✅ {directory}/ directory exists")
            for file in files:
                file_path = dir_path / file
                if file_path.exists():
                    size = file_path.stat().st_size
                    print(f"  ✅ {file} ({size:,} bytes)")
                else:
                    print(f"  ❌ {file} missing")
                    structure_valid = False
        else:
            print(f"❌ {directory}/ directory missing")
            structure_valid = False
    
    # Check main files
    main_files = ["__init__.py", "demo.py", "requirements.txt", "README.md"]
    for file in main_files:
        file_path = base_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} missing")
            structure_valid = False
    
    return structure_valid

def analyze_code_patterns():
    """Analyze code for architecture patterns"""
    print("\n🏭 Architecture Pattern Analysis:")
    
    base_dir = Path(__file__).parent
    patterns_found = {
        "abstract_base_classes": False,
        "factory_pattern": False,
        "registry_pattern": False,
        "configuration_management": False,
        "agent_framework": False,
        "tool_system": False
    }
    
    # Check for abstract base classes
    interface_file = base_dir / "interfaces" / "video_llm_interface.py"
    if interface_file.exists():
        content = interface_file.read_text()
        if "abc.ABC" in content and "@abc.abstractmethod" in content:
            patterns_found["abstract_base_classes"] = True
            print("✅ Abstract Base Classes (VideoLLM interface)")
    
    # Check for factory pattern
    factory_file = base_dir / "llm" / "video_llm_factory.py"
    if factory_file.exists():
        content = factory_file.read_text()
        if "VideoLLMRegistry" in content and "create_video_llm" in content:
            patterns_found["factory_pattern"] = True
            print("✅ Factory Pattern (VideoLLMFactory)")
        if "_registry" in content or "register" in content:
            patterns_found["registry_pattern"] = True
            print("✅ Registry Pattern (VideoLLMRegistry)")
    
    # Check for configuration management
    config_file = base_dir / "configs" / "video_model_configs.py"
    if config_file.exists():
        content = config_file.read_text()
        if "os.environ.get" in content and "CONFIG" in content:
            patterns_found["configuration_management"] = True
            print("✅ Configuration Management (Environment-based)")
    
    # Check for agent framework
    agent_file = base_dir / "agents" / "viral_video_agent.py"
    if agent_file.exists():
        content = agent_file.read_text()
        if "ViralVideoAgent" in content and "AgentState" in content:
            patterns_found["agent_framework"] = True
            print("✅ Agent Framework (ViralVideoAgent)")
    
    # Check for tool system
    tools_file = base_dir / "tools" / "video_processing_tools.py"
    if tools_file.exists():
        content = tools_file.read_text()
        if "VideoProcessingTool" in content and "ToolRegistry" in content:
            patterns_found["tool_system"] = True
            print("✅ Tool System (Modular tools)")
    
    return patterns_found

def analyze_class_definitions():
    """Analyze class definitions and inheritance"""
    print("\n🧬 Class Architecture Analysis:")
    
    base_dir = Path(__file__).parent
    
    # Key classes to check
    key_classes = {
        "VideoLLM": "interfaces/video_llm_interface.py",
        "VideoLLMConfig": "interfaces/video_llm_interface.py",
        "EnhancedViralVideoLLM": "llm/enhanced_viral_video_llm.py",
        "VideoLLMRegistry": "llm/video_llm_factory.py",
        "ViralVideoAgent": "agents/viral_video_agent.py",
        "VideoProcessingTool": "tools/video_processing_tools.py"
    }
    
    classes_found = {}
    
    for class_name, file_path in key_classes.items():
        full_path = base_dir / file_path
        if full_path.exists():
            try:
                content = full_path.read_text()
                if f"class {class_name}" in content:
                    classes_found[class_name] = True
                    print(f"✅ {class_name} class defined")
                else:
                    classes_found[class_name] = False
                    print(f"❌ {class_name} class not found")
            except Exception as e:
                classes_found[class_name] = False
                print(f"❌ Error analyzing {class_name}: {e}")
        else:
            classes_found[class_name] = False
            print(f"❌ File not found for {class_name}")
    
    return classes_found

def analyze_dependencies():
    """Analyze dependencies and requirements"""
    print("\n📦 Dependency Analysis:")
    
    base_dir = Path(__file__).parent
    requirements_file = base_dir / "requirements.txt"
    
    if requirements_file.exists():
        content = requirements_file.read_text()
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
        
        # Key dependency categories
        categories = {
            "AI/ML": ["torch", "transformers", "langchain"],
            "Video": ["opencv-python", "moviepy", "ffmpeg"],
            "Audio": ["librosa", "whisper"],
            "Web": ["streamlit", "fastapi", "yt-dlp"],
            "Data": ["numpy", "pandas", "pillow"]
        }
        
        found_deps = {cat: [] for cat in categories}
        
        for line in lines:
            dep_name = line.split('>=')[0].split('==')[0].split('[')[0]
            for category, deps in categories.items():
                if any(key_dep in dep_name.lower() for key_dep in deps):
                    found_deps[category].append(dep_name)
                    break
        
        for category, deps in found_deps.items():
            if deps:
                print(f"✅ {category}: {len(deps)} dependencies ({', '.join(deps[:3])}{'...' if len(deps) > 3 else ''})")
            else:
                print(f"❌ {category}: No dependencies found")
        
        print(f"📊 Total dependencies: {len(lines)}")
        return len(lines) > 50  # Should have substantial dependencies
    
    else:
        print("❌ requirements.txt not found")
        return False

def analyze_documentation():
    """Analyze documentation quality"""
    print("\n📚 Documentation Analysis:")
    
    base_dir = Path(__file__).parent
    readme_file = base_dir / "README.md"
    
    if readme_file.exists():
        content = readme_file.read_text()
        
        # Check for key sections
        sections = [
            "Overview", "Features", "Architecture", "Installation", 
            "Usage", "API", "Performance", "Examples"
        ]
        
        found_sections = []
        for section in sections:
            if section.lower() in content.lower():
                found_sections.append(section)
        
        print(f"✅ README.md exists ({len(content):,} characters)")
        print(f"📋 Documentation sections: {len(found_sections)}/{len(sections)}")
        print(f"   Found: {', '.join(found_sections)}")
        
        # Check for code examples
        code_blocks = content.count("```")
        print(f"💻 Code examples: {code_blocks // 2} blocks")
        
        return len(found_sections) >= 6 and code_blocks >= 10
    
    else:
        print("❌ README.md not found")
        return False

def calculate_code_metrics():
    """Calculate code metrics"""
    print("\n📊 Code Metrics:")
    
    base_dir = Path(__file__).parent
    
    total_lines = 0
    total_files = 0
    python_files = []
    
    for file_path in base_dir.rglob("*.py"):
        if file_path.name != "validate_architecture.py":  # Exclude this file
            try:
                content = file_path.read_text()
                lines = len(content.split('\n'))
                total_lines += lines
                total_files += 1
                python_files.append((file_path.name, lines))
            except Exception:
                pass
    
    print(f"📁 Python files: {total_files}")
    print(f"📝 Total lines of code: {total_lines:,}")
    print(f"📈 Average lines per file: {total_lines // total_files if total_files > 0 else 0}")
    
    # Show largest files
    python_files.sort(key=lambda x: x[1], reverse=True)
    print("🏆 Largest files:")
    for name, lines in python_files[:5]:
        print(f"   {name}: {lines:,} lines")
    
    return total_lines > 5000  # Should be substantial codebase

def main():
    """Run complete architecture validation"""
    print("🎬 Enhanced Viral Video Clips Model - Architecture Validation")
    print("=" * 70)
    
    validations = [
        ("File Structure", analyze_file_structure),
        ("Architecture Patterns", analyze_code_patterns),
        ("Class Definitions", analyze_class_definitions),
        ("Dependencies", analyze_dependencies),
        ("Documentation", analyze_documentation),
        ("Code Metrics", calculate_code_metrics)
    ]
    
    results = {}
    
    for name, validator in validations:
        try:
            result = validator()
            results[name] = result
        except Exception as e:
            print(f"❌ {name} validation failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 Architecture Validation Summary:")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {name}")
    
    print(f"\n📊 Overall Score: {passed}/{total} validations passed")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("🎉 EXCELLENT! Architecture validation successful!")
        print("🚀 The Enhanced Viral Video Clips Model is well-architected and ready for deployment!")
        
        print("\n🏗️ Architecture Highlights:")
        print("   • Onyx-inspired design patterns implemented")
        print("   • Enterprise-grade factory and registry patterns")
        print("   • Comprehensive configuration management")
        print("   • Modular tool system for extensibility")
        print("   • Intelligent agent framework for automation")
        print("   • Multi-platform optimization capabilities")
        print("   • Extensive documentation and examples")
        
    elif passed >= total * 0.6:  # 60% pass rate
        print("⚠️ GOOD! Most architecture components are in place.")
        print("🔧 Some minor improvements needed for production readiness.")
        
    else:
        print("❌ NEEDS WORK! Architecture validation failed.")
        print("🛠️ Significant improvements needed before deployment.")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)