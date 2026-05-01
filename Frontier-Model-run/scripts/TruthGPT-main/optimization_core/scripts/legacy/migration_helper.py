"""
Migration Helper for TruthGPT Optimization Core
Helps migrate from original to enhanced architecture
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MigrationHelper:
    """Helper class for migrating optimization code."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.constants_file = self.base_path / "constants.py"
        self.config_file = self.base_path / "config" / "architecture.py"
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the codebase for migration opportunities."""
        analysis = {
            "hardcoded_values": [],
            "duplicated_code": [],
            "missing_constants": [],
            "architecture_issues": [],
            "optimization_opportunities": []
        }
        
        # Find all Python files
        python_files = list(self.base_path.rglob("*.py"))
        
        for file_path in python_files:
            if "migration_helper" in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze for hardcoded values
                hardcoded = self._find_hardcoded_values(content, str(file_path))
                analysis["hardcoded_values"].extend(hardcoded)
                
                # Analyze for duplicated code
                duplicated = self._find_duplicated_code(content, str(file_path))
                analysis["duplicated_code"].extend(duplicated)
                
                # Analyze for missing constants
                missing = self._find_missing_constants(content, str(file_path))
                analysis["missing_constants"].extend(missing)
                
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
        
        return analysis
    
    def _find_hardcoded_values(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Find hardcoded values that could be constants."""
        hardcoded_values = []
        
        # Common hardcoded patterns
        patterns = [
            (r'\b\d+\.\d+\b', 'float'),
            (r'\b\d+\b', 'int'),
            (r'"[^"]*"', 'string'),
            (r"'[^']*'", 'string')
        ]
        
        for pattern, value_type in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                value = match.group()
                line_num = content[:match.start()].count('\n') + 1
                
                # Skip common values that are likely not constants
                if value in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    continue
                if value in ['""', "''", '0.0', '1.0']:
                    continue
                
                hardcoded_values.append({
                    'file': file_path,
                    'line': line_num,
                    'value': value,
                    'type': value_type,
                    'context': self._get_context(content, match.start())
                })
        
        return hardcoded_values
    
    def _find_duplicated_code(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Find duplicated code patterns."""
        duplicated_code = []
        
        # Common duplicated patterns
        patterns = [
            r'class.*Optimizer.*:',
            r'def.*optimize.*:',
            r'speed_improvement.*=.*\d+',
            r'memory_reduction.*=.*\d+',
            r'techniques_applied.*=.*\[\]'
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, content))
            if len(matches) > 1:
                duplicated_code.append({
                    'file': file_path,
                    'pattern': pattern,
                    'count': len(matches),
                    'matches': [match.group() for match in matches]
                })
        
        return duplicated_code
    
    def _find_missing_constants(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Find places where constants should be used."""
        missing_constants = []
        
        # Look for repeated values
        value_counts = {}
        for match in re.finditer(r'\b\d+\.\d+\b|\b\d+\b', content):
            value = match.group()
            if value not in value_counts:
                value_counts[value] = []
            value_counts[value].append(match.start())
        
        for value, positions in value_counts.items():
            if len(positions) > 2:  # Value appears more than twice
                missing_constants.append({
                    'file': file_path,
                    'value': value,
                    'count': len(positions),
                    'positions': positions
                })
        
        return missing_constants
    
    def _get_context(self, content: str, position: int, context_lines: int = 3) -> str:
        """Get context around a position."""
        lines = content.split('\n')
        line_num = content[:position].count('\n')
        
        start = max(0, line_num - context_lines)
        end = min(len(lines), line_num + context_lines + 1)
        
        return '\n'.join(lines[start:end])
    
    def generate_migration_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a migration plan based on analysis."""
        plan = {
            "phases": [],
            "estimated_time": "2-3 days",
            "risk_level": "low",
            "recommendations": []
        }
        
        # Phase 1: Extract constants
        if analysis["hardcoded_values"]:
            plan["phases"].append({
                "name": "Extract Constants",
                "description": "Extract hardcoded values to constants.py",
                "files_affected": len(set(item["file"] for item in analysis["hardcoded_values"])),
                "priority": "high"
            })
        
        # Phase 2: Refactor duplicated code
        if analysis["duplicated_code"]:
            plan["phases"].append({
                "name": "Refactor Duplicated Code",
                "description": "Create base classes and common utilities",
                "files_affected": len(set(item["file"] for item in analysis["duplicated_code"])),
                "priority": "medium"
            })
        
        # Phase 3: Implement enhanced architecture
        plan["phases"].append({
            "name": "Implement Enhanced Architecture",
            "description": "Implement enhanced optimizer with advanced features",
            "files_affected": 1,
            "priority": "high"
        })
        
        # Generate recommendations
        if analysis["hardcoded_values"]:
            plan["recommendations"].append("Start with extracting the most common hardcoded values")
        
        if analysis["duplicated_code"]:
            plan["recommendations"].append("Create base optimizer class to reduce duplication")
        
        plan["recommendations"].append("Implement enhanced optimizer as new file first")
        plan["recommendations"].append("Add comprehensive tests for new functionality")
        
        return plan
    
    def create_constants_mapping(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Create mapping of hardcoded values to constant names."""
        mapping = {}
        
        # Group hardcoded values by type and value
        value_groups = {}
        for item in analysis["hardcoded_values"]:
            value = item["value"]
            value_type = item["type"]
            key = f"{value_type}_{value}"
            
            if key not in value_groups:
                value_groups[key] = []
            value_groups[key].append(item)
        
        # Create constant names
        for key, items in value_groups.items():
            if len(items) > 1:  # Only for values that appear multiple times
                value_type, value = key.split("_", 1)
                
                # Generate constant name
                if value_type == "float":
                    constant_name = f"FACTOR_{value.replace('.', '_')}"
                elif value_type == "int":
                    constant_name = f"LEVEL_{value}"
                else:
                    constant_name = f"CONFIG_{value.replace('"', '').replace("'", '')}"
                
                mapping[value] = constant_name
        
        return mapping
    
    def generate_migration_script(self, analysis: Dict[str, Any]) -> str:
        """Generate a migration script."""
        script = """#!/usr/bin/env python3
\"\"\"
Auto-generated migration script for TruthGPT Optimization Core
Generated by MigrationHelper
\"\"\"

import os
import re
from pathlib import Path

def migrate_file(file_path: str, constants_mapping: dict):
    \"\"\"Migrate a single file to use constants.\"\"\"
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply replacements
    for old_value, constant_name in constants_mapping.items():
        # Replace hardcoded values with constants
        pattern = re.escape(old_value)
        replacement = constant_name
        content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Migrated {file_path}")

def main():
    \"\"\"Main migration function.\"\"\"
    # Constants mapping
    constants_mapping = {
"""
        
        # Add constants mapping
        mapping = self.create_constants_mapping(analysis)
        for value, constant_name in mapping.items():
            script += f'        "{value}": "{constant_name}",\n'
        
        script += """    }
    
    # Files to migrate
    files_to_migrate = [
"""
        
        # Add files to migrate
        files = set(item["file"] for item in analysis["hardcoded_values"])
        for file_path in files:
            script += f'        "{file_path}",\n'
        
        script += """    ]
    
    # Migrate files
    for file_path in files_to_migrate:
        if os.path.exists(file_path):
            migrate_file(file_path, constants_mapping)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
"""
        
        return script
    
    def validate_migration(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that migration is safe."""
        validation = {
            "safe": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check for potential issues
        if len(analysis["hardcoded_values"]) > 100:
            validation["warnings"].append("Large number of hardcoded values - consider phased migration")
        
        if len(analysis["duplicated_code"]) > 50:
            validation["warnings"].append("Significant code duplication - refactor carefully")
        
        # Check for critical files
        critical_files = ["__init__.py", "main.py", "core.py"]
        for file_info in analysis["hardcoded_values"]:
            if any(critical in file_info["file"] for critical in critical_files):
                validation["warnings"].append(f"Critical file {file_info['file']} has hardcoded values")
        
        # Generate recommendations
        if validation["warnings"]:
            validation["recommendations"].append("Start with non-critical files")
            validation["recommendations"].append("Create comprehensive tests before migration")
            validation["recommendations"].append("Use version control for safety")
        
        return validation

def main():
    """Main function for migration helper."""
    # Initialize migration helper
    base_path = Path(__file__).parent
    helper = MigrationHelper(str(base_path))
    
    print("🔍 Analyzing TruthGPT Optimization Core...")
    
    # Analyze codebase
    analysis = helper.analyze_codebase()
    
    print(f"📊 Analysis Results:")
    print(f"  - Hardcoded values: {len(analysis['hardcoded_values'])}")
    print(f"  - Duplicated code: {len(analysis['duplicated_code'])}")
    print(f"  - Missing constants: {len(analysis['missing_constants'])}")
    
    # Generate migration plan
    plan = helper.generate_migration_plan(analysis)
    
    print(f"\n📋 Migration Plan:")
    for i, phase in enumerate(plan["phases"], 1):
        print(f"  {i}. {phase['name']} - {phase['description']}")
        print(f"     Priority: {phase['priority']}, Files: {phase['files_affected']}")
    
    # Create constants mapping
    mapping = helper.create_constants_mapping(analysis)
    
    print(f"\n🔧 Constants Mapping:")
    for value, constant_name in list(mapping.items())[:10]:  # Show first 10
        print(f"  {value} -> {constant_name}")
    if len(mapping) > 10:
        print(f"  ... and {len(mapping) - 10} more")
    
    # Validate migration
    validation = helper.validate_migration(analysis)
    
    print(f"\n✅ Migration Validation:")
    print(f"  Safe: {validation['safe']}")
    if validation["warnings"]:
        print(f"  Warnings: {len(validation['warnings'])}")
        for warning in validation["warnings"][:3]:
            print(f"    - {warning}")
    if validation["recommendations"]:
        print(f"  Recommendations:")
        for rec in validation["recommendations"]:
            print(f"    - {rec}")
    
    # Generate migration script
    script = helper.generate_migration_script(analysis)
    
    # Save migration script
    script_path = base_path / "migration_script.py"
    with open(script_path, 'w') as f:
        f.write(script)
    
    print(f"\n📝 Migration script saved to: {script_path}")
    
    # Save analysis results
    results_path = base_path / "migration_analysis.json"
    with open(results_path, 'w') as f:
        json.dump({
            "analysis": analysis,
            "plan": plan,
            "mapping": mapping,
            "validation": validation
        }, f, indent=2)
    
    print(f"📊 Analysis results saved to: {results_path}")
    
    print(f"\n🚀 Ready for migration! Run: python {script_path}")

if __name__ == "__main__":
    main()











