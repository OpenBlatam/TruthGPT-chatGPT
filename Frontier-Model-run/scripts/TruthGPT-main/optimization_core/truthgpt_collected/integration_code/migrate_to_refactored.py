#!/usr/bin/env python3
"""
Script de Migración - Sistema Refactorizado
===========================================

Ayuda a migrar código existente al sistema refactorizado.
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationHelper:
    """Helper para migración de código."""
    
    # Mapeo de imports antiguos a nuevos
    IMPORT_MAPPINGS = {
        'from paper_registry import': 'from papers.core.paper_registry_refactored import',
        'from paper_loader import': 'from papers.paper_loader_refactored import',
        'from paper_extractor import': 'from papers.paper_extractor_refactored import',
        'import paper_registry': 'from papers.core import paper_registry_refactored',
        'import paper_loader': 'from papers import paper_loader_refactored',
        'import paper_extractor': 'from papers import paper_extractor_refactored',
    }
    
    @classmethod
    def migrate_file(cls, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Migra un archivo al sistema refactorizado.
        
        Returns:
            (changed, list_of_changes)
        """
        if not file_path.exists():
            return False, []
        
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        changes = []
        
        # Migrar imports
        for old_import, new_import in cls.IMPORT_MAPPINGS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes.append(f"Updated import: {old_import} → {new_import}")
        
        # Migrar get_registry() calls
        if 'get_registry(' in content and 'paper_registry_refactored' not in content:
            # Ya está migrado o necesita migración manual
            pass
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True, changes
        
        return False, []
    
    @classmethod
    def migrate_directory(cls, directory: Path, pattern: str = "*.py") -> Dict[str, List[str]]:
        """Migra todos los archivos en un directorio."""
        results = {}
        
        for file_path in directory.rglob(pattern):
            if 'core' in file_path.parts or 'refactored' in file_path.name:
                continue  # Skip archivos refactorizados
            
            changed, changes = cls.migrate_file(file_path)
            if changed:
                results[str(file_path)] = changes
                logger.info(f"✅ Migrated: {file_path.name}")
        
        return results
    
    @classmethod
    def check_migration_status(cls, directory: Path) -> Dict[str, Any]:
        """Verifica estado de migración."""
        status = {
            'total_files': 0,
            'migrated': 0,
            'needs_migration': [],
            'already_migrated': []
        }
        
        for file_path in directory.rglob("*.py"):
            if 'core' in file_path.parts or 'refactored' in file_path.name:
                continue
            
            status['total_files'] += 1
            content = file_path.read_text(encoding='utf-8')
            
            # Verificar si usa imports antiguos
            uses_old = any(old in content for old in cls.IMPORT_MAPPINGS.keys())
            uses_new = any('refactored' in content or 'core' in content)
            
            if uses_new and not uses_old:
                status['already_migrated'].append(str(file_path))
                status['migrated'] += 1
            elif uses_old:
                status['needs_migration'].append(str(file_path))
        
        return status


def main():
    """Función principal."""
    print("="*80)
    print("🔄 MIGRATION HELPER - Sistema Refactorizado")
    print("="*80)
    
    integration_dir = Path(__file__).parent
    
    # Verificar estado
    print("\n📊 Checking migration status...")
    helper = MigrationHelper()
    status = helper.check_migration_status(integration_dir)
    
    print(f"\n📈 Status:")
    print(f"  Total files: {status['total_files']}")
    print(f"  Already migrated: {status['migrated']}")
    print(f"  Needs migration: {len(status['needs_migration'])}")
    
    if status['needs_migration']:
        print(f"\n⚠️  Files needing migration:")
        for file_path in status['needs_migration'][:10]:
            print(f"    - {file_path}")
        
        # Preguntar si migrar
        print(f"\n❓ Migrate {len(status['needs_migration'])} files? (y/n)")
        # En producción, usar input()
        # response = input()
        # if response.lower() == 'y':
        #     results = helper.migrate_directory(integration_dir)
        #     print(f"\n✅ Migrated {len(results)} files")
    else:
        print("\n✅ All files already migrated!")


if __name__ == "__main__":
    main()


