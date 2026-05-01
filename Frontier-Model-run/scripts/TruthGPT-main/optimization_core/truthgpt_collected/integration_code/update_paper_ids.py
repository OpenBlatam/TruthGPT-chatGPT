#!/usr/bin/env python3
"""
Script para actualizar los arXiv IDs en los archivos de papers.
Usa este script después de encontrar los IDs reales de los papers.
"""

import re
from pathlib import Path
from typing import Dict

# Mapeo de papers a sus arXiv IDs
# Actualiza este diccionario cuando encuentres los IDs reales
PAPER_IDS = {
    'paper_longrope': None,  # Ejemplo: '2402.12345v1'
    'paper_longrope2': None,
    'paper_focusllm': None,
    'paper_cepe': None,
    'paper_lift': None,
    'paper_adagrope': None,
    'paper_longreward': None,
    'paper_longembed': None,
}

def update_paper_id(file_path: Path, paper_id: str) -> bool:
    """Actualiza el ID del paper en el archivo."""
    if not paper_id:
        return False
    
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Patrones para reemplazar
        patterns = [
            (r'\[ID_PENDIENTE\]', paper_id),
            (r'\[ID pendiente\]', paper_id),
            (r'https://arxiv\.org/abs/\[ID_PENDIENTE\]', f'https://arxiv.org/abs/{paper_id}'),
            (r'arXiv: \[ID pendiente\]', f'arXiv: {paper_id}'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # Si hubo cambios, escribir el archivo
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
        
        return False
    except Exception as e:
        print(f"❌ Error actualizando {file_path}: {e}")
        return False

def main():
    papers_dir = Path(__file__).parent / 'papers'
    
    print("🔧 Actualizando IDs de papers...\n")
    
    updated_count = 0
    skipped_count = 0
    
    for paper_name, paper_id in PAPER_IDS.items():
        # Buscar el archivo
        paper_file = None
        for py_file in papers_dir.rglob(f'{paper_name}.py'):
            paper_file = py_file
            break
        
        if not paper_file:
            print(f"⚠️  No se encontró: {paper_name}")
            continue
        
        if not paper_id:
            print(f"⏭️  Saltando {paper_name} (ID no proporcionado)")
            skipped_count += 1
            continue
        
        if update_paper_id(paper_file, paper_id):
            print(f"✅ Actualizado: {paper_name} → {paper_id}")
            updated_count += 1
        else:
            print(f"ℹ️  Sin cambios: {paper_name}")
    
    print(f"\n📊 Resumen:")
    print(f"   ✅ Actualizados: {updated_count}")
    print(f"   ⏭️  Saltados: {skipped_count}")
    print(f"   📝 Total: {len(PAPER_IDS)}")
    
    if skipped_count > 0:
        print(f"\n💡 Para actualizar los papers saltados, edita este script")
        print(f"   y agrega los IDs en el diccionario PAPER_IDS.")

if __name__ == "__main__":
    main()


