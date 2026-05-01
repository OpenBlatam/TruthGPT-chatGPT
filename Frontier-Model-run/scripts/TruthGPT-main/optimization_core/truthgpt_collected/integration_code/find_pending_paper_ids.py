#!/usr/bin/env python3
"""
Script para encontrar los arXiv IDs pendientes de los papers.
Busca en los archivos de código y compila una lista de papers que necesitan IDs.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple

def find_pending_ids(papers_dir: Path) -> List[Dict[str, str]]:
    """Encuentra todos los papers con IDs pendientes."""
    pending = []
    
    # Patrones para buscar IDs pendientes
    patterns = [
        r'\[ID_PENDIENTE\]',
        r'\[ID pendiente\]',
        r'ID_PENDIENTE',
        r'ID pendiente',
    ]
    
    # Buscar en todos los archivos .py en papers/
    for py_file in papers_dir.rglob('*.py'):
        if 'paper_' in py_file.name:
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Buscar si tiene ID pendiente
                has_pending = any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)
                
                if has_pending:
                    # Extraer información del paper
                    title_match = re.search(r'^""".*?\n([^\n]+)', content, re.MULTILINE)
                    title = title_match.group(1).strip() if title_match else py_file.stem
                    
                    # Buscar autores si están disponibles
                    authors_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+et\s+al\.|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', content)
                    authors = authors_match.group(0) if authors_match else "Unknown"
                    
                    # Buscar año
                    year_match = re.search(r'\((\d{4})\)', content)
                    year = year_match.group(1) if year_match else "Unknown"
                    
                    pending.append({
                        'file': str(py_file.relative_to(papers_dir.parent)),
                        'title': title,
                        'authors': authors,
                        'year': year,
                        'paper_name': py_file.stem
                    })
            except Exception as e:
                print(f"Error reading {py_file}: {e}")
    
    return pending

def main():
    papers_dir = Path(__file__).parent / 'papers'
    
    print("🔍 Buscando papers con IDs pendientes...\n")
    
    pending = find_pending_ids(papers_dir)
    
    if not pending:
        print("✅ No se encontraron papers con IDs pendientes!")
        return
    
    print(f"📋 Encontrados {len(pending)} papers con IDs pendientes:\n")
    
    for i, paper in enumerate(pending, 1):
        print(f"{i}. {paper['paper_name']}")
        print(f"   📄 Título: {paper['title']}")
        print(f"   👤 Autores: {paper['authors']}")
        print(f"   📅 Año: {paper['year']}")
        print(f"   📁 Archivo: {paper['file']}")
        print()
    
    # Generar lista para búsqueda
    print("\n" + "="*60)
    print("📝 Lista para búsqueda en arXiv:")
    print("="*60 + "\n")
    
    for paper in pending:
        print(f"- {paper['title']} ({paper['authors']}, {paper['year']})")
        print(f"  Archivo: {paper['file']}\n")
    
    # Guardar en archivo
    output_file = Path(__file__).parent / 'pending_paper_ids.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Papers con IDs Pendientes\n")
        f.write("="*60 + "\n\n")
        for paper in pending:
            f.write(f"Paper: {paper['paper_name']}\n")
            f.write(f"Título: {paper['title']}\n")
            f.write(f"Autores: {paper['authors']}\n")
            f.write(f"Año: {paper['year']}\n")
            f.write(f"Archivo: {paper['file']}\n")
            f.write("-"*60 + "\n\n")
    
    print(f"\n✅ Lista guardada en: {output_file}")

if __name__ == "__main__":
    main()


