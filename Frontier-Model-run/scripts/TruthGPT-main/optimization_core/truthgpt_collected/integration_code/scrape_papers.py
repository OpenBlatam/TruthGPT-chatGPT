#!/usr/bin/env python3
"""
Script para hacer scraping de papers de arXiv y extraer detalles exactos
para implementación precisa.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Optional
from pathlib import Path
import time

def scrape_arxiv_html(url: str) -> Dict[str, any]:
    """
    Scrape paper de arXiv HTML format.
    
    Args:
        url: URL del paper en formato HTML
        
    Returns:
        Dict con contenido extraído
    """
    try:
        print(f"📥 Scraping: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extraer título
        title_elem = soup.find('h1', class_='ltx_title')
        title = title_elem.get_text(strip=True) if title_elem else "Unknown"
        
        # Extraer autores
        authors = []
        author_elems = soup.find_all('span', class_='ltx_personname')
        for author in author_elems:
            authors.append(author.get_text(strip=True))
        
        # Extraer abstract
        abstract_elem = soup.find('div', class_='ltx_abstract')
        abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
        
        # Extraer secciones principales
        sections = {}
        section_elems = soup.find_all(['section', 'div'], class_=re.compile(r'ltx_section|ltx_subsection'))
        for section in section_elems:
            heading = section.find(['h2', 'h3', 'h4'])
            if heading:
                section_title = heading.get_text(strip=True)
                section_content = section.get_text(separator='\n', strip=True)
                sections[section_title] = section_content[:2000]  # Limitar tamaño
        
        # Extraer ecuaciones (LaTeX)
        equations = []
        math_elems = soup.find_all(['span', 'div'], class_=re.compile(r'ltx_equation|ltx_math'))
        for math in math_elems:
            eq_text = math.get_text(strip=True)
            if eq_text:
                equations.append(eq_text)
        
        # Extraer algoritmos
        algorithms = []
        algo_elems = soup.find_all(['div', 'pre'], class_=re.compile(r'algorithm|pseudocode'))
        for algo in algo_elems:
            algo_text = algo.get_text(separator='\n', strip=True)
            if algo_text:
                algorithms.append(algo_text)
        
        result = {
            'url': url,
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'sections': sections,
            'equations': equations[:10],  # Primeras 10 ecuaciones
            'algorithms': algorithms,
            'raw_html_length': len(response.content)
        }
        
        print(f"✅ Extraído: {len(sections)} secciones, {len(equations)} ecuaciones")
        return result
        
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return {'url': url, 'error': str(e)}


def scrape_arxiv_pdf(url: str) -> Dict[str, any]:
    """
    Scrape paper de arXiv PDF (extraer metadata).
    """
    try:
        # Para PDFs, extraer metadata de la página HTML
        html_url = url.replace('/pdf/', '/html/')
        return scrape_arxiv_html(html_url)
    except Exception as e:
        print(f"❌ Error scraping PDF {url}: {e}")
        return {'url': url, 'error': str(e)}


def scrape_github_repo(url: str) -> Dict[str, any]:
    """
    Scrape código de GitHub.
    """
    try:
        print(f"📥 Scraping GitHub: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/vnd.github.v3.raw'
        }
        
        # Convertir URL de GitHub a raw
        if 'github.com' in url and '/blob/' in url:
            raw_url = url.replace('/blob/', '/').replace('github.com', 'raw.githubusercontent.com')
        else:
            raw_url = url
        
        response = requests.get(raw_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Si es código Python
        if raw_url.endswith('.py'):
            code = response.text
            return {
                'url': url,
                'raw_url': raw_url,
                'code': code,
                'code_length': len(code),
                'type': 'python_code'
            }
        else:
            # HTML page
            soup = BeautifulSoup(response.content, 'html.parser')
            return {
                'url': url,
                'content': soup.get_text(separator='\n', strip=True)[:5000],
                'type': 'html_content'
            }
            
    except Exception as e:
        print(f"❌ Error scraping GitHub {url}: {e}")
        return {'url': url, 'error': str(e)}


# Papers a scrapear
PAPERS = {
    '2505.05315v2': 'https://arxiv.org/html/2505.05315v2',
    '2505.11140v1': 'https://arxiv.org/html/2505.11140v1',
    '2503.00735v3': 'https://arxiv.org/html/2503.00735v3',
    '2506.10987v1': 'https://arxiv.org/html/2506.10987v1',
    '2509.04439v1': 'https://arxiv.org/html/2509.04439v1',
    '2506.15841v2': 'https://arxiv.org/html/2506.15841v2',
    '2508.06471': 'https://arxiv.org/html/2508.06471',
    '2510.04871v1': 'https://arxiv.org/html/2510.04871v1',
    '2506.10848v2': 'https://arxiv.org/html/2506.10848v2',
    '2510.00071': 'https://arxiv.org/html/2510.00071',
    '2510.26788v1': 'https://arxiv.org/html/2510.26788v1',
}

GITHUB_REPOS = {
    'OLMoE': 'https://github.com/allenai/OLMoE',
    'MEM1': 'https://github.com/MIT-MI/MEM1/blob/main/Mem1/inference/amem/memory_system.py',
    'SAM2': 'https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/backbones/hieradet.py',
}

if __name__ == "__main__":
    output_dir = Path(__file__).parent / 'scraped_papers'
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Iniciando scraping de papers...\n")
    
    # Scrapear papers de arXiv
    papers_data = {}
    for paper_id, url in PAPERS.items():
        print(f"\n📄 Paper: {paper_id}")
        if '/pdf/' in url:
            data = scrape_arxiv_pdf(url)
        else:
            data = scrape_arxiv_html(url)
        
        papers_data[paper_id] = data
        
        # Guardar individual
        output_file = output_dir / f"{paper_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        time.sleep(2)  # Rate limiting
    
    # Scrapear repos de GitHub
    repos_data = {}
    for repo_name, url in GITHUB_REPOS.items():
        print(f"\n💻 Repo: {repo_name}")
        data = scrape_github_repo(url)
        repos_data[repo_name] = data
        
        # Guardar individual
        output_file = output_dir / f"{repo_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        time.sleep(2)  # Rate limiting
    
    # Guardar resumen
    summary = {
        'papers': {k: {'title': v.get('title', 'Unknown'), 'url': v.get('url', '')} 
                   for k, v in papers_data.items()},
        'repos': {k: {'url': v.get('url', ''), 'type': v.get('type', 'unknown')} 
                 for k, v in repos_data.items()}
    }
    
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Scraping completado!")
    print(f"📁 Archivos guardados en: {output_dir}")
    print(f"📊 Papers: {len(papers_data)}, Repos: {len(repos_data)}")




