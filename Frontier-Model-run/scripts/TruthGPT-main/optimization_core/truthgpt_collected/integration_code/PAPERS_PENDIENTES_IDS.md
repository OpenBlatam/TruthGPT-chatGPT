# Papers con IDs Pendientes

Este documento lista todos los papers que necesitan sus arXiv IDs actualizados.

## 📋 Lista de Papers Pendientes (8 papers)

### 1. **LongRoPE** - Extending LLM Context Window Beyond 2 Million Tokens
- **Archivo**: `papers/research/paper_longrope.py`
- **Autores**: Ding, Zhang, Xu, Shang, et al.
- **Año**: 2024
- **Búsqueda sugerida**: 
  - "LongRoPE" "Extending LLM Context Window" "2 Million Tokens" Ding Zhang
  - Buscar en: https://arxiv.org/search/?query=LongRoPE+Ding+Zhang

### 2. **LongRoPE2** - Near-Lossless LLM Context Window Scaling
- **Archivo**: `papers/research/paper_longrope2.py`
- **Autores**: Shang, Zhang, Wang, et al.
- **Año**: 2025
- **Búsqueda sugerida**:
  - "LongRoPE2" "Near-Lossless" "Context Window Scaling" Shang Zhang
  - Buscar en: https://arxiv.org/search/?query=LongRoPE2+Shang+Zhang

### 3. **FocusLLM** - Scaling LLM's Context by Parallel Decoding
- **Archivo**: `papers/research/paper_focusllm.py`
- **Autores**: Lee, Thu, etc.
- **Año**: 2024
- **Búsqueda sugerida**:
  - "FocusLLM" "Parallel Decoding" "Scaling LLM Context" Lee
  - Buscar en: https://arxiv.org/search/?query=FocusLLM+parallel+decoding

### 4. **CEPE** - Context Expansion with Parallel Encoding
- **Archivo**: `papers/research/paper_cepe.py`
- **Autores**: Yen, Gao, Chen
- **Año**: 2024
- **Búsqueda sugerida**:
  - "CEPE" "Context Expansion" "Parallel Encoding" Yen Gao Chen
  - Buscar en: https://arxiv.org/search/?query=CEPE+context+expansion+Yen

### 5. **LIFT** - Long Input Fine-Tuning
- **Archivo**: `papers/research/paper_lift.py`
- **Autores**: Mao, Xu, Li, et al.
- **Año**: 2025
- **Búsqueda sugerida**:
  - "LIFT" "Long Input Fine-Tuning" Mao Xu Li
  - Buscar en: https://arxiv.org/search/?query=LIFT+long+input+fine-tuning+Mao

### 6. **AdaGroPE** - Adaptive Grouped Positional Encoding
- **Archivo**: `papers/research/paper_adagrope.py`
- **Autores**: Xu, Li, Chen, Lin, Han, Ding
- **Año**: 2025 (ACL 2025)
- **Búsqueda sugerida**:
  - "AdaGroPE" "Adaptive Grouped Positional Encoding" Xu Li Chen ACL
  - Buscar en: https://arxiv.org/search/?query=AdaGroPE+adaptive+grouped+positional

### 7. **LongReward** - Improving Long-context Large Language Models
- **Archivo**: `papers/research/paper_longreward.py`
- **Autores**: Hossein, Bai, etc.
- **Año**: 2025 (ACL 2025)
- **Búsqueda sugerida**:
  - "LongReward" "Improving Long-context" "Large Language Models" Hossein Bai
  - Buscar en: https://arxiv.org/search/?query=LongReward+long-context+LLM

### 8. **LongEmbed** - Extending Embedding Models for Long Context Retrieval
- **Archivo**: `papers/research/paper_longembed.py`
- **Autores**: Zhu, Wang, Yang, Song, Wu, Wei, Li
- **Año**: 2024
- **Búsqueda sugerida**:
  - "LongEmbed" "Extending Embedding Models" "Long Context Retrieval" Zhu Wang
  - Buscar en: https://arxiv.org/search/?query=LongEmbed+embedding+models+retrieval

## 🔍 Cómo Buscar los IDs

1. **Usar arXiv Search**: Visita https://arxiv.org/search/ y busca con los términos sugeridos
2. **Buscar por título exacto**: Copia el título completo en la búsqueda
3. **Buscar por autores**: Usa el formato "apellido nombre" en la búsqueda
4. **Revisar conferencias**: Para papers de ACL 2025, revisar el sitio de ACL Anthology

## 📝 Formato del ID de arXiv

Los IDs de arXiv tienen el formato: `YYYY.MMDDNNNvN`
- YYYY: Año (4 dígitos)
- MM: Mes (2 dígitos)
- DD: Día (2 dígitos)
- NNN: Número secuencial (3 dígitos)
- vN: Versión (opcional, v1, v2, etc.)

Ejemplo: `2402.12345v1` sería febrero 24, 2024, paper #12345, versión 1.

## 🔧 Actualizar los IDs

Una vez que encuentres los IDs, usa el script `update_paper_ids.py` para actualizar automáticamente todos los archivos:

```bash
python update_paper_ids.py
```

O actualiza manualmente cada archivo reemplazando:
- `[ID_PENDIENTE]` o `[ID pendiente]` con el ID real
- `https://arxiv.org/abs/[ID_PENDIENTE]` con `https://arxiv.org/abs/YYYY.MMDDNNNvN`

## ✅ Estado

- **Total de papers pendientes**: 8
- **Papers con IDs encontrados**: 0
- **Última actualización**: [Fecha actual]

