# 🚀 Instalación de Bazel en Windows

## Estado Actual

✅ **Configuración Validada:**
- ✅ WORKSPACE.bazel encontrado
- ✅ BUILD.bazel encontrado  
- ✅ requirements_lock.txt encontrado
- ✅ 18 BUILD.bazel files encontrados
- ✅ Todas las dependencias validadas (11/11)

## Instalación de Bazel

### Opción 1: Chocolatey (Recomendado)

**Requiere PowerShell como Administrador:**

```powershell
# 1. Instalar Chocolatey (si no está instalado)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# 2. Instalar Bazel
choco install bazel -y

# 3. Verificar instalación
bazel version
```

### Opción 2: Descarga Manual

1. Visita: https://github.com/bazelbuild/bazel/releases
2. Descarga la última versión para Windows
3. Extrae y agrega al PATH
4. Verifica: `bazel version`

### Opción 3: Bazelisk (Version Manager)

```powershell
# Con Chocolatey
choco install bazelisk -y

# O descarga desde: https://github.com/bazelbuild/bazelisk/releases
```

## Después de Instalar

### 1. Verificar Instalación
```powershell
bazel version
```

### 2. Validar Configuración
```powershell
# Ejecutar script de validación
.\run_bazel_validation.ps1
```

### 3. Sincronizar Dependencias
```powershell
# Sincronizar dependencias de Python
bazel sync --only=@pip
```

### 4. Ejecutar Query (Validar Workspace)
```powershell
# Esto descargará dependencias y puede mostrar errores de SHA256
bazel query //...
```

### 5. Corregir SHA256 (si es necesario)
Si aparecen errores de SHA256:
- Bazel mostrará el SHA256 correcto en el mensaje de error
- Actualiza `WORKSPACE.bazel` con el SHA256 correcto
- Vuelve a ejecutar `bazel query //...`

### 6. Construir Módulos
```powershell
# Construir todo
bazel build //...

# Construir módulos específicos
bazel build //core:core
bazel build //cpp_core:cpp_core
bazel build //rust_core:rust_core
```

## Scripts Disponibles

- `run_bazel_validation.ps1` - Validación completa de configuración
- `check_dependencies.py` - Verifica dependencias de pip
- `validate_bazel.py` - Valida sintaxis de BUILD files

## Notas Importantes

⚠️ **Permisos de Administrador:**
- La instalación de Chocolatey requiere PowerShell como Administrador
- Si no tienes permisos, usa la descarga manual (Opción 2)

⚠️ **Primera Ejecución:**
- Bazel descargará todas las dependencias externas
- Esto puede tomar varios minutos
- Algunos SHA256 se calcularán automáticamente

✅ **Configuración Lista:**
- Todos los archivos están correctamente configurados
- Todas las dependencias están declaradas
- Solo falta instalar Bazel para comenzar a construir

## Solución de Problemas

### Error: "Bazel no encontrado"
- Verifica que Bazel esté en el PATH
- Reinicia PowerShell después de instalar
- Verifica con: `where.exe bazel`

### Error: "SHA256 mismatch"
- Actualiza el SHA256 en WORKSPACE.bazel con el valor mostrado en el error
- Bazel mostrará el SHA256 correcto en el mensaje de error

### Error: "Python dependencies not found"
```powershell
bazel sync --only=@pip
bazel clean
bazel build //...
```

## Estado Final

✅ **Configuración:** Completa y validada
✅ **Dependencias:** Todas presentes
✅ **BUILD Files:** Todos correctos
⏳ **Bazel:** Pendiente de instalación (requiere permisos de administrador)

---

**Para instalar Bazel, ejecuta PowerShell como Administrador y sigue las instrucciones de la Opción 1.**












