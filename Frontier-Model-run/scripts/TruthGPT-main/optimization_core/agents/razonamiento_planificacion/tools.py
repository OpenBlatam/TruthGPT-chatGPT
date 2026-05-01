import os
import sys
import asyncio
import subprocess
import logging
import httpx
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ToolResult:
    """
    Standardized result from a tool execution.
    Can contain the final output string and optional internal signals for the orchestrator.
    """
    def __init__(
        self, 
        output: str, 
        metadata: Optional[Dict[Any, Any]] = None, 
        signal: Optional[str] = None
    ):
        self.output = output
        self.metadata = metadata or {}
        self.signal = signal  # e.g., "core_memory_update"

class BaseTool(ABC):
    """
    Clase base para herramientas automatizadas. 
    Permite que el agente obtenga la descripción automáticamente del docstring.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre único de la herramienta."""
        pass

    @property
    def description(self) -> str:
        """Descripción extraída del docstring para el LLM."""
        return self.__doc__.strip() if self.__doc__ else "No description available."
        
    @property
    def requires_approval(self) -> bool:
        """Si es True, la ejecución requerirá aprobación manual del usuario (HITL)."""
        return False

    @abstractmethod
    async def run(self, arg: str) -> Any:
        """
        Ejecución asíncrona de la herramienta. 
        Puede devolver un string simple o un objeto ToolResult.
        """
        pass

# --- Herramientas de Sistema ---

class SystemBashTool(BaseTool):
    """
    Ejecuta comandos de terminal de forma segura (bash/shell). 
    Útil para listar archivos, comprobar procesos o leer información del sistema.
    """
    name = "system_bash"
    
    @property
    def requires_approval(self) -> bool:
        return True

    async def run(self, cmd: str) -> str:
        # Guardrail básico: prevenir comandos destructivos
        forbidden = ["rm ", "format ", "> /dev/", "chmod ", ":(){ :|:& };:"]
        if any(f in cmd.lower() for f in forbidden):
            return "Error: Comando prohibido por políticas de seguridad."
            
        try:
            # Ejecutar en proceso separado
            process = await subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10)
            return process.decode('utf-8')
        except subprocess.CalledProcessError as e:
            return f"Error al ejecutar: {e.output.decode('utf-8')}"
        except Exception as e:
            return f"Excepción de sistema: {str(e)}"

# --- Herramientas Web ---

class WebSearchTool(BaseTool):
    """
    Realiza una búsqueda web y extrae información relevante. 
    Acepta una consulta (query) y devuelve un resumen de los resultados.
    """
    name = "web_search"

    async def run(self, query: str) -> str:
        # Mock de búsqueda (en prod usar DuckDuckGo API o similar)
        logger.info(f"Buscando en la web: {query}")
        if "bitcoin" in query.lower():
            return "Resumen Web: El precio de Bitcoin sigue volátil, rondando los $60k-$70k USD según los últimos reportes de CoinMarketCap."
        return f"No se encontraron resultados específicos para '{query}'."

class WebReaderTool(BaseTool):
    """
    Lee el contenido textual de una URL específica usando Crawl4AI. 
    Extrae texto limpio y estructurado en Markdown perfecto para el LLM.
    """
    name = "web_reader"

    async def run(self, url: str) -> str:
        if not url.startswith("http"):
            return "Error: URL inválida."
            
        try:
            from crawl4ai import AsyncWebCrawler
            
            logger.info(f"Crawling URL con Crawl4AI: {url}")
            async with AsyncWebCrawler(verbose=True) as crawler:
                result = await crawler.arun(url=url)
                
                if result.success:
                    # Return perfectly formatted markdown
                    markdown = result.markdown
                    return markdown[:5000] + "\n...[Truncated]" if len(markdown) > 5000 else markdown
                else:
                    return f"Error al crawlear: {result.error_message}"
                    
        except ImportError:
            # Fallback to bs4 if crawl4ai is not installed
            logger.warning("crawl4ai no instalado. Usando fallback bs4.")
            try:
                import bs4
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    soup = bs4.BeautifulSoup(response.text, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    return text[:5000]
            except Exception as e:
                return f"Error en fallback bs4: {str(e)}"
        except Exception as e:
            return f"Error inesperado en WebReaderTool: {str(e)}"

# --- Herramientas de Sistema de Archivos y Código ---

class FileReadTool(BaseTool):
    """
    Lee el contenido de un archivo local.
    Acepta la ruta absoluta o relativa del archivo y devuelve su contenido.
    """
    name = "file_read"

    async def run(self, filepath: str) -> str:
        try:
            if not os.path.exists(filepath):
                return f"Error: El archivo '{filepath}' no existe."
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()[:5000]  # Limitar a 5000 caracteres
        except Exception as e:
            return f"Error al leer archivo: {str(e)}"

class FileWriteTool(BaseTool):
    """
    Escribe contenido en un archivo local.
    Acepta un formato específico: ruta:::contenido
    Ejemplo: /ruta/al/archivo.txt:::Este es el contenido.
    """
    name = "file_write"

    async def run(self, cmd: str) -> str:
        try:
            parts = cmd.split(":::", 1)
            if len(parts) != 2:
                return "Error: Formato inválido. Use 'ruta:::contenido'."
            filepath, content = parts
            filepath = filepath.strip()
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Éxito: Contenido escrito en '{filepath}'."
        except Exception as e:
            return f"Error al escribir archivo: {str(e)}"

class PythonExecutionTool(BaseTool):
    """
    Ejecuta código Python de forma asíncrona dentro de un contenedor Docker aislado (Sandbox).
    Acepta código fuente en Python y devuelve la salida.
    """
    name = "python_execute"
    
    @property
    def requires_approval(self) -> bool:
        return True

    async def run(self, code: str) -> str:
        try:
            import docker
            from docker.errors import ContainerError, ImageNotFound, APIError
            
            client = docker.from_env()
            
            def _run_docker_securely():
                # Pull image if not exists
                try:
                    client.images.get("python:3.9-slim")
                except ImageNotFound:
                    logger.info("Descargando imagen python:3.9-slim para el sandbox...")
                    client.images.pull("python:3.9-slim")

                # Ejecutar de forma segura usando un contenedor efímero
                result = client.containers.run(
                    "python:3.9-slim",
                    command=["python", "-c", code],
                    remove=True,
                    network_mode="none", # Aislar red
                    mem_limit="128m",    # Limitar memoria
                    stderr=True,
                    stdout=True
                )
                return result.decode("utf-8")
                
            output = await asyncio.to_thread(_run_docker_securely)
            return output[:5000] if output else "Ejecutado sin salida."
            
        except ImportError:
            return "Error: La librería 'docker' no está instalada. Instala con 'pip install docker'."
        except Exception as e:
            return f"Error en el Sandbox de Docker: {str(e)}"


# --- Herramientas de Delegación (Multi-Agente) ---

class DelegateTaskTool(BaseTool):
    """
    Delega una sub-tarea compleja a otro agente del enjambre.
    Acepta el nombre del agente y la tarea en formato 'agente:::tarea_a_completar'.
    Ejemplo: MarketingAgent:::Escribe un tweet sobre este resumen.
    Si no sabes qué agente usar, usa 'swarm', ej: swarm:::Crea un reporte de estos datos.
    """
    name = "delegate_task"

    def __init__(self, agent_client: Any = None):
        """Require AgentClient instance to allow recursive calling."""
        self.agent_client = agent_client

    async def run(self, cmd: str) -> str:
        if not self.agent_client:
            return "Error: DelegateTaskTool requiere una instancia de AgentClient."
            
        try:
            parts = cmd.split(":::", 1)
            if len(parts) != 2:
                return "Error: Formato inválido. Use 'agente:::tarea'."
            
            agent_target, task = parts
            agent_target = agent_target.strip()
            task = task.strip()
            
            logger.info(f"Delegando tarea a '{agent_target}': {task[:50]}...")
            
            # Isolated sub-memory namespace for the delegated task
            sub_user_id = f"delegate_{agent_target}_temp"
            
            # Run the task through the orchestrator/client
            # This allows hierarchical agent branching!
            result = await self.agent_client.run(user_id=sub_user_id, prompt=task)
            return f"Respuesta de {agent_target}:\n{result}"
        except Exception as e:
            return f"Error en delegación de tarea: {str(e)}"

# --- Herramientas de Interoperabilidad (MCP) ---

class MCPTool(BaseTool):
    """
    Wrapper para herramientas externas del Model Context Protocol (MCP).
    Permite usar herramientas servidas por un MCP Server remoto.
    """
    def __init__(self, mcp_client: Any, tool_info: Dict[str, Any]):
        self.mcp_client = mcp_client
        self._name = tool_info["name"]
        self._description = tool_info.get("description", "No description available via MCP.")
        self.arguments_schema = tool_info.get("inputSchema", {})

    @property
    def name(self) -> str:
        return f"mcp_{self._name}"

    @property
    def description(self) -> str:
        return f"[MCP Tool] {self._description}\nSchema: {json.dumps(self.arguments_schema)}"

    async def run(self, arg: str) -> str:
        """
        Ejecuta la herramienta MCP. Intenta parsear JSON si la herramienta lo requiere.
        """
        try:
            # MCP tools often expect a JSON object for arguments
            try:
                args_dict = json.loads(arg)
            except json.JSONDecodeError:
                args_dict = {"input": arg} # Fallback simple
                
            result = await self.mcp_client.call_tool(self._name, args_dict)
            return str(result)
        except Exception as e:
            return f"Error executing MCP tool '{self._name}': {str(e)}"

