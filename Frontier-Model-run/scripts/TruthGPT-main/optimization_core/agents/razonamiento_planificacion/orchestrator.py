import logging
import re
import asyncio
from typing import List, Dict, Any, Callable, Protocol, Optional, runtime_checkable, Type

# Contexto de imports
# Standardized absolute imports for TruthGPT/OpenClaw production structure
from agents.memoria_aprendizaje.sqlite_memory import SQLiteMemory, BaseMemory
from agents.memoria_aprendizaje.vector_memory import VectorMemory
from agents.memoria_aprendizaje.core_memory import CoreMemory
from agents.memoria_aprendizaje.core_memory_tools import CoreMemoryAppendTool, CoreMemoryReplaceTool
from agents.razonamiento_planificacion.tools import BaseTool, ToolResult, MCPTool
from agents.models import AgentAction, AgentResponse, InferenceResult, AgentConfig
from agents.engines import AsyncLLMEngine, safe_llm_call
from agents.razonamiento_planificacion.config import settings
from agents.mcp_client import MCPClient
from agents.prompts.prompt_manager import prompt_manager
from agents.exceptions import TruthGPTError, InferenceError, ToolExecutionError

try:
    from agents.observability import global_tracer
except ImportError:
    from ..observability import global_tracer

logger = logging.getLogger(__name__)

# AgentAction and AgentResponse are now imported from .models

class MultiUserReActAgent:
    """
    Orquestador ReAct Multi-Usuario de Grado Empresarial.
    Gestión asíncrona de bucles de razonamiento e integración Pydantic (JSON).
    """

    def __init__(
        self, 
        config: AgentConfig,
        llm_engine: Optional[AsyncLLMEngine] = None, 
        memory: Optional[BaseMemory] = None,
        vector_memory: Optional[VectorMemory] = None,
        custom_system_instructions: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None
    ):
        self.config = config
        self.llm = llm_engine or config.llm_engine
        self.memory = memory or SQLiteMemory(db_path=config.memory_db_path)
        self.vector_memory = vector_memory
        self.core_memory = CoreMemory()
        self.tools: Dict[str, BaseTool] = {}
        self.custom_system_instructions = custom_system_instructions
        self.use_reflexion = config.use_reflexion
        self.name = "MultiUserReActAgent"

        if tools:
            for tool in tools:
                self.register_tool(tool)
        
        # Add memory self-update tools
        self.register_tool(CoreMemoryAppendTool(self.core_memory))
        self.register_tool(CoreMemoryReplaceTool(self.core_memory))

    def register_tool(self, tool: BaseTool) -> None:
        """Registra una herramienta disponible para el agente."""
        self.tools[tool.name] = tool
        logger.info(f"Agente {self.name}: Herramienta '{tool.name}' registrada.")

    async def load_mcp_tools(self, server_url: str):
        """
        Descubre y registra dinámicamente herramientas desde un servidor MCP.
        """
        logger.info(f"Cargando herramientas MCP desde {server_url}...")
        client = MCPClient(server_url)
        tools_info = await client.list_tools()
        
        for t_info in tools_info:
            mcp_tool = MCPTool(client, t_info)
            self.register_tool(mcp_tool)
        
        logger.info(f"Se cargaron {len(tools_info)} herramientas MCP.")

    def _get_system_instructions(self) -> str:
        """Genera instrucciones dinámicas usando el PromptManager centralizado."""
        tools_list = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        
        # Build prompt using centralized templates
        base = prompt_manager.get_prompt("base_agent", name=settings.AGENT_NAME, role="Enterprise AI Assistant")
        react = prompt_manager.get_prompt("react_core")
        json_schema = prompt_manager.get_prompt("json_output", schema=AgentAction.model_json_schema())
        
        instructions = f"{base}\n{react}\n\nTienes acceso a estas herramientas:\n{tools_list}\n\n"
        if self.custom_system_instructions:
            instructions += f"{self.custom_system_instructions}\n\n"
            
        return instructions + json_schema

    async def _format_context(self, user_id: str) -> str:
        history = await self.memory.get_history(user_id)
        if not history:
            return f"--- MEMORIA PRIVADA ({user_id}) ---\nSin historial previo.\n"
        
        context = f"--- MEMORIA PRIVADA ({user_id}) ---\n"
        for msg in history:
            context += f"{msg['role'].upper()}: {msg['content']}\n"
        context += "--------------------------------------\n"
        return context

    async def process_message(self, user_id: str, message: str) -> AgentResponse:
        """
        Procesa un mensaje de forma asíncrona aislando el contexto por usuario.
        """
        logger.info(f"Iniciando proceso asíncrono para {user_id}")
        await self.memory.add_message(user_id, "user", message)
        
        system_prompt = self._get_system_instructions()
        user_context = await self._format_context(user_id)
        
        # Inject Long-Term RAG Context
        if self.vector_memory and self.vector_memory.enabled:
            rag_context = await self.vector_memory.get_context_for_prompt(message, user_id)
            if rag_context:
                user_context += f"\n{rag_context}\n"
        
        core_context = await self.core_memory.get_formatted_context(user_id)
        system_prompt = f"{system_prompt}\n{core_context}"
        
        current_prompt = f"{system_prompt}\n{user_context}\nUSER: {message}\nTRUTHGPT: "
        
        # Iniciar traza de observabilidad
        trace_id = global_tracer.start_trace(name="process_message", agent_name="MultiUserReActAgent")
        
        # safe_llm_call handles tracing and retries internally now.
        
        for i in range(settings.MAX_ITERATIONS):
            # Inferencia asíncrona robusta (con reintentos)
            response = await safe_llm_call(self.llm, current_prompt, trace_id)
            
            try:
                import json
                clean_resp = response.strip()
                if clean_resp.startswith("```json"):
                    clean_resp = clean_resp[7:-3].strip()
                elif clean_resp.startswith("```"):
                    clean_resp = clean_resp[3:-3].strip()
                    
                action = AgentAction.model_validate_json(clean_resp)
                
                if action.tool:
                    if action.tool in self.tools:
                        tool_instance = self.tools[action.tool]
                        if tool_instance.requires_approval:
                            logger.info(f"HITL PAUSE: Require aprobación para {action.tool}")
                            # Guardamos la intención en memoria
                            await self.memory.add_message(user_id, "assistant", clean_resp)
                            await self.memory.add_message(user_id, "assistant", f"⏳ Esperando aprobación manual para ejecutar: {action.tool}")
                            return AgentResponse(
                                content=f"⏳ Esperando aprobación para: {action.tool}",
                                action_type="approval_required",
                                metadata={"tool": action.tool, "cmd": action.tool_input}
                            )
                            
                        logger.info(f"Ejecutando {action.tool} asíncronamente...")
                        result = await self._execute_tool_action(trace_id, action, user_id)
                    else:
                        result = f"Error: La herramienta '{action.tool}' no existe."
                    current_prompt += f"{clean_resp}\nTOOL_RESULT: {result}\nTRUTHGPT: "
                    
                elif action.final_answer:
                    # Llegamos a la respuesta final
                    if self.use_reflexion:
                        logger.info("Auto-Reflexion: Evaluando respuesta interna...")
                        critique_prompt = (
                            f"{current_prompt}\n{clean_resp}\n"
                            "[SISTEMA INTERNO]: Evalúa críticamente tu respuesta anterior frente a la petición. "
                            "¿Resuelve completamente el problema o la pregunta? "
                            "Si la respuesta es perfecta y sin errores, responde EXACTAMENTE '<final>APROBADO</final>'. "
                            "Si falta información o hubo un error, escribe tu crítica y planifica el siguiente paso (puedes usar herramientas de nuevo)."
                        )
                        critique_response = await safe_llm_call(self.llm, critique_prompt, trace_id)
                        
                        if "<final>APROBADO</final>" in critique_response:
                            logger.info("Auto-Reflexion: Aprobado.")
                            if self.vector_memory and self.vector_memory.enabled:
                                await self.vector_memory.add_episodic(user_id, "ReActAgent", f"User: {message}\nSuccess: {action.final_answer}")
                                asyncio.create_task(self.vector_memory.compact_episodic_memory(user_id, _safe_llm_call))
                            
                            await self.memory.add_message(user_id, "assistant", action.final_answer)
                            global_tracer.finish_trace(trace_id)
                            return AgentResponse(content=action.final_answer, action_type="final_answer")
                        else:
                            logger.info("Auto-Reflexion: Crítica detectó áreas de mejora. Reintentando...")
                            current_prompt = f"{critique_prompt}\n{critique_response}\nTRUTHGPT: "
                    else:
                        if self.vector_memory and self.vector_memory.enabled:
                            await self.vector_memory.add_episodic(user_id, "ReActAgent", f"User: {message}\nAnswer: {action.final_answer}")
                            asyncio.create_task(self.vector_memory.compact_episodic_memory(user_id, _safe_llm_call))
                        await self.memory.add_message(user_id, "assistant", action.final_answer)
                        return AgentResponse(content=action.final_answer, action_type="final_answer")
                elif action.handoff:
                    logger.info(f"Iniciando Swarm Handoff hacia: {action.handoff}")
                    await self.memory.add_message(user_id, "assistant", f"Transferring control to {action.handoff}...")
                    return AgentResponse(
                        content=f"Transferring to {action.handoff}...",
                        action_type="handoff",
                        handoff_target=action.handoff
                    )
                else:
                    raise ValueError("Debes proveer 'tool', 'respuesta_final' o 'handoff' en tu JSON.")
                    
            except Exception as e:
                logger.warning(f"Error parseando Pydantic JSON: {e}")
                err_msg = f"Tu respuesta violó el esquema JSON obligatorio. Detalle: {str(e)}"
                current_prompt += f"\n[ERROR DE SISTEMA]: {err_msg}\nCorrige y responde solo en JSON.\nTRUTHGPT: "
        
        fallback = "Límite de razonamiento alcanzado. Por favor, simplifica tu petición."
        await self.memory.add_message(user_id, "assistant", fallback)
        global_tracer.finish_trace(trace_id)
        return AgentResponse(content=fallback, action_type="final_answer")

    from typing import AsyncIterator
    
    async def astream_process_message(self, user_id: str, message: str) -> 'AsyncIterator[str]':
        """
        Procesa un mensaje de forma asíncrona y hace yield de los pasos (Streaming / SSE).
        Emite JSON strings que representan eventos o estados parciales.
        """
        import json
        logger.info(f"Iniciando proceso STREAMING para {user_id}")
        await self.memory.add_message(user_id, "user", message)
        
        system_prompt = self._get_system_instructions()
        user_context = await self._format_context(user_id)
        
        # Inject Long-Term RAG Context
        if self.vector_memory and self.vector_memory.enabled:
            rag_context = await self.vector_memory.get_context_for_prompt(message, user_id)
            if rag_context:
                user_context += f"\n{rag_context}\n"
        
        core_context = await self.core_memory.get_formatted_context(user_id)
        system_prompt = f"{system_prompt}\n{core_context}"
                
        current_prompt = f"{system_prompt}\n{user_context}\nUSER: {message}\nTRUTHGPT: "
        
        trace_id = global_tracer.start_trace(name="astream_process", agent_name="MultiUserReActAgent")
        
        # safe_llm_call handles tracing and retries internally now.
        
        for i in range(settings.MAX_ITERATIONS):
            yield json.dumps({"event": "thinking", "iteration": i+1}) + "\n"
            
            response = await safe_llm_call(self.llm, current_prompt, trace_id)
            
            try:
                clean_resp = response.strip()
                if clean_resp.startswith("```json"):
                    clean_resp = clean_resp[7:-3].strip()
                elif clean_resp.startswith("```"):
                    clean_resp = clean_resp[3:-3].strip()
                    
                action = AgentAction.model_validate_json(clean_resp)
                
                if action.tool:
                    if action.tool in self.tools:
                        tool_instance = self.tools[action.tool]
                        if tool_instance.requires_approval:
                            logger.info(f"STREAMING HITL PAUSE: Require aprobación para {action.tool}")
                            yield json.dumps({"event": "requires_approval", "tool": action.tool, "cmd": action.cmd}) + "\n"
                            
                            # Guardamos la intención en memoria
                            await self.memory.add_message(user_id, "assistant", clean_resp)
                            approval_msg = f"<WAITING_FOR_APPROVAL tool='{action.tool}' cmd='{action.cmd}'/>"
                            await self.memory.add_message(user_id, "assistant", f"⏳ Esperando aprobación manual para ejecutar: {action.tool}")
                            yield json.dumps({"event": "final_answer", "content": approval_msg}) + "\n"
                            return
                            
                        yield json.dumps({"event": "tool_call", "tool": action.tool, "cmd": action.cmd}) + "\n"
                        result = await self._execute_tool_action(trace_id, action, user_id)
                    else:
                        yield json.dumps({"event": "tool_call", "tool": action.tool, "cmd": action.cmd}) + "\n"
                        result = f"Error: La herramienta '{action.tool}' no existe."
                    
                    yield json.dumps({"event": "tool_result", "tool": action.tool, "result": str(result)[:200] + "..."}) + "\n"
                    current_prompt += f"{clean_resp}\nTOOL_RESULT: {result}\nTRUTHGPT: "
                    
                elif action.final_answer:
                    if self.use_reflexion:
                        yield json.dumps({"event": "reflexion", "status": "evaluating"}) + "\n"
                        critique_prompt = (
                            f"{current_prompt}\n{clean_resp}\n"
                            "[SISTEMA INTERNO]: Evalúa críticamente tu respuesta anterior frente a la petición. "
                            "¿Resuelve completamente el problema o la pregunta? "
                            "Si la respuesta es perfecta y sin errores, responde EXACTAMENTE '<final>APROBADO</final>'. "
                            "Si falta información o hubo un error, escribe tu crítica y planifica el siguiente paso (puedes usar herramientas de nuevo)."
                        )
                        critique_response = await safe_llm_call(self.llm, critique_prompt, trace_id)
                        
                        if "<final>APROBADO</final>" in critique_response:
                            yield json.dumps({"event": "reflexion_approved"}) + "\n"
                            if self.vector_memory and self.vector_memory.enabled:
                                await self.vector_memory.add_episodic(user_id, "ReActAgent", f"User: {message}\nSuccess: {action.final_answer}")
                                asyncio.create_task(self.vector_memory.compact_episodic_memory(user_id, _safe_llm_call))
                            await self.memory.add_message(user_id, "assistant", action.final_answer)
                            yield json.dumps({"event": "final_answer", "content": action.final_answer}) + "\n"
                            global_tracer.finish_trace(trace_id)
                            return
                        else:
                            yield json.dumps({"event": "reflexion_rejected", "critique": critique_response}) + "\n"
                            current_prompt = f"{critique_prompt}\n{critique_response}\nTRUTHGPT: "
                    else:
                        if self.vector_memory and self.vector_memory.enabled:
                            await self.vector_memory.add_episodic(user_id, "ReActAgent", f"User: {message}\nAnswer: {action.final_answer}")
                            asyncio.create_task(self.vector_memory.compact_episodic_memory(user_id, _safe_llm_call))
                        await self.memory.add_message(user_id, "assistant", action.final_answer)
                        yield json.dumps({"event": "final_answer", "content": action.final_answer}) + "\n"
                        return
                elif action.handoff:
                    logger.info(f"STREAMING: Iniciando Swarm Handoff hacia: {action.handoff}")
                    yield json.dumps({"event": "handoff", "target": action.handoff}) + "\n"
                    handoff_msg = f"<HANDOFF target='{action.handoff}'/>"
                    await self.memory.add_message(user_id, "assistant", f"Transferring control to {action.handoff}...")
                    yield json.dumps({"event": "final_answer", "content": handoff_msg}) + "\n"
                    return
                else:
                    raise ValueError("Debes proveer 'tool', 'final_answer' o 'handoff' en tu JSON.")
                    
            except Exception as e:
                logger.warning(f"Error parseando Pydantic JSON: {e}")
                yield json.dumps({"event": "error", "message": f"Syntax error recovering: {str(e)}"}) + "\n"
                current_prompt += f"\n[ERROR DE SISTEMA]: Tu respuesta violó el esquema JSON obligatorio. Detalle: {str(e)}\nCorrige y responde solo en JSON.\nTRUTHGPT: "
        
        fallback = "Límite de razonamiento alcanzado. Por favor, simplifica tu petición."
        await self.memory.add_message(user_id, "assistant", fallback)
        yield json.dumps({"event": "error", "message": fallback}) + "\n"
        global_tracer.finish_trace(trace_id)

    async def _execute_tool_action(self, trace_id: str, action: AgentAction, user_id: str) -> str:
        """Helper para ejecutar una herramienta y manejar señales internas (Core Memory)."""
        tool_instance = self.tools[action.tool]
        tool_span = global_tracer.start_span(trace_id, name=action.tool, kind="tool_call", input_data=str(action.tool_input))
        
        try:
            raw_result = await tool_instance.run(str(action.tool_input) or "")
            
            # Handle ToolResult signals
            if isinstance(raw_result, ToolResult):
                result_str = raw_result.output
                if raw_result.signal == "core_memory_append":
                    block = raw_result.metadata.get("block")
                    content = raw_result.metadata.get("content")
                    await self.core_memory.append_to_block(user_id, block, content)
                    result_str = f"SYSTEM: Memoria CORE ({block}) actualizada."
                elif raw_result.signal == "core_memory_replace":
                    block = raw_result.metadata.get("block")
                    content = raw_result.metadata.get("content")
                    await self.core_memory.update_block(user_id, block, content)
                    result_str = f"SYSTEM: Memoria CORE ({block}) remplazada totalmente."
            else:
                result_str = str(raw_result)
                
            tool_span.finish(output=result_str)
            return result_str
            
        except Exception as e:
            logger.error(f"Error ejecutando {action.tool}: {e}")
            tool_span.finish(output=str(e), status="error")
            raise ToolExecutionError(f"Tool {action.tool} failed: {str(e)}", metadata={"tool": action.tool})

