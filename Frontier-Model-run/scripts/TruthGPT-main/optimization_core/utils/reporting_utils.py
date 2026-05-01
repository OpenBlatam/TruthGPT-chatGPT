"""
Reporting utilities for optimization_core.

Provides utilities for generating reports and summaries.
"""
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


from .base import BaseOptimizationModel


class ReportSection(BaseOptimizationModel):
    """Report section."""
    title: str
    content: Dict[str, Any]
    order: int = 0


class ReportGenerator:
    """Generator for reports."""
    
    def __init__(self, title: str):
        """
        Initialize report generator.
        
        Args:
            title: Report title
        """
        self.title = title
        self.sections: List[ReportSection] = []
        self.metadata: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
        }
    
    def add_section(
        self,
        title: str,
        content: Dict[str, Any],
        order: int = 0
    ):
        """
        Add a section to the report.
        
        Args:
            title: Section title
            content: Section content
            order: Section order
        """
        section = ReportSection(
            title=title,
            content=content,
            order=order
        )
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.order)
    
    def add_metadata(self, key: str, value: Any):
        """
        Add metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def generate_json(
        self,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON report.
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            Report dictionary
        """
        report = {
            "title": self.title,
            "metadata": self.metadata,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                }
                for section in self.sections
            ]
        }
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def generate_markdown(
        self,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate Markdown report.
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            Markdown string
        """
        lines = [
            f"# {self.title}",
            "",
            f"**Generated:** {self.metadata.get('generated_at', 'Unknown')}",
            "",
        ]
        
        # Add metadata
        if len(self.metadata) > 1:
            lines.append("## Metadata")
            lines.append("")
            for key, value in self.metadata.items():
                if key != "generated_at":
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # Add sections
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(self._format_content(section.content))
            lines.append("")
        
        markdown = "\n".join(lines)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(markdown)
            logger.info(f"Report saved to {output_path}")
        
        return markdown
    
    def _format_content(self, content: Dict[str, Any], indent: int = 0) -> str:
        """Format content for Markdown."""
        lines = []
        prefix = "  " * indent
        
        for key, value in content.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}- **{key}**:")
                lines.append(self._format_content(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}- **{key}**:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._format_content(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}- **{key}**: {value}")
        
        return "\n".join(lines)


def create_report(title: str) -> ReportGenerator:
    """
    Create a report generator.
    
    Args:
        title: Report title
    
    Returns:
        Report generator
    """
    return ReportGenerator(title)













