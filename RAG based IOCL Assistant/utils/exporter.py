# """
# Export Manager — chat history, search results, and PDF generation.

# Supports plain-text, CSV, and PDF export of conversations and search data.
# """

# from __future__ import annotations

# import csv
# import io
# import logging
# import textwrap
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# from fpdf import FPDF

# logger = logging.getLogger(__name__)

# EXPORT_DIR = Path(__file__).resolve().parent.parent / "exports"
# EXPORT_DIR.mkdir(exist_ok=True)


# class PDFReport(FPDF):
#     """Minimal branded PDF report for chat/session export."""

#     def header(self) -> None:
#         self.set_font("Helvetica", "B", 14)
#         self.cell( 0, 10, safe_pdf_text("IOCL Knowledge Assistant - Report"), new_x="LMARGIN", new_y="NEXT", align="C" )
#         self.set_font("Helvetica", "", 9)
#         self.cell( 0, 6, safe_pdf_text(f"Generated: {datetime.now():%Y-%m-%d %H:%M}"), new_x="LMARGIN", new_y="NEXT", align="C" )
#         self.ln(4)

#     def footer(self) -> None:
#         self.set_y(-15)
#         self.set_font("Helvetica", "I", 8)
#         self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


# class ExportManager:
#     """Handles all export operations for the Knowledge Assistant."""

#     # ------------------------------------------------------------------
#     # Chat History — TXT
#     # ------------------------------------------------------------------
#     @staticmethod
#     def chat_to_txt(messages: List[Dict[str, str]]) -> str:
#         """Return a plain-text representation of a chat session."""
#         lines: List[str] = []
#         lines.append("=" * 60)
#         lines.append("  IOCL Knowledge Assistant — Chat History")
#         lines.append(f"  Exported: {datetime.now():%Y-%m-%d %H:%M:%S}")
#         lines.append("=" * 60)
#         lines.append("")
#         for msg in messages:
#             role = msg.get("role", "user").title()
#             content = msg.get("content", "")
#             lines.append(f"[{role}]")
#             lines.append(content)
#             lines.append("")
#         lines.append("-" * 60)
#         lines.append("End of chat history.")
#         return "\n".join(lines)

#     @staticmethod
#     def save_txt(text: str, filename: Optional[str] = None) -> Path:
#         if filename is None:
#             filename = f"chat_{datetime.now():%Y%m%d_%H%M%S}.txt"
#         path = EXPORT_DIR / filename
#         path.write_text(text, encoding="utf-8")
#         return path

#     # ------------------------------------------------------------------
#     # Chat History — PDF
#     # ------------------------------------------------------------------
#     @staticmethod
#     def chat_to_pdf(messages: List[Dict[str, str]]) -> bytes:
#         """Generate a PDF report of the chat session."""
#         pdf = PDFReport()
#         pdf.alias_nb_pages()
#         pdf.set_auto_page_break(auto=True, margin=20)
#         pdf.add_page()

#         for msg in messages:
#             role = msg.get("role", "user").title()
#             content = msg.get("content", "")

#             pdf.set_font("Helvetica", "B", 11)
#             pdf.set_text_color(50, 50, 150) if role == "User" else pdf.set_text_color(30, 120, 30)
#             pdf.cell(0, 8, f"{role}:", new_x="LMARGIN", new_y="NEXT")

#             pdf.set_font("Helvetica", "", 10)
#             pdf.set_text_color(40, 40, 40)
#             # Wrap long lines for PDF layout
#             for line in content.split("\n"):
#                 wrapped = textwrap.wrap(line, width=90) or [""]
#                 for wl in wrapped:
#                     pdf.multi_cell(0, 5, wl)
#             pdf.ln(4)

#         return pdf.output()

#     @staticmethod
#     def save_pdf(pdf_bytes: bytes, filename: Optional[str] = None) -> Path:
#         if filename is None:
#             filename = f"chat_{datetime.now():%Y%m%d_%H%M%S}.pdf"
#         path = EXPORT_DIR / filename
#         path.write_bytes(pdf_bytes)
#         return path

#     # ------------------------------------------------------------------
#     # Search Results — CSV
#     # ------------------------------------------------------------------
#     @staticmethod
#     def results_to_csv(results: List[Dict[str, Any]]) -> str:
#         """Serialize search results as a CSV string."""
#         if not results:
#             return ""
#         output = io.StringIO()
#         # Omit internal keys
#         keys = [k for k in results[0].keys() if not k.startswith("_")]
#         writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
#         writer.writeheader()
#         for row in results:
#             filtered = {k: v for k, v in row.items() if not k.startswith("_")}
#             writer.writerow(filtered)
#         return output.getvalue()

#     @staticmethod
#     def save_csv(csv_text: str, filename: Optional[str] = None) -> Path:
#         if filename is None:
#             filename = f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"
#         path = EXPORT_DIR / filename
#         path.write_text(csv_text, encoding="utf-8")
#         return path

#     # ------------------------------------------------------------------
#     # Session persistence
#     # ------------------------------------------------------------------
#     @staticmethod
#     def save_session(messages: List[Dict[str, str]], session_id: str = "default") -> Path:
#         """Persist a full session to disk as a TXT file."""
#         text = ExportManager.chat_to_txt(messages)
#         filename = f"session_{session_id}_{datetime.now():%Y%m%d_%H%M%S}.txt"
#         return ExportManager.save_txt(text, filename)













"""python

Export Manager — chat history, search results, and PDF generation.

Supports plain-text, CSV, and PDF export of conversations and search data.
"""

from __future__ import annotations

import csv
import io
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpdf import FPDF

logger = logging.getLogger(__name__)

EXPORT_DIR = Path(__file__).resolve().parent.parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)


def safe_pdf_text(text: str) -> str:
    """
    Convert unsupported Unicode characters into safe Latin-1 text
    so FPDF doesn't crash.
    """
    if text is None:
        return ""

    replacements = {
        "—": "-",
        "–": "-",
        "•": "*",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "₹": "Rs.",
        "✓": "OK",
        "→": "->",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode("latin-1", "replace").decode("latin-1")


class PDFReport(FPDF):
    """Minimal branded PDF report for chat/session export."""

    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)

        self.cell(
            0,
            10,
            safe_pdf_text("IOCL Knowledge Assistant - Report"),
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )

        self.set_font("Helvetica", "", 9)

        self.cell(
            0,
            6,
            safe_pdf_text(
                f"Generated: {datetime.now():%Y-%m-%d %H:%M}"
            ),
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )

        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)

        self.cell(
            0,
            10,
            safe_pdf_text(
                f"Page {self.page_no()}/{{nb}}"
            ),
            align="C",
        )


class ExportManager:
    """Handles all export operations for the Knowledge Assistant."""

    # ------------------------------------------------------------------
    # Chat History — TXT
    # ------------------------------------------------------------------
    @staticmethod
    def chat_to_txt(messages: List[Dict[str, str]]) -> str:
        """Return a plain-text representation of a chat session."""

        lines: List[str] = []

        lines.append("=" * 60)
        lines.append("  IOCL Knowledge Assistant - Chat History")
        lines.append(
            f"  Exported: {datetime.now():%Y-%m-%d %H:%M:%S}"
        )
        lines.append("=" * 60)
        lines.append("")

        for msg in messages:
            role = msg.get("role", "user").title()
            content = msg.get("content", "")

            lines.append(f"[{role}]")
            lines.append(content)
            lines.append("")

        lines.append("-" * 60)
        lines.append("End of chat history.")

        return "\n".join(lines)

    @staticmethod
    def save_txt(text: str, filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = (
                f"chat_{datetime.now():%Y%m%d_%H%M%S}.txt"
            )

        path = EXPORT_DIR / filename
        path.write_text(text, encoding="utf-8")

        return path

    # ------------------------------------------------------------------
    # Chat History — PDF
    # ------------------------------------------------------------------
    @staticmethod
    def chat_to_pdf(messages: List[Dict[str, str]]) -> bytes:
        """Generate a PDF report of the chat session."""

        pdf = PDFReport()

        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        for msg in messages:
            role = safe_pdf_text(
                msg.get("role", "user").title()
            )

            content = safe_pdf_text(
                msg.get("content", "")
            )

            pdf.set_font("Helvetica", "B", 11)

            if role == "User":
                pdf.set_text_color(50, 50, 150)
            else:
                pdf.set_text_color(30, 120, 30)

            pdf.cell(
                0,
                8,
                f"{role}:",
                new_x="LMARGIN",
                new_y="NEXT",
            )

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)

            for line in content.split("\n"):
                wrapped = (
                    textwrap.wrap(line, width=90)
                    or [""]
                )

                for wl in wrapped:
                    text = safe_pdf_text(wl)

                    pdf.set_x(pdf.l_margin)

                    available_width = pdf.w - pdf.l_margin - pdf.r_margin

                    pdf.multi_cell(
                        available_width,
                        5,
                        text
                    )

            pdf.ln(4)

        pdf_output = pdf.output()

        if isinstance(pdf_output, bytearray):
            return bytes(pdf_output)

        if isinstance(pdf_output, str):
            return pdf_output.encode("latin-1")

        return pdf_output

    @staticmethod
    def save_pdf(
        pdf_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Path:

        if filename is None:
            filename = (
                f"chat_{datetime.now():%Y%m%d_%H%M%S}.pdf"
            )

        path = EXPORT_DIR / filename
        path.write_bytes(pdf_bytes)

        return path

    # ------------------------------------------------------------------
    # Search Results — CSV
    # ------------------------------------------------------------------
    @staticmethod
    def results_to_csv(
        results: List[Dict[str, Any]]
    ) -> str:

        if not results:
            return ""

        output = io.StringIO()

        keys = [
            k
            for k in results[0].keys()
            if not k.startswith("_")
        ]

        writer = csv.DictWriter(
            output,
            fieldnames=keys,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in results:
            filtered = {
                k: v
                for k, v in row.items()
                if not k.startswith("_")
            }

            writer.writerow(filtered)

        return output.getvalue()

    @staticmethod
    def save_csv(
        csv_text: str,
        filename: Optional[str] = None,
    ) -> Path:

        if filename is None:
            filename = (
                f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"
            )

        path = EXPORT_DIR / filename
        path.write_text(csv_text, encoding="utf-8")

        return path

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------
    @staticmethod
    def save_session(
        messages: List[Dict[str, str]],
        session_id: str = "default",
    ) -> Path:

        text = ExportManager.chat_to_txt(messages)

        filename = (
            f"session_{session_id}_"
            f"{datetime.now():%Y%m%d_%H%M%S}.txt"
        )

        return ExportManager.save_txt(
            text,
            filename,
        )

