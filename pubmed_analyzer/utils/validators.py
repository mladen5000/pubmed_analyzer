import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of PDF validation"""
    is_valid: bool
    file_size: int = 0
    error_message: Optional[str] = None
    details: Dict[str, Any] = None


class PDFValidator:
    """Utility class for validating PDF files"""

    @staticmethod
    def is_valid_pdf(file_path: str, min_size: int = 1024) -> bool:
        """
        Validate that a file is a proper PDF

        Args:
            file_path: Path to the PDF file
            min_size: Minimum file size in bytes

        Returns:
            True if file is a valid PDF, False otherwise
        """
        try:
            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                logger.debug(f"PDF validation failed: {file_path} does not exist")
                return False

            # Check file size
            if path.stat().st_size < min_size:
                logger.debug(f"PDF validation failed: {file_path} too small ({path.stat().st_size} bytes)")
                return False

            # Check PDF magic bytes
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if not header.startswith(b'%PDF'):
                    logger.debug(f"PDF validation failed: {file_path} missing PDF header")
                    return False

            return True

        except Exception as e:
            logger.error(f"PDF validation error for {file_path}: {e}")
            return False

    @staticmethod
    def cleanup_invalid_pdf(file_path: str) -> bool:
        """
        Remove invalid PDF file

        Args:
            file_path: Path to the PDF file

        Returns:
            True if file was removed, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed invalid PDF: {file_path}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to remove invalid PDF {file_path}: {e}")
            return False