"""nightshift: let a local model polish your Markdown draft without touching code or links."""

from .checks import CheckConfig, check_block
from .chunking import Chunk, split
from .diffing import unified_diff, word_diff
from .masking import Masked, MaskingError, mask, unmask
from .pipeline import BlockResult, DocumentResult, PipelineConfig, process_document

__version__ = "0.1.0"

__all__ = [
    "BlockResult",
    "CheckConfig",
    "Chunk",
    "DocumentResult",
    "Masked",
    "MaskingError",
    "PipelineConfig",
    "check_block",
    "mask",
    "process_document",
    "split",
    "unified_diff",
    "unmask",
    "word_diff",
    "__version__",
]
