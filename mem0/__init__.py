import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepmem0")
except importlib.metadata.PackageNotFoundError:  # editable/source checkouts
    __version__ = "0.1.0"

# Marker for tooling that needs to distinguish DeepMem0 from upstream mem0ai
# (e.g. runtime patches that must no-op when the feature is already built in).
__deepmem0__ = True

from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa
