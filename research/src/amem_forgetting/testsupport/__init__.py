"""Test support: import stubs and deterministic fakes.

These exist so that the instrumentation can be developed and wired
against the real ``memory_layer`` code on machines without the ML stack
(no torch / sentence-transformers / litellm). The stubs only activate
when the real package is missing; on a properly provisioned server the
real modules load and the stubs stay dormant.
"""

from amem_forgetting.testsupport.stubs import install_import_stubs
from amem_forgetting.testsupport.fakes import FakeLLMController

__all__ = ["install_import_stubs", "FakeLLMController"]
