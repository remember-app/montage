from src.integrations.ai.openrouter import OpenRouterProvider

assert OpenRouterProvider().name == "openrouter"
try:
    OpenRouterProvider().ask("", "")
except NotImplementedError:
    print("ok")
else:
    raise AssertionError("openrouter must not pretend to work")
