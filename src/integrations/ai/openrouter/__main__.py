from src.integrations.ai.openrouter import OpenRouterProvider

try:
    OpenRouterProvider().ask("", "")
except NotImplementedError:
    print("ok")
else:
    raise AssertionError("openrouter must not pretend to work")
