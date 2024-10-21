try:
    from llmstudio_tracker.tracker import Tracker, TrackingConfig
except ImportError:
    name = None
    message = f"{name} requires the 'llmstudio-tracker' package. Add it to the dependencies, or install it using 'pip install llmstudio[tracker]'."

    class Tracker:
        def __init__(self, *args, **kwargs):
            raise ImportError(message.format(name="Tracker"))

    class TrackingConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError(message.format(name="TrackingConfig"))


try:
    from llmstudio_proxy.provider import LLMProxyProvider, ProxyConfig
except ImportError:
    name = None
    message = f"{name} requires the 'llmstudio-proxy' package. Add it to the dependencies, or install it using 'pip install llmstudio[proxy]'."

    class LLMProxyProvider:
        def __init__(self, *args, **kwargs):
            raise ImportError(message.format(name="LLMProxyProvider"))

    class ProxyConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError(message.format(name="ProxyConfig"))


from llmstudio.providers.provider import LLM
