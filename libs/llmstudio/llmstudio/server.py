_proxy_server_started = False
_tracker_server_started = False


def start_servers(proxy=True, tracker=True):
    global _proxy_server_started, _tracker_server_started
    if proxy and not _proxy_server_started:
        from llmstudio_proxy.server import setup_engine_server

        setup_engine_server()
    if tracker and not _tracker_server_started:
        from llmstudio_tracker.server import setup_tracking_server

        setup_tracking_server()
