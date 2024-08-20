from openai import OpenAI
class Client:
    # 初始化llm client
    def __init__(self, api_key: str, base_url:str, http_client=None ) -> None:
        self.client = OpenAI(api_key=api_key,base_url=base_url,http_client=http_client)

    def __getattr__(self, name):
        """Delegate attribute access to the self.client object."""
        return getattr(self.client, name)
