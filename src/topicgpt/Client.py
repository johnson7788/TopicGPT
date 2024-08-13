class Client:
    # 初始化llm client
    def __init__(self, api_key: str, base_url:str, azure_endpoint: dict = None, http_client=None ) -> None:
        if azure_endpoint:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(api_key=api_key, api_version=azure_endpoint['api_version'], azure_endpoint=azure_endpoint['endpoint'])
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key,base_url=base_url,http_client=http_client)
    
    def __getattr__(self, name):
        """Delegate attribute access to the self.client object."""
        return getattr(self.client, name)
