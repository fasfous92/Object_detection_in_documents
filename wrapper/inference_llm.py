from litellm import completion
import base64
from wrapper.model_llm import LLM

class InferenceLLM(LLM):
    def __init__(self, model_id: str, api_key: str = None, api_base: str = None):
        super().__init__(model_id)
        self.api_key = api_key
        self.api_base = api_base # Used for local servers like Ollama/vLLM

    def load(self):
        # APIs don't strictly "load", but we can ping to check connection
        print(f"🌐 Connected to API Model: {self.model_id}")

    def predict(self, image_path: str, prompt: str) -> str:
        # Encode image to Base64
        with open(image_path, "rb") as img:
            base64_img = base64.b64encode(img.read()).decode('utf-8')
        
        full_prompt = f"{self.system_prompt}\nTask: {prompt}"
        
        try:
            response = completion(
                model=self.model_id,
                api_key=self.api_key,
                api_base=self.api_base,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
