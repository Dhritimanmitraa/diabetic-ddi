import ollama
import asyncio

async def test():
    try:
        client = ollama.AsyncClient()
        print("Testing Ollama connection...")
        
        # Test listing models
        models = await asyncio.wait_for(client.list(), timeout=5.0)
        print(f"Raw models response: {models}")
        
        # Handle different response formats
        if isinstance(models, dict) and 'models' in models:
            model_names = [m.get('name', m.get('model', 'unknown')) for m in models.get('models', [])]
        elif hasattr(models, 'models'):
            model_names = [m.name if hasattr(m, 'name') else str(m) for m in models.models]
        else:
            model_names = str(models)
        print(f"Available models: {model_names}")
        
        # Test a simple chat
        print("\nTesting chat... (this may take up to 60 seconds for 8B model)")
        result = await asyncio.wait_for(
            client.chat(
                model='llama3.1:8b',
                messages=[{'role': 'user', 'content': 'Say hello in one word. Respond with just the word, nothing else.'}],
                options={'temperature': 0.2}
            ),
            timeout=60.0
        )
        print(f"Raw result type: {type(result)}")
        print(f"Raw result: {result}")
        
        # Handle different response formats
        if isinstance(result, dict):
            content = result.get('message', {}).get('content', 'No content')
        elif hasattr(result, 'message'):
            content = result.message.content if hasattr(result.message, 'content') else str(result.message)
        else:
            content = str(result)
        
        print(f"Response: {content}")
        print("\n✅ Ollama is working correctly!")
    except asyncio.TimeoutError:
        print("❌ Timeout - Ollama took too long to respond")
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
