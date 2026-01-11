# LangChain Introduction - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Setup & Configuration](#setup--configuration)
3. [OpenAI Models](#openai-models)
4. [Google Gemini Models](#google-gemini-models)
5. [GROQ Models](#groq-models)
6. [Streaming Output](#streaming-output)
7. [Batch Processing](#batch-processing)
8. [Key Concepts](#key-concepts)

---

## Introduction

**LangChain** ek powerful framework hai jo Large Language Models (LLMs) ke saath applications develop karne ke liye design kiya gaya hai. Ye framework aapko different AI models ko easily integrate karne ki facility deta hai.

### LangChain Version
Is tutorial mein hum **LangChain v1.2.3** use kar rahe hain, jo latest features aur improvements ke saath aata hai.

### Why LangChain?
- **Multi-Model Support**: OpenAI, Google, Anthropic, GROQ jaise multiple providers ko support karta hai
- **Unified Interface**: Sabhi models ke liye consistent API
- **Advanced Features**: Streaming, batching, chains, agents, aur bohot kuch
- **Easy Integration**: Minimal code mein powerful AI applications

---

## Setup & Configuration

### Environment Variables Setup

Pehle hum apne API keys ko securely load karte hain using `python-dotenv`:

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
```

**Explanation:**
- `load_dotenv()`: `.env` file se environment variables load karta hai
- `os.environ[]`: Environment variables ko set karta hai jo LangChain automatically detect kar leta hai
- **Security**: API keys ko code mein hardcode karne ki bajaye `.env` file mein store karna secure practice hai

### Required Dependencies
```bash
pip install langchain
pip install langchain-openai
pip install langchain-google-genai
pip install langchain-groq
pip install python-dotenv
```

---

## OpenAI Models

### Method 1: Using `init_chat_model()` (Recommended)

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini")
response = model.invoke("Write me an essay about the benefits of using generative AI")
```

**Key Features:**
- **Automatic Provider Detection**: Model name se automatically provider detect hota hai
- **Simplified Initialization**: Ek line mein model ready
- **Flexibility**: Different models easily switch kar sakte hain

**Response Structure:**
```python
AIMessage(
    content="...",  # Actual text response
    additional_kwargs={'refusal': None},
    response_metadata={
        'token_usage': {
            'completion_tokens': 819,
            'prompt_tokens': 19,
            'total_tokens': 838
        },
        'model_name': 'gpt-4o-mini-2024-07-18',
        'finish_reason': 'stop'
    },
    usage_metadata={...}
)
```

### Method 2: Using Direct Import

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5")
response = model.invoke("Hello, how are you?")
```

**When to Use:**
- Jab aapko specific provider ki advanced settings chahiye
- Custom configurations ke liye (temperature, max_tokens, etc.)

### Accessing Response Content

```python
# Full response object
print(response)

# Only text content
print(response.content)
```

---

## Google Gemini Models

### Using Google's Generative AI

```python
from langchain.chat_models import init_chat_model

# Method 1: Using init_chat_model
model = init_chat_model(model="google_genai:gemini-flash-latest")
response = model.invoke("write me a poem about the benefits of using generative AI")
```

**Google Models Available:**
- `gemini-2.0-flash-exp`: Latest experimental model
- `gemini-flash-latest`: Fast, efficient model
- `gemini-2.5-flash-preview`: Preview version with advanced features

### Direct Import Method

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")
response = model.invoke("Your prompt here")
```

**Important Notes:**
- Google models ka quota limit hota hai
- Free tier mein rate limiting ho sakti hai
- Production mein proper API key aur billing setup zaruri hai

### Response Structure (Google)
```python
AIMessage(
    content=[{
        'type': 'text',
        'text': '...',
        'extras': {'signature': '...'}
    }],
    response_metadata={
        'finish_reason': 'STOP',
        'model_name': 'gemini-2.5-flash-preview',
        'safety_ratings': []
    }
)
```

---

## GROQ Models

**GROQ** ek high-performance inference engine hai jo extremely fast responses deta hai.

### Using GROQ with LangChain

```python
from langchain.chat_models import init_chat_model

# Method 1: Unified approach
model = init_chat_model("groq:llama-3.1-8b-instant")
response = model.invoke("write me a poem about the benefits of using generative AI")
```

### Direct GROQ Import

```python
from langchain_groq import ChatGroq

model = ChatGroq(model="qwen/qwen3-32b")
response = model.invoke("What is the mindset behind generative AI?")
```

**Popular GROQ Models:**
- `llama-3.1-8b-instant`: Fast, lightweight Llama model
- `qwen/qwen3-32b`: Powerful Qwen model with reasoning capabilities
- `mixtral-8x7b-32768`: Large context window model

**GROQ Advantages:**
- ⚡ **Extremely Fast**: Hardware-accelerated inference
- 💰 **Cost-Effective**: Competitive pricing
- 🎯 **High Quality**: State-of-the-art models
- 📊 **Detailed Metadata**: Comprehensive token usage info

### Response Metadata Example
```python
response_metadata={
    'token_usage': {
        'completion_tokens': 267,
        'prompt_tokens': 47,
        'total_tokens': 314,
        'completion_time': 0.524182573,
        'prompt_time': 0.002347235
    },
    'model_name': 'llama-3.1-8b-instant',
    'finish_reason': 'stop'
}
```

---

## Streaming Output

**Streaming** se aap real-time mein response dekh sakte hain, word-by-word.

### Basic Streaming Example

```python
for chunk in model.stream("Write me a 500 word essay on the benefits of using generative AI"):
    print(chunk.text, end="", flush=True)
```

**How It Works:**
1. Model response ko chunks mein break karta hai
2. Har chunk immediately available hota hai
3. `end=""` se newlines avoid hote hain
4. `flush=True` se immediate output milta hai

### Benefits of Streaming:
- ✅ **Better UX**: User ko wait nahi karna padta
- ✅ **Real-time Feedback**: Progressive response visible
- ✅ **Perception of Speed**: Faster feel hota hai
- ✅ **Long Responses**: Bade responses ke liye ideal

### Use Cases:
- Chatbots aur conversational AI
- Content generation tools
- Interactive applications
- Live demonstrations

---

## Batch Processing

**Batch processing** se aap multiple queries ko ek saath process kar sakte hain.

### Basic Batch Example

```python
responses = model.batch(
    [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?",
    ],
    config={
        "max_concurrency": 5,  # Maximum parallel requests
    }
)

# Access individual responses
for response in responses:
    print(response.content)
```

### Configuration Options

```python
config = {
    "max_concurrency": 5,      # Parallel requests limit
    "max_retries": 3,          # Retry failed requests
    "request_timeout": 60,     # Timeout in seconds
}
```

### Benefits of Batch Processing:
- 🚀 **Efficiency**: Multiple queries parallel mein process hote hain
- ⏱️ **Time Saving**: Total time significantly reduce hota hai
- 💰 **Cost Optimization**: Bulk processing often cheaper hai
- 📊 **Better Resource Usage**: Network aur API calls optimize hote hain

### Use Cases:
- Data analysis aur processing
- Bulk content generation
- Testing aur evaluation
- Report generation

### Example: Processing Multiple Questions

```python
questions = [
    "What is machine learning?",
    "Explain neural networks",
    "What is deep learning?",
    "Define artificial intelligence",
]

responses = model.batch(questions, config={"max_concurrency": 3})

for question, response in zip(questions, responses):
    print(f"Q: {question}")
    print(f"A: {response.content[:100]}...")  # First 100 chars
    print("-" * 50)
```

---

## Key Concepts

### 1. Model Initialization

**Three Ways to Initialize:**

```python
# Way 1: Universal init_chat_model (Recommended)
model = init_chat_model("gpt-4o-mini")
model = init_chat_model("groq:llama-3.1-8b-instant")
model = init_chat_model("google_genai:gemini-flash-latest")

# Way 2: Provider-specific imports
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")

# Way 3: With custom parameters
model = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)
```

### 2. Response Types

**AIMessage Object Structure:**
```python
response = AIMessage(
    content="...",              # Main text content
    additional_kwargs={},       # Extra metadata
    response_metadata={         # API response details
        'token_usage': {...},
        'model_name': '...',
        'finish_reason': '...'
    },
    usage_metadata={...}        # Token usage details
)
```

### 3. Token Usage Understanding

```python
{
    'completion_tokens': 819,   # AI ne generate kiye
    'prompt_tokens': 19,        # Aapne bheje
    'total_tokens': 838,        # Total used
    'completion_time': 0.524,   # Generation time (seconds)
}
```

**Cost Calculation:**
- Prompt tokens: Input ki cost
- Completion tokens: Output ki cost (usually higher)
- Total cost = (prompt_tokens × input_price) + (completion_tokens × output_price)

### 4. Model Parameters

**Common Parameters:**
```python
model = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,        # Creativity (0-1)
    max_tokens=1000,        # Max response length
    top_p=0.9,             # Nucleus sampling
    frequency_penalty=0,    # Repetition penalty
    presence_penalty=0,     # Topic diversity
    timeout=30,            # Request timeout
    max_retries=3          # Retry attempts
)
```

**Temperature Explained:**
- `0.0`: Deterministic, consistent responses
- `0.5`: Balanced creativity
- `1.0`: Maximum creativity, more random

### 5. Error Handling

```python
try:
    response = model.invoke("Your prompt")
except Exception as e:
    print(f"Error: {e}")
    # Handle specific errors
    if "quota" in str(e).lower():
        print("API quota exceeded")
    elif "timeout" in str(e).lower():
        print("Request timed out")
```

### 6. Best Practices

**✅ DO:**
- Environment variables use karein API keys ke liye
- Error handling implement karein
- Token usage monitor karein
- Appropriate model select karein task ke according
- Streaming use karein long responses ke liye

**❌ DON'T:**
- API keys ko code mein hardcode na karein
- Bina error handling ke production mein deploy na karein
- Unnecessarily expensive models use na karein
- Rate limits ignore na karein

---

## Comparison: OpenAI vs Google vs GROQ

| Feature | OpenAI | Google Gemini | GROQ |
|---------|--------|---------------|------|
| **Speed** | Fast | Fast | Very Fast ⚡ |
| **Cost** | Medium | Low-Medium | Low 💰 |
| **Quality** | Excellent | Excellent | Excellent |
| **Context Window** | Up to 128K | Up to 1M | Up to 32K |
| **Multimodal** | Yes | Yes | Limited |
| **Best For** | General purpose | Large contexts | Speed-critical apps |

---

## Practical Examples

### Example 1: Simple Chatbot

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini")

def chatbot(user_input):
    response = model.invoke(user_input)
    return response.content

# Usage
print(chatbot("Hello! How are you?"))
```

### Example 2: Content Generator with Streaming

```python
def generate_content(topic):
    prompt = f"Write a detailed article about {topic}"
    print(f"Generating content about: {topic}\n")
    
    for chunk in model.stream(prompt):
        print(chunk.text, end="", flush=True)
    print("\n")

generate_content("Artificial Intelligence")
```

### Example 3: Multi-Question Analyzer

```python
def analyze_questions(questions):
    responses = model.batch(questions, config={"max_concurrency": 3})
    
    results = []
    for q, r in zip(questions, responses):
        results.append({
            "question": q,
            "answer": r.content,
            "tokens": r.usage_metadata['total_tokens']
        })
    
    return results

questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Java?"
]

results = analyze_questions(questions)
for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer'][:100]}...")
    print(f"Tokens: {result['tokens']}\n")
```

---

## Troubleshooting

### Common Issues & Solutions

**1. API Key Error**
```
Error: Invalid API key
Solution: Check .env file aur ensure karo key correct hai
```

**2. Rate Limit Exceeded**
```
Error: Rate limit exceeded
Solution: Wait karein ya paid plan upgrade karein
```

**3. Model Not Found**
```
Error: Model not found
Solution: Model name check karein, typos avoid karein
```

**4. Timeout Error**
```
Error: Request timeout
Solution: timeout parameter increase karein ya internet check karein
```

---

## Next Steps

1. **Chains**: Multiple LLM calls ko chain karein
2. **Agents**: Decision-making capabilities add karein
3. **Memory**: Conversation history maintain karein
4. **Tools**: External tools integrate karein
5. **RAG**: Retrieval Augmented Generation implement karein

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Google AI Studio](https://makersuite.google.com/)
- [GROQ Documentation](https://console.groq.com/docs)

---

## Summary

Is tutorial mein humne seekha:
- ✅ LangChain setup aur configuration
- ✅ Multiple AI providers (OpenAI, Google, GROQ) ka usage
- ✅ Streaming responses for better UX
- ✅ Batch processing for efficiency
- ✅ Best practices aur error handling
- ✅ Practical examples aur use cases

**Remember**: Right model choose karna task, budget, aur requirements pe depend karta hai. Experiment karein aur apne use case ke liye best option find karein!

---

*Created with ❤️ for LangChain learners*
