from rich.console import Console
console = Console()
from langchain_openai import OpenAIEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configuration import EMBEDDING_MODEL,QDRANT_TIMEOUT,COLLECTION_NAME,MAX_CONTEXT_CHUNKS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st


def initialize_clients_for_indexing():
    """Initialize Qdrant client and embedding model with caching"""
    try:
        qdrant_host = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate environment variables
        missing_vars = []
        if not qdrant_host:
            missing_vars.append("QDRANT_URL")
        if not qdrant_api_key:
            missing_vars.append("QDRANT_API_KEY")
        if not openai_api_key:
            missing_vars.append("OPENAI_API_KEY")
        
        if missing_vars:
            console.print(f"🔴 [bold red]Missing environment variables: {', '.join(missing_vars)} [/bold red]")
            console.print("[bold orange]Please check your .env file or environment configuration[/bold orange]")
        
   
        
        
        
        # Initialize OpenAI embeddings
        try:
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            # Test embedding creation
            test_embedding = embedding_model.embed_query("test")
            if len(test_embedding) > 0:
                 console.print("✅ [bold green] OpenAI embeddings initialized successfully! [/bold green]")
        except Exception as e:
            console.print(f"❌ [bold red]OpenAI embeddings initialization failed: {str(e)}[/bold red]")
            console.print("[bold orange]Please check your OPENAI_API_KEY[/bold orange]")   

        # initialize text splitter
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        except Exception as e:
            console.print(f"❌ [bold red]Something went wrong initializing text splitter: {str(e)}[/bold red]")  

        return  embedding_model, text_splitter,qdrant_host,qdrant_api_key,console
        
    except Exception as e:
        console.print(f"❌ [bold red]Failed to initialize clients: {str(e)}[/bold red]")


def get_chat_response(query,  vector_store):
    """Generate chat response based on query and context"""
    try:
        # Retrieve relevant documents
        results = vector_store.similarity_search(query=query, k=MAX_CONTEXT_CHUNKS)
        
        if not results:
            return "I couldn't find relevant information in the document to answer your question.", []
        
        # Prepare context
        context_parts = []
        for i, res in enumerate(results, 1):
           
            resource_link = res.metadata.get('source')
            title = res.metadata.get('title')
            description = res.metadata.get('description')
            context_parts.append(f"**Chunk {i} (Title {title}):**\n Description {description}):**\n Source {resource_link}):**\n{res.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced system prompt
        SYSTEM_PROMPT = f"""You are a helpful AI assistant that answers questions based on Chai Code docs.

IMPORTANT INSTRUCTIONS:
1. FIRST, check if the answer exists in the provided context below
2. If the answer IS in the context: Based on the context and query Understand their relation, verify if the context is really solution to the query and  come up with the best creative, detailed and easy to understand response and include Title and Source at the end of the response for references
3. If you feel the context doesnot fully answer the query give the response based on your understanding of the query.
4. If the answer is NOT in the context: Clearly state that the information is not in the document, [In this case response should always start the response with "Sorry, information for your query is not in the ChaiCode document"]
5. If the query is regarding programming always try giving an example for better understanding.
6. If the query is regarding you or your training Give a creative and witty answer 
[ Remember: 
-Your name is  Chai Docs Bot
-You are trained by SaiCharan B M
-You are trained on Chaicode docs
-Don't give any other information about you please.] 


Guidelines:
- Be concise but comprehensive for document-based answers
- For out-of-context answers: Keep it to ONE SENTENCE only
- If the query is about programming, always provide examples (for document-based answers)
- If you're uncertain about your general knowledge, acknowledge it

Context from the document:
{context}

Example of query present in context:
Example 1:
user: How to write function in c++?
Assistant: In C++, a function is a block of code designed to perform a specific task, and it follows a basic syntax: `returnType functionName(parameters) {{ // function body }}`.

Here is an example:

```cpp
#include <iostream>
using namespace std;ß

// Function to check tea temperature
int checkTeaTemperature(int temperature) {{
    return temperature;
}}

int main() {{
    int temp = checkTeaTemperature(85);  // Function call
    cout << "The tea temperature is " << temp << "°C" << endl;
    return 0;
}}
```

In this example:
- `int` is the return type, specifying that the function will return an integer value. If no value is returned, you can use `void`.
- `checkTeaTemperature` is the name of the function, which describes what the function does.
- `(int temperature)` are the parameters, which are input values for the function. Parameters are optional.
[Functions | Chai aur Docs](https://docs.chaicode.com/youtube/chai-aur-c/functions/)

Example 2:
User:How can i create a form in HTML?
Assistant: You can create a form in HTML using the `<form>` tag, which acts as a container for various input elements like text fields, buttons, and drop-down lists. Here’s how you can do it:

```html
<form>
  <input type="text" placeholder="Enter your name">
  <button>Click me</button>
</form>
```

In the code above:
-  `<form>`: This tag defines the form container.
-  `<input>`: This tag creates an input field for text. The `type` attribute specifies the type of input (e.g., "text").
-  `<button>`: This tag creates a clickable button.

Here are some other tags which can be used:
- `<textarea>` – Multi-line text input
- `<select>` – Drop-down list
- `<option>` – Option within a drop-down

[Common HTML Tags | Chai aur Docs](https://docs.chaicode.com/youtube/chai-aur-html/html-tags/)
"""

        
        # Generate response
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            timeout=30,
            max_retries=2,
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = llm.invoke(messages)

        print(response.content)
        
        return response.content
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while processing your question.", []
