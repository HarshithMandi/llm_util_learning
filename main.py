from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context
from dotenv import load_dotenv
import os

load_dotenv()

llm = Ollama(model="mistral", request_timeout=180.0)
parser = LlamaParse(result_type='markdown')

# Function to determine parser based on file extension
def choose_parser(filename):
    # Define parsers for specific file types as needed
    if filename.endswith(".pdf"):
        return parser
    elif filename.endswith(".txt"):
        # Example: Use a different parser for .txt files if needed
        return parser
    elif filename.endswith(".csv"):
        return parser
    else:
        # Default parser if no specific parser is defined
        return parser

# Dictionary mapping file extensions to parsers
file_extractor = {os.path.splitext(file)[1]: choose_parser(file) for file in os.listdir("./data")}

# Load all files from directory with respective parsers
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

tools = [
    QueryEngineTool(
        query_engine=query_engine, 
        metadata=ToolMetadata(
            name="general_file",
            description="Use this tool to gather insights from various file types"
        ),
    )
]

code_llm = Ollama(model="codellama", request_timeout=180.0)
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit):")) != "q":
    result = agent.query(prompt)
    print(result)
