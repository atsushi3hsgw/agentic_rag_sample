import os
import argparse
import logging
import sys
from dotenv import load_dotenv

from agentic_rag_sample.rag import AgenticRAG

def setup_logging(log_level: str):
    """Set up logging configuration"""
    level = getattr(logging, log_level.upper(), logging.DEBUG)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("agentic_rag_cli.log"),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Agentic RAG CLI')
    parser.add_argument('question', type=str, nargs='?', help='User question to answer')
    parser.add_argument('--score_threshold', type=float, 
                        help='Override environment SCORE_THRESHOLD')
    parser.add_argument('--k', type=int, 
                        help='Override environment K value')
    parser.add_argument('--web_k', type=int, 
                        help='Override environment WEB_K value')
    parser.add_argument('--log_level', type=str, 
                        help='Override LOG_LEVEL (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, 
                        default=True, help='Enable verbose output (default: True)')
    parser.add_argument('--dump_graph', action='store_true',
                        help='Output LangGraph structure in Mermaid format and exit without processing')
    return parser.parse_args()

def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Parse CLI arguments
        args = parse_arguments()
        
        # Configure logging
        log_level = args.log_level or os.getenv('LOG_LEVEL', 'INFO')
        setup_logging(log_level)
        
        # Validate required environment variables
        required_env_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'TAVILY_API_KEY']
        for var in required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"Missing required environment variable: {var}")

        # Build configuration dictionary
        config = {
            'llm_model': os.getenv('LLM_MODEL', 'tngtech/deepseek-r1t2-chimera:free'),
            'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'agentic_rag_data'),
            'base_url': os.getenv('BASE_URL', 'https://openrouter.ai/api/v1'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'openai/text-embedding-3-small'),
            'score_threshold': float(args.score_threshold or os.getenv('SCORE_THRESHOLD', 0.3)),
            'k': args.k or int(os.getenv('K', 5)),
            'web_k': args.web_k or int(os.getenv('WEB_K', 3)),
            'verbose_output': args.verbose
        }

        # Initialize RAG system
        logging.info("Initializing AgenticRAG system")
        rag = AgenticRAG(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            tavily_api_key=os.getenv('TAVILY_API_KEY'),
            **config
        )
        
        if args.dump_graph:
            print("# Mermaid Flowchart Syntax\n")
            print(rag.graph.get_graph().draw_mermaid())
            exit(0)
        
        if not args.question:
            raise ValueError("Question argument is required when not using --dump_graph")
        
        # Execute query
        logging.info(f"Processing question: {args.question}")
        result = rag.query(args.question)
        
        # Output result
        print("\nAnswer:")
        print(result['answer'])
        if 'sources' in result:
            print("\nSources:")
            for source in result['sources']:
                print(f"- {source}")
                
    except Exception as e:
        logging.error(f"Operation failed: {str(e)}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
