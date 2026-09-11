import os
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema import Document

load_dotenv()

class TokenCounter:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
    
    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
    
    def print_usage(self):
        print(f"\n{'='*50}")
        print(f"Token Usage:")
        print(f"  Input Tokens:  {self.input_tokens}")
        print(f"  Output Tokens: {self.output_tokens}")
        print(f"  Total Tokens:  {self.total_tokens}")
        print(f"{'='*50}\n")

def format_time(seconds):
    """Format seconds into readable time"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"

def get_vectorstore_path(pdf_path, chunk_size):
    """Generate vectorstore path for a single PDF"""
    pdf_name = Path(pdf_path).stem
    vectorstore_name = f"{pdf_name}_vectorstore_chunk{chunk_size}"
    return vectorstore_name

def get_combined_vectorstore_path(pdf_paths, chunk_size):
    """Generate combined vectorstore path based on PDF filenames and chunk size"""
    pdf_names = [Path(pdf).stem for pdf in pdf_paths]
    combined_name = "_".join(sorted(pdf_names))
    vectorstore_name = f"{combined_name}_vectorstore_chunk{chunk_size}"
    return vectorstore_name

def load_pdf(file_path):
    """Load PDF and extract text with metadata"""
    start_time = time.time()
    
    pdf_filename = Path(file_path).name
    print(f"Loading PDF: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    reader = PdfReader(file_path)
    documents = []
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_filename,
                    "page": page_num,
                    "pdf_path": file_path
                }
            )
            documents.append(doc)
    
    elapsed = time.time() - start_time
    total_text = sum(len(doc.page_content) for doc in documents)
    
    print(f"PDF loaded successfully in {format_time(elapsed)}")
    print(f"Pages: {len(documents)}")
    print(f"Extracted text: {total_text:,} characters ({total_text/1024:.2f} KB)")
    print(f"Estimated tokens: ~{total_text//4:,}")
    return documents

def create_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks while preserving metadata"""
    start_time = time.time()
    
    print(f"Creating text chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    elapsed = time.time() - start_time
    print(f"Created {len(chunks)} chunks in {format_time(elapsed)}")
    
    # Count chunks per PDF
    chunks_per_pdf = {}
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        chunks_per_pdf[source] = chunks_per_pdf.get(source, 0) + 1
    
    print(f"Chunks per PDF:")
    for pdf, count in chunks_per_pdf.items():
        print(f"  - {pdf}: {count} chunks")
    
    print(f"⚠️  COST ESTIMATE:")
    print(f"   Embedding tokens: ~{len(chunks) * 250:,}")
    print(f"   Estimated cost: ${len(chunks) * 250 * 0.00013 / 1000:.4f}")
    print(f"   Proceed? Press Enter or Ctrl+C to cancel...")
    input()
    return chunks

def create_vectorstore(chunks):
    """Create vector store with embeddings"""
    start_time = time.time()
    
    print(f"Creating embeddings using OpenAI...")
    print(f"This will use approximately {len(chunks) * 250} tokens")
    embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    elapsed = time.time() - start_time
    print(f"Vector store created successfully in {format_time(elapsed)}")
    return vectorstore

def load_or_create_single_vectorstore(pdf_path, chunk_size, chunk_overlap, embeddings):
    """Load or create vectorstore for a single PDF"""
    vectorstore_path = get_vectorstore_path(pdf_path, chunk_size)
    pdf_name = Path(pdf_path).name
    
    if os.path.exists(vectorstore_path):
        print(f"✓ Found existing vectorstore for {pdf_name}: {vectorstore_path}")
        print(f"  Loading from disk (no tokens used)...")
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore, False  # False = not newly created
    else:
        print(f"✗ No vectorstore found for {pdf_name}")
        print(f"  Creating new vectorstore...")
        docs = load_pdf(pdf_path)
        chunks = create_chunks(docs, chunk_size, chunk_overlap)
        vectorstore = create_vectorstore(chunks)
        
        # Save the individual vectorstore
        print(f"  Saving vectorstore to: {vectorstore_path}")
        vectorstore.save_local(vectorstore_path)
        print(f"  ✓ Saved!")
        
        return vectorstore, True  # True = newly created

def get_or_create_combined_vectorstore(pdf_paths, combined_path, chunk_size, chunk_overlap):
    """Load existing combined vectorstore or create by merging individual ones"""
    total_start = time.time()
    embeddings = OpenAIEmbeddings()
    
    # Check if combined vectorstore already exists
    if os.path.exists(combined_path):
        print(f"\n{'='*50}")
        print("Found existing COMBINED vectorstore!")
        print(f"Location: {combined_path}")
        print(f"{'='*50}")
        
        load_start = time.time()
        print("Loading from disk (no tokens used)...")
        
        vectorstore = FAISS.load_local(
            combined_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        load_time = time.time() - load_start
        total_time = time.time() - total_start
        
        print(f"✓ Vectorstore loaded in {format_time(load_time)}")
        print(f"✓ Total setup time: {format_time(total_time)}")
        return vectorstore
    
    # Combined vectorstore doesn't exist, create by merging individual ones
    print(f"\n{'='*50}")
    print("No combined vectorstore found.")
    print("Checking for individual vectorstores to merge...")
    print(f"{'='*50}\n")
    
    vectorstores = []
    any_newly_created = False
    
    for pdf_path in pdf_paths:
        print(f"\n--- Processing {Path(pdf_path).name} ---")
        vs, is_new = load_or_create_single_vectorstore(pdf_path, chunk_size, chunk_overlap, embeddings)
        vectorstores.append(vs)
        if is_new:
            any_newly_created = True
    
    # Merge vectorstores
    print(f"\n{'='*50}")
    print("Merging vectorstores...")
    print(f"{'='*50}")
    
    merge_start = time.time()
    combined_vectorstore = vectorstores[0]
    
    for vs in vectorstores[1:]:
        combined_vectorstore.merge_from(vs)
    
    merge_time = time.time() - merge_start
    print(f"✓ Merged in {format_time(merge_time)}")
    
    # Save combined vectorstore
    save_start = time.time()
    print(f"\nSaving combined vectorstore to: {combined_path}")
    combined_vectorstore.save_local(combined_path)
    save_time = time.time() - save_start
    
    total_time = time.time() - total_start
    
    print(f"✓ Combined vectorstore saved in {format_time(save_time)}")
    print(f"✓ Total setup time: {format_time(total_time)}")
    if not any_newly_created:
        print(f"✓ No new embeddings created - reused existing vectorstores!")
    
    return combined_vectorstore

def setup_qa_chain(vectorstore, k=2):
    """Setup QA chain with LLM"""
    start_time = time.time()
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    
    # Create retriever that returns scores
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": 0.0  # Return all results with scores
        }
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    elapsed = time.time() - start_time
    print(f"QA chain initialized in {format_time(elapsed)} (k={k})")
    
    return qa_chain

def highlight_query_terms(text, query):
    """Show which parts of text match query terms (case-insensitive)"""
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why', 'who'}
    query_words = [w.lower().strip('?.,!') for w in query.split() if w.lower() not in stop_words and len(w) > 2]
    
    if not query_words:
        return text
    
    text_lower = text.lower()
    matches = []
    for word in query_words:
        if word in text_lower:
            matches.append(word)
    
    if matches:
        return f"{text}\n   → Matches: {', '.join(set(matches))}"
    return text

def display_source_chunks(source_documents, query, vectorstore):
    """Display retrieved chunks with similarity scores and source PDF"""
    print("\n" + "="*70)
    print("📚 SOURCE CHUNKS RETRIEVED")
    print("="*70)
    
    if not source_documents:
        print("No source documents found.")
        return
    
    # Get similarity scores by re-querying with scores
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=len(source_documents))
        score_map = {doc.page_content: score for doc, score in docs_with_scores}
    except Exception as e:
        print(f"Note: Could not retrieve similarity scores ({e})")
        score_map = {}
    
    for i, doc in enumerate(source_documents, 1):
        print(f"\n{'─'*70}")
        print(f"CHUNK {i}:")
        
        # Display source PDF prominently
        source_pdf = doc.metadata.get('source', 'Unknown')
        page_num = doc.metadata.get('page', 'Unknown')
        print(f"📄 SOURCE PDF: {source_pdf} (Page {page_num})")
        
        # Display similarity score if available
        score = score_map.get(doc.page_content)
        if score is not None:
            print(f"Similarity Score: {score:.4f} (lower = better match)")
        
        # Display chunk content
        print(f"\nContent ({len(doc.page_content)} chars):")
        print("─" * 70)
        
        # Show first 400 chars of chunk
        preview = doc.page_content[:550]
        if len(doc.page_content) > 550:
            preview += "..."
        
        # Highlight matching terms
        highlighted = highlight_query_terms(preview, query)
        print(highlighted)
    
    print("\n" + "="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Multi-PDF RAG System with Vectorstore Reuse")
    parser.add_argument(
        "--pdf1",
        type=str,
        required=True,
        help="Path to first PDF file"
    )
    parser.add_argument(
        "--pdf2",
        type=str,
        required=True,
        help="Path to second PDF file"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text splitting (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for text splitting (default: 200)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild combined vectorstore (but reuse individual ones)"
    )
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"Multi-PDF RAG System - Using OpenAI")
    print(f"{'='*50}\n")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file")
        return
    
    # Check if PDFs exist
    pdf_paths = [args.pdf1, args.pdf2]
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Error: {pdf_path} not found")
            return
    
    # Generate paths
    combined_path = get_combined_vectorstore_path(pdf_paths, args.chunk_size)
    
    print(f"PDF files:")
    for i, pdf in enumerate(pdf_paths, 1):
        individual_vs = get_vectorstore_path(pdf, args.chunk_size)
        exists = "✓" if os.path.exists(individual_vs) else "✗"
        print(f"  {i}. {pdf} {exists}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Combined vectorstore: {combined_path}/")
    print(f"Retrieval: k=2 chunks per query")
    print()
    
    # Force rebuild if requested
    if args.rebuild and os.path.exists(combined_path):
        print(f"Removing existing combined vectorstore (--rebuild flag used)...")
        import shutil
        shutil.rmtree(combined_path)
    
    # Load or create combined vectorstore
    vectorstore = get_or_create_combined_vectorstore(pdf_paths, combined_path, args.chunk_size, args.chunk_overlap)
    qa_chain = setup_qa_chain(vectorstore, k=2)
    
    print("\n" + "="*50)
    print("RAG system ready! Type your questions (or 'quit' to exit)")
    print("="*50 + "\n")
    
    token_counter = TokenCounter()
    
    # Query loop
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        token_counter.reset()
        query_start = time.time()
        
        print(f"\n[Processing query...]")
        
        try:
            with get_openai_callback() as cb:
                result = qa_chain({"query": query})
                token_counter.input_tokens = cb.prompt_tokens
                token_counter.output_tokens = cb.completion_tokens
                token_counter.total_tokens = cb.total_tokens
            
            query_time = time.time() - query_start
            
            print("\nAnswer:")
            print("-" * 50)
            print(result['result'])
            print("-" * 50)
            print(f"Response time: {format_time(query_time)}")
            
            # Display source chunks with scores and PDF sources
            if 'source_documents' in result and result['source_documents']:
                display_source_chunks(result['source_documents'], query, vectorstore)
            
            token_counter.print_usage()
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print()

if __name__ == "__main__":
    main()