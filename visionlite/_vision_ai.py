from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import List
import time

from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from parselite import parse
from searchlite import google
from visionlite import vision
from wordllama import WordLlama


class SearchGen:
    """A class to generate search queries using ChatOllama"""

    def __init__(self, model: str = "llama3.2:1b-instruct-q4_K_M",
                 temperature: float = 0.1,
                 max_retries: int = 3, base_url="http://localhost:11434"):
        """
        Initialize QueryGenerator.

        Args:
            model_name: Name of the Ollama model to use
            max_retries: Maximum number of retry attempts
        """
        self.chat_model = ChatOllama(model=model, temperature=temperature,
                                     base_url=base_url)
        self.max_retries = max_retries

    def _get_system_prompt(self, n: int) -> str:
        """Get the system prompt for query generation"""
        return f'''Generate exactly {n} simple Google search queries based on the user query.
            Format the output as follows:

            Examples:

            <query>what is capital of france</query>
            1. capital of france
            2. paris france capital
            3. france capital city
            4. what is france capital city
            5. capital city of france

            <query>how to sum two arrays in python</query>
            1. python sum two arrays
            2. how to add arrays python
            3. python array addition code
            4. combine two arrays python
            5. python merge arrays tutorial'''

    def _parse_response(self, response: str, n: int) -> List[str]:
        """Parse the response into a list of queries"""
        queries = []
        for line in response.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, n + 1)):
                query = line.split('.', 1)[1].strip()
                queries.append(query.strip('"'))
        return queries

    def __call__(self, query: str, n: int = 5) -> List[str]:
        """
        Generate search queries with retry mechanism.

        Args:
            query: Input query string
            n: Number of queries to generate (default: 5)

        Returns:
            List of generated queries

        Raises:
            Exception: If generation fails after max retries
        """
        messages = [
            SystemMessage(content=self._get_system_prompt(n)),
            HumanMessage(content=f'<query>{query}</query>')
        ]

        retry_count = 0
        backoff_time = 1

        while retry_count < self.max_retries:
            try:
                if retry_count > 0:
                    print(f"\nRetry attempt {retry_count} of {self.max_retries}...")
                    time.sleep(backoff_time)
                    backoff_time *= 2

                response = self.chat_model.invoke(messages)
                queries = self._parse_response(response.content, n)

                if len(queries) == n:
                    if retry_count > 0:
                        print("Retry successful!")
                    return queries

                raise ValueError(f"Expected {n} queries, got {len(queries)}")

            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    print(f"\nFailed after {self.max_retries} attempts. Last error: {str(e)}")
                    return []
                print(f"Error occurred: {str(e)}")

_loaded_llm = None
def get_llm():
    global _loaded_llm
    if _loaded_llm is None:
        _loaded_llm = WordLlama.load()
    return _loaded_llm


def get_topk(llm=None, query=None, contents=None, k=None):
    try:
        return llm.topk(query, llm.split("".join(contents)), k=k)
    except:
        return []
def vision(query, k=1, max_urls=5, animation=False,
           allow_pdf_extraction=True,
           allow_youtube_urls_extraction=False,
           embed_model=None,genai_query_k=None):
    try:
        urls = google(query, max_urls=max_urls, animation=animation)
        contents = parse(urls, allow_pdf_extraction=allow_pdf_extraction,
                         allow_youtube_urls_extraction=allow_youtube_urls_extraction)
        llm = embed_model or get_llm()
        if genai_query_k is None:
            res = get_topk(
                llm=llm,
                query=query,
                contents=contents,
                k=k
            )
        else:
            queries = SearchGen()(query,genai_query_k)
            vars = []
            for query in queries:
                ans = get_topk(
                llm=llm,
                query=query,
                contents=contents,
                k=k
            )
                vars.extend(ans)
            res = list(set(vars))

        updated_res = "\n".join(res) + "\n\nURLS:\n" + "\n".join(urls)
    except Exception as e:
        return f"Error: {str(e)}"
    return updated_res


def visionai(query,
             max_urls=10,
             k=3,
             model="llama3.2:1b-instruct-q4_K_M",
             base_url="http://localhost:11434",
             temperature=0.0,
             max_retries=5,
             animation=False,
             allow_pdf_extraction=True,
             allow_youtube_urls_extraction=False,
             embed_model=None,
             genai_query_k: int | None = 5,
             query_k: int | None = 10,
             return_type="str"):
    gen_queries = SearchGen(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        base_url=base_url
    )
    queries = gen_queries(query, query_k)
    if not queries:
        return ["Failed to generate search queries"] if return_type == "list" else "Failed to generate search queries"

    vision_with_args = partial(
        vision,
        k=k,
        max_urls=max_urls,
        animation=animation,
        allow_pdf_extraction=allow_pdf_extraction,
        allow_youtube_urls_extraction=allow_youtube_urls_extraction,
        embed_model=embed_model,
        genai_query_k=genai_query_k
    )

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(vision_with_args, queries))
        results = list(set(results))

    if return_type == "list":
        return results
    else:
        result_str = "#####".join(results)
        return result_str


def minivisionai(query):
    return visionai(query, max_urls=5, k=2, temperature=0.05, max_retries=3, animation=False,
                    genai_query_k=3, query_k=5)


def deepvisionai(query):
    return visionai(query, max_urls=15, k=10, temperature=0.05, max_retries=10,
                    allow_pdf_extraction=True, genai_query_k=7, query_k=15)


if __name__ == "__main__":
    res = visionai('what are the new features introduced in latest crewai framework',animation=True)
    Path("res.txt").write_text(res)


