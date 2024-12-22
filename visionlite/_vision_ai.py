from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import List
import time

import pandas as pd
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from parselite import parse, FastParserResult
from searchlite import google
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


def get_parsed_result(urls=None, allow_pdf_extraction=False, allow_youtube_urls_extraction=False):
    parser_results: List[FastParserResult] = parse(urls, allow_pdf_extraction=allow_pdf_extraction,
                                                   allow_youtube_urls_extraction=allow_youtube_urls_extraction)
    parser_results_v1 = [_ for _ in parser_results if _.content]
    return parser_results_v1


@dataclass
class MultiQuerySearchResult:
    query: str
    results: List[FastParserResult]


def multi_extract_url_and_contents(query, max_urls):
    list_of_list_of_urls = google(query=query,max_urls=max_urls)
    parsed_list_of_list_of_urls = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        parsed_list_of_list_of_urls.extend(list(executor.map(get_parsed_result, list_of_list_of_urls)))

    final = [MultiQuerySearchResult(query=q, results=res) for q, res in zip(query, parsed_list_of_list_of_urls)]
    return final


def get_fastparser_result(query=None, animation=None, allow_pdf_extraction=None, allow_youtube_urls_extraction=None):
    urls: List[str] = google(query, animation=animation)
    parser_results_v1 = get_parsed_result(urls, allow_pdf_extraction, allow_youtube_urls_extraction)
    return parser_results_v1


def get_relevant_chunks(query, contents: list,
                        k=None, split_targe_size: int = 1000, llm=None):
    final_results = []
    for result in contents:
        contents = llm.split(result.content, target_size=split_targe_size)
        topk = llm.topk(query=query, candidates=contents, k=k)
        final_results.extend([{"url": result.url, "query": query,
                               "chunk": c} for c in topk])
    return final_results


def create_dataframe_from_results(multiquery_results):
    df = pd.DataFrame([{"query": x.query,
                        "url": y.url,
                        "content": y.content} for x in multiquery_results for y in x.results])
    return df


def get_topk_chunk_and_url(df):
    newdf = df.explode('top_k').reset_index(drop=True)
    df2 = newdf.groupby('top_k')['url'].first().reset_index()
    return df2




def visionai_version3(query, k=3, max_urls=10, animation=False,
                      allow_pdf_extraction=True,
                      allow_youtube_urls_extraction=False,
                      embed_model=None, genai_query_k=4, model="llama3.2:1b-instruct-q4_K_M",
                      base_url="http://localhost:11434",
                      temperature=0.1, max_retries=3,
                      return_type=None):
    llm = embed_model or get_llm()
    query_generator = SearchGen(model=model,
                                temperature=temperature,
                                max_retries=max_retries,
                                base_url=base_url)

    genai_query_variations: List = [query] + query_generator(query, genai_query_k)
    multiquery_results: List[MultiQuerySearchResult] = multi_extract_url_and_contents(query=genai_query_variations,
                                                                                      max_urls=max_urls)
    df = create_dataframe_from_results(multiquery_results)

    # First get top-k chunks for each content
    def split_func(x):
        chunks = llm.split(x)
        if len(chunks)<=k:
            return list(chunks)
        return list(llm.topk(query, candidates=chunks, k=k))

    df['top_k'] = df['content'].apply(split_func)

    # Explode the top_k column to get one chunk per row
    chunks_df = df.explode('top_k')[['url', 'top_k']].reset_index(drop=True)

    # Get similarity scores for all chunks
    chunks_df['similarity'] = chunks_df['top_k'].apply(lambda x: llm.similarity(query, x))

    # Get final top k chunks based on similarity scores
    final_chunks_df = chunks_df.nlargest(k, 'similarity')[['top_k', 'url']]

    if return_type == "list":
        return final_chunks_df.to_dict('records')

    # Format the string output
    updated_res = "\n".join(f"{row['top_k']}\n{row['url']}" for _, row in final_chunks_df.iterrows())
    return updated_res

def visionai(query,
             max_urls=5,
             k=3,
             model="llama3.2:1b-instruct-q4_K_M",
             base_url="http://localhost:11434",
             temperature=0.1,
             max_retries=5,
             animation=False,
             allow_pdf_extraction=True,
             allow_youtube_urls_extraction=False,
             embed_model=None,
             genai_query_k: int | None = 5,
             query_k: int | None = 5,
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
        visionai_version3,
        k=k,
        max_urls=max_urls,
        animation=animation,
        allow_pdf_extraction=allow_pdf_extraction,
        allow_youtube_urls_extraction=allow_youtube_urls_extraction,
        embed_model=embed_model,
        genai_query_k=genai_query_k,
        temperature=temperature,
        model=model,
        max_retries=max_retries,
        base_url=base_url
    )

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(vision_with_args, queries))
        results = list(set(results))

    if return_type == "list":
        return results
    else:
        result_str = "#####".join(results)
        return result_str


def minivisionai(query,
                 max_urls=5,
                 k=2,
                 model="llama3.2:1b-instruct-q4_K_M",
                 base_url="http://localhost:11434",
                 temperature=0.1,
                 max_retries=3,
                 animation=False,
                 allow_pdf_extraction=True,
                 allow_youtube_urls_extraction=False,
                 embed_model=None,
                 genai_query_k: int | None = 3,
                 query_k: int | None = 5,
                 return_type="str"):
    return visionai(query, max_urls=max_urls, k=k, model=model, base_url=base_url, temperature=temperature,
                    max_retries=max_retries, animation=animation, allow_pdf_extraction=allow_pdf_extraction,
                    allow_youtube_urls_extraction=allow_youtube_urls_extraction, embed_model=embed_model,
                    genai_query_k=genai_query_k, query_k=query_k, return_type=return_type)


def deepvisionai(query,
                 max_urls=15,
                 k=10,
                 model="llama3.2:1b-instruct-q4_K_M",
                 base_url="http://localhost:11434",
                 temperature=0.05,
                 max_retries=10,
                 animation=False,
                 allow_pdf_extraction=True,
                 allow_youtube_urls_extraction=False,
                 embed_model=None,
                 genai_query_k: int | None = 7,
                 query_k: int | None = 15,
                 return_type="str"):
    return visionai(query, max_urls=max_urls, k=k, model=model, base_url=base_url, temperature=temperature,
                    max_retries=max_retries, animation=animation, allow_pdf_extraction=allow_pdf_extraction,
                    allow_youtube_urls_extraction=allow_youtube_urls_extraction, embed_model=embed_model,
                    genai_query_k=genai_query_k, query_k=query_k, return_type=return_type)


def main():
    queries = [
        # 'what are the new features introduced in latest crewai framework',
        # 'how does the crewai framework improve performance',
        # 'can you explain the architecture of crewai framework',
        # 'what are the use cases for crewai framework',
        # 'how does crewai framework handle data privacy',
        # 'what are the system requirements for crewai framework',
        # 'can crewai framework be integrated with other tools',
        # 'what are the licensing terms for crewai framework',
        # 'how can I get support for crewai framework',
        'are there any tutorials or documentation available for crewai framework'
    ]

    for query in queries:
        r = visionai(query,base_url="http://192.168.170.76:11434")
        print(r)


if __name__ == "__main__":
    main()
