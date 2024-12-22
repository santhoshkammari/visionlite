from parselite import parse, FastParser
from searchlite import google, bing
from wordllama import WordLlama

_loaded_llm = None


def get_llm():
    global _loaded_llm
    if _loaded_llm is None:
        _loaded_llm = WordLlama.load()
    return _loaded_llm


def get_urls(name='google', query=None, max_urls=None, animation=None):
    _func = google if name == 'google' else bing
    return _func(query, max_urls=max_urls, animation=animation)

def vision(query, k=1, max_urls=5, animation=False,
           allow_pdf_extraction=True,
           allow_youtube_urls_extraction=False,
           embed_model=None,
           return_only_urls = False,
           return_with_urls = False):
    try:
        urls = get_urls(
            name='google',
            query=query,
            max_urls=max_urls,
            animation=animation
        )
        if return_only_urls:
            return urls
        contents = parse(urls, allow_pdf_extraction=allow_pdf_extraction,
                         allow_youtube_urls_extraction=allow_youtube_urls_extraction)
        contents = [_.content for _ in contents]
        llm = embed_model or get_llm()
        res = llm.topk(query, llm.split("".join(contents)), k=k)
        if return_with_urls:
            return res, urls
        return res
    except Exception as e:
        return f"Error: {str(e)}"


def visionbing(query, k=1, max_urls=5, animation=False,
               allow_pdf_extraction=True,
               allow_youtube_urls_extraction=False,
               embed_model=None,
               return_only_urls=False,
               return_with_urls=False
               ):
    try:
        urls = get_urls(
            name='bing',
            query=query,
            max_urls=max_urls,
            animation=animation
        )
        if return_only_urls:
            return urls
        contents = parse(urls, allow_pdf_extraction=allow_pdf_extraction,
                         allow_youtube_urls_extraction=allow_youtube_urls_extraction)
        contents = [_.content for _ in contents]
        llm = embed_model or get_llm()
        res = llm.topk(query, llm.split("".join(contents)), k=k)
        if return_with_urls:
            return res, urls
        return res
    except Exception as e:
        return f"Error: {str(e)}"

