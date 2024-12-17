from parselite import parse, FastParser
from searchlite import google, bing
from wordllama import WordLlama

_loaded_llm = None


def get_llm():
    global _loaded_llm
    if _loaded_llm is None:
        _loaded_llm = WordLlama.load()
    return _loaded_llm


def vision(query, k=1, max_urls=5, animation=False,
           allow_pdf_extraction=True,
           allow_youtube_urls_extraction=False,
           embed_model=None):
    try:
        urls = google(query, max_urls=max_urls, animation=animation)
        contents = parse(urls, allow_pdf_extraction=allow_pdf_extraction,
                         allow_youtube_urls_extraction=allow_youtube_urls_extraction)
        llm = embed_model or get_llm()
        res = llm.topk(query, llm.split("".join(contents)), k=k)
        updated_res = "\n".join(res) + "\n\nURLS:\n" + "\n".join(urls)
    except Exception as e:
        return f"Error: {str(e)}"
    return updated_res


def visionbing(query, k=1, max_urls=5, animation=False,
               allow_pdf_extraction=True,
               allow_youtube_urls_extraction=False,
               embed_model=None):
    try:
        urls = bing(query, max_urls=max_urls, animation=animation)
        contents = parse(urls, allow_pdf_extraction=allow_pdf_extraction,
                         allow_youtube_urls_extraction=allow_youtube_urls_extraction)
        llm = embed_model or get_llm()
        res = llm.topk(query, llm.split("".join(contents)), k=k)
        updated_res = "\n".join(res) + "\n\nURLS:\n" + "\n".join(urls)
    except Exception as e:
        return f"Error: {str(e)}"
    return updated_res


async def avision(query, k=1, max_urls=5, animation=False,
                  allow_pdf_extraction=True,
                  allow_youtube_urls_extraction=False,
                  embed_model=None):
    try:
        urls = google(query, max_urls=max_urls, animation=animation)
        parser = FastParser(extract_pdf=allow_pdf_extraction,
                            allow_youtube_urls_extraction=allow_youtube_urls_extraction)
        contents = await parser._async_html_parser(urls)
        llm = embed_model or get_llm()
        res = llm.topk(query, llm.split("".join(contents)), k=k)
        updated_res = "\n".join(res) + "\n\nURLS:\n" + "\n".join(urls)
    except Exception as e:
        return f"Error: {str(e)}"
    return updated_res
