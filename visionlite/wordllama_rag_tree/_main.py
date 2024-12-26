from parselite import parse
from searchlite import google

from ._retriever import WordLlamaRetreiverTree
from ._sentvar import SentenceVariator

def wordllama_qa(query: str, max_depth: int = 3, max_tot_score: float = 1,
                 k=3):
    sv = SentenceVariator()
    qvs = sv.gen(query)
    print('question:')
    for q in qvs:
        print(q)
    ans = ""
    for q in qvs:
        try:
            tr = WordLlamaRetreiverTree("\n".join([_.content for _ in parse(google(query))]))
            res = tr.retrieve(
                query=q,
                max_depth=max_depth,
                max_tot_score=max_tot_score,
                k=k
            )
            ans += f"\n### QUERY\n{q}\n\n### RESULTS\n{res}\n"
        except:
            pass
    return ans

if __name__ == '__main__':
    # query="sangai's max life span"
    query="how to run gguf model using transformers library"
    wordllama_qa(query)

