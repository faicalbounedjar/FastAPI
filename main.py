import torch
from fastapi import FastAPI, Request
import uvicorn
#from scipy.spatial.distance import cosine
from transformers import AutoModel,AutoModelForCausalLM, AutoTokenizer
import mysql.connector

dbcon = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="media")

mycursor = dbcon.cursor()
app = FastAPI()


# @app.get("/")
# def read_root():
#     return {
#         "queries": queries,
#         "docs":docs
#     }

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("bounedjarr/sgpt-finetuned-natcat")
    model = AutoModel.from_pretrained("bounedjarr/sgpt-finetuned-natcat")
    return tokenizer,model

tokenizer, model = get_model()

modelML=AutoModelForCausalLM.from_pretrained("bounedjarr/sgpt-finetuned-natcat")

# queries=[]
# docs=[]
# #adding a queries
# @app.post("/insert_queries")
# def insert_queries(text:str):
#     if(text!=""):
#         queries.append(text)
#         return {
#             "message":"text inserted succesfully to the queries",
#             "text inserted ": text,
#             "results after insert":queries
#         }
#     else: 
#         return { "ERROR":"Failed to add the text because it was empty",}
#delete a queries
# @app.post("/insert_docs")
# def insert_docs(text:str):
#     if(text!=""):
#         docs.append(text)
#         return {
#             "message":"text inserted succesfully to the docs",
#             "text inserted ": text,
#             "results after insert":docs
#         }


SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]


def tokenize_with_specb(texts, is_queries):
    # Tokenize without padding
    batch_tokens = tokenizer(texts, padding=False, truncation=True)   
    # Add special brackets & pay attention to them
    for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
        if is_queries:
            seq.insert(0, SPECB_QUE_BOS)
            seq.append(SPECB_QUE_EOS)
        else:
            seq.insert(0, SPECB_DOC_BOS)
            seq.append(SPECB_DOC_EOS)
        att.insert(0, 1)
        att.append(1)
    # Add padding
    batch_tokens = tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
    return batch_tokens


def get_weightedmean_embedding(batch_tokens, model):
    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask

    return embeddings


# Get Embedding
@app.post("/embedding/text")
async def get_embedding(id: int, title: str , desc: str):
    text = title + ' : ' + desc
    doc = []
    doc.append(text)
    embedding = get_weightedmean_embedding(tokenize_with_specb(doc, is_queries=False), model)
    doc_embedding = '''{}'''.format(embedding)
    sql = "INSERT INTO `embedding`(`video_id`, `video_text`, `video_embedding`) VALUES (%s,%s,JSON_OBJECT(%s, %s))"
    val = (id, text, "index", doc_embedding)
    mycursor.execute(sql,val)
    dbcon.commit()
    return {"status" : "success"}


# def get_cosine_similarities(query_embeddings, doc_embeddings):
#     similarities = []
#     for query_emb in query_embeddings:
#         query_sims = []
#         for doc_emb in doc_embeddings:
#             sim = 1 - cosine(query_emb, doc_emb)
#             query_sims.append(sim)
#         similarities.append(query_sims)
#     return similarities
# def assign_scores_to_docs(docs, similarities):
#     result = []
#     for i, doc in enumerate(docs):
#         result.append({doc: similarities[0][i]})
#     return result


#biencoder
# @app.get("/resultsBE")
# def get_embedding():
    
#     queries_embeddings = get_weightedmean_embedding(tokenize_with_specb(queries, is_queries=True), model)
#     doc_embeddings = get_weightedmean_embedding(tokenize_with_specb(docs, is_queries=False), model)
#     # Calculate cosine similarities
#     # Cosine similarities are in [-1, 1]. Higher means more similar
#     results =assign_scores_to_docs(docs, get_cosine_similarities(queries_embeddings, doc_embeddings))

#     return sorted(results, key=lambda x: list(x.values())[0], reverse=True)

# prompt = 'Documents are searched to find matches with the same content.\nThe document "{}" is a good search result for "'

# result=[]
# def sort_query_results(query_results):
#     for query in query_results:
#         results = query['results']
#         results.sort(key=lambda x: float(x['Score']))
#     return query_results

# @app.get("/resultsCE")
# def get_results():
#     modelML.eval()
#     document_results =[]
#     for query in queries:
#         for doc in docs:
#             context = prompt.format(doc)

            # context_enc = tokenizer.encode(context, add_special_tokens=False)
            # continuation_enc = tokenizer.encode(query, add_special_tokens=False)
            # Slice off the last token, as we take its probability from the one before
            # model_input = torch.tensor(context_enc+continuation_enc[:-1])
            # continuation_len = len(continuation_enc)
            # input_len, = model_input.shape

            # [seq_len] -> [seq_len, vocab]
            # logprobs = torch.nn.functional.log_softmax(modelML(model_input)[0], dim=-1).cpu()
            # [seq_len, vocab] -> [continuation_len, vocab]
            # logprobs = logprobs[input_len-continuation_len:]
            # Gather the log probabilities of the continuation tokens -> [continuation_len]
            # logprobs = torch.gather(logprobs, 1, torch.tensor(continuation_enc).unsqueeze(-1)).squeeze(-1)
            # score = torch.sum(logprobs)
            # The higher (closer to 0), the more similar
    #         print(f" ")
    #         document_results.append({"document":doc,"Score":f" {score}"})
    #     result.append({"queries" : query , "results" : document_results})
    # sort_query_results(result)
    # return result[::-1]

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)
