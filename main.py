import torch
from fastapi import FastAPI, Request
import uvicorn
from transformers import AutoModel,AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from scipy.spatial.distance import cosine
# ================ MySql ================
# import mysql.connector

# dbcon = mysql.connector.connect(
#     host="127.0.0.1",
#     user="root",
#     password="",
#     database="media")

# ================ PostgreSQL ================
import psycopg2

dbcon = psycopg2.connect(
    host="localhost",
    database="media",
    user="postgres",
    password="0123",
    port="5432"
)

mycursor = dbcon.cursor()
app = FastAPI()

# model loading
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("bounedjarr/sgpt-finetuned-natcat")
    model = AutoModel.from_pretrained("bounedjarr/sgpt-finetuned-natcat")
    return tokenizer,model

tokenizer, model = get_model()

modelML=AutoModelForCausalLM.from_pretrained("bounedjarr/sgpt-finetuned-natcat")

# encoding the spec tokens
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


# Get Embedding and upload them to the database
@app.post("/embedding/text")
async def get_embedding(id: int, title: str , desc: str):
    text = title.lower() + ' . ' + desc.lower()
    doc = []
    doc.append(text)
    embedding = get_weightedmean_embedding(tokenize_with_specb(doc, is_queries=False), model)
    x_np = embedding.numpy()
    x_str = json.dumps(x_np.tolist())
    # ================ MySql ================
    # sql = "INSERT INTO `embedding`(`video_id`, `video_text`, `video_embedding`) VALUES (%s,%s,%s)"
    # ================ PostgreSQL ================
    sql = "INSERT INTO embedding (video_id, video_text, video_embedding) VALUES (%s,%s,%s)"
    val = (id, text, x_str)
    mycursor.execute(sql,val)
    dbcon.commit()
    return {"status" : "success"}

# Geting all the documents
def get_all_docs():
    # ================ MySql ================
    # sql = "SELECT video_id,video_embedding FROM `embedding`"
    # ================ PostgreSQL ================
    sql = "SELECT video_id,video_embedding FROM embedding"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    return result


# Fixing the format of the list 
def get_fixed_docs():
    all = []
    listt=get_all_docs()
    for result in listt:
        id = result[0]
        json_str = result[1]
        x_np = np.array(json.loads(json_str))
        x = torch.tensor(x_np)
        all.append((id, x))
    id = [x[0] for x in all]
    doc = [x[1] for x in all]
    # fix the format of the embeddings
    concatenated_array = []
    for arr in doc:
        concatenated_array += arr
    doc = concatenated_array
    return id, doc

# Calculating the cosim
def get_cosine_similarities(query_embeddings, doc_embeddings):
    similarities = []
    for doc_emb in doc_embeddings:
        sim = 1 - cosine(query_embeddings[0], doc_emb)
        similarities.append(sim)
    return similarities

# Assign the top docs to thier corespading ids
def assign_scores_to_docs(ids, similarities):
    result = []
    for i, id in enumerate(ids):
        if(similarities[i]>0.45):
            result.append({id: similarities[i]})
    return result

# Get the results ids 
def extract_ids(lst):
    y=[]
    for i in lst:
        for x in i.keys():
            y.append(x)
    return y

# Get Embedding of search and return ids of videos
@app.get("/search")
def get_serach(text:str):
    ids = []
    docs = []
    queries = []
    queries.append(text.lower())
    queries_embeddings = get_weightedmean_embedding(tokenize_with_specb(queries, is_queries=True), model)
    ids, docs = get_fixed_docs()
    doc_embeddings = docs
    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    results =assign_scores_to_docs(ids, get_cosine_similarities(queries_embeddings, doc_embeddings))
    x= sorted(results, key=lambda x: list(x.values())[0], reverse=True)
    k = extract_ids(x)
    return k


# Run Serveur in command line with : python .\main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)