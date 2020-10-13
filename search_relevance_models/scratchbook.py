# %%
# import NLP libraries
import spacy
from gensim.corpora.dictionary import Dictionary

# %% 
# import utility and data libraries
import os
import re
import pandas as pd
import numpy as np

# %%
# load spacy model
nlp = spacy.load("en_core_web_md")

# %%
# load state of the union texts
def load_texts(dir_path):
    """
    - Parameters: dir_path (string) for a directory containing text files.
    - Returns: A list of dictionaries with keys file_name and text.
    """
    docs = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r+", encoding="utf-8") as file:
                text = file.read()
                current = {
                    "file_name": file_name,
                    "text": text
                }
                docs.append(current)
    return docs

def add_sotu_metadata(sotu_doc_dict):
    """
    - Parameters: sotu_doc_dict (dictionary) with sotu metadata.
      Expects a file_name key in format "president_year.txt"
    - Returns: A dictionary with appended president and year keys.
    """
    file_name = sotu_doc_dict["file_name"]
    pres, year, filetype = re.split(r"[^A-Za-z0-9]", file_name)
    sotu_doc_dict["president"] = pres
    sotu_doc_dict["year"] = int(year)
    return sotu_doc_dict

def load_sotu_texts(dir_path):
    """
    - Parameters: dir_path (string) for a directory containing text files.
      Expects sotu text files in dir_path in format "president_year.txt".
    - Returns: A Pandas DataFrame with file_name, text, president, and year
      columns for each sotu text in dir_path.
    """
    docs = load_texts(dir_path)
    docs = [add_sotu_metadata(d) for d in docs]
    docs = sorted(docs, key=lambda d: d["year"])
    df = pd.DataFrame(docs)
    return df[["year", "president", "text"]]

sotu_df = load_sotu_texts("data")

# %%
# simple boolean search
def search_df_texts(df, query_string: str):
    """
    - Parameters: df (Pandas DataFrame), query_string (string). df must 
      contain a "text" column.
    - Returns: A subset of df containing only rows where each term in 
      query_string appeared in df["text"].
    """
    terms = query_string.lower().split(" ")
    filters = [df["text"].str.lower().str.contains(term) for term in terms]
    return df[np.all(filters, axis=0)]

search_term = "space rocket soviet"
results = search_df_texts(sotu_df, search_term)

print(f"Num results for query '{search_term}': {results.shape[0]}")
print(results.head())

# %%
# tokenize documents
def spacy_doc(model, text, lower=True):
    """
    - Parameters: model (Spacy model), text (string), lower (bool).
    - Returns: A Spacy Document object processed using the provided
      model. Document is all lowercase if lower is True.
    """
    if lower:
        text = text.lower()
    return model(text)

sotu_docs = [spacy_doc(nlp, text) for text in sotu_df["text"]]

# %%
# build dictionary
def get_token_texts(doc):
    """
    - Parameters: doc (Spacy Document object).
    - Returns: A list of strings based on the text value of each token
      in doc.
    """
    token_list = [token for token in doc]
    return [token.text for token in token_list]

def build_dictionary(doc_list):
    """
    - Parameters: doc_list (list of Spacy Document objects).
    - Returns: A Gensim Dictionary, built using the tokens in each document
      contained in doc_list.
    """
    return Dictionary([get_token_texts(doc) for doc in doc_list])

sotu_dictionary = build_dictionary(sotu_docs)

# %%
# build bag-of-words model
def build_corpus(doc_list, dictionary):
    """
    - Parameters: doc_list (list of Spacy Document objects), dictionary
      (Gensim Dictionary object).
    - Returns: A list of documents in bag-of-words format, containing tuples
      with (token_id, token_count) for each token in the text.
    """
    return [dictionary.doc2bow(get_token_texts(doc)) for doc in doc_list]

def build_td_matrix(doc_list, dictionary):
    """
    - Parameters: doc_list (list of Spacy Document objects), dictionary
      (Gensim Dictionary object).
    - Returns: A term-document matrix in the form of a 2D NumPy Array, where
      each row contains the count of a token in the corresponding document
      and each column index is the id of a token in the dictionary.
    """
    corpus = build_corpus(sotu_docs, sotu_dictionary)
    tdm = []
    for bow in corpus:
        vector = np.zeros(len(dictionary))
        for token_id, token_count in bow:
            vector[token_id] = token_count
        tdm.append(vector)
    return np.array(tdm)

def build_term_document_df(doc_list, dictionary):
    """
    - Parameters: doc_list (list of Spacy Document objects), dictionary
      (Gensim Dictionary object).
    - Returns a term-document matrix in the form of a Pandas Dataframe, 
      where each row is a document and each column is a token. Values in
      the dataframe are token counts for the given document / token.
    """
    tdm = build_td_matrix(doc_list, dictionary)
    cols = list(dictionary.token2id.keys())
    return pd.DataFrame(tdm, columns=cols, dtype=pd.Int64Dtype)

sotu_corpus = build_corpus(sotu_docs, sotu_dictionary)
sotu_tdm = build_td_matrix(sotu_docs, sotu_dictionary)
sotu_td_df = build_term_document_df(sotu_docs, sotu_dictionary)


# %%
# term-document frequency search based on the bag-of-words model
def search_td_df(td_df, text_df, query_string: str):
    """
    - Parameters: td_df (Pandas DataFrame) representing a term-document matrix,
      text_df (Pandas DataFrame) with a "text" column and rows that correspond
      to the td_df, and query_string (string).
    - Returns: A new dataframe that only contains rows from text_df where the 
      "text" column had at least one occurence of each term in query_string.
      Additional columns are added to show the count of each term and the
      total count of all terms.
    """
    terms = query_string.lower().split(" ")
    filters = [td_df[term] > 0 for term in terms]
    filtered_td_df = td_df[np.all(filters, axis=0)][terms]
    filtered_td_df["terms_sum"] = filtered_td_df.agg(sum, axis=1).astype("int64")
    full_df = text_df.merge(filtered_td_df, left_index=True, right_index=True)

    return full_df.sort_values("terms_sum", ascending=False)

search_td_df(sotu_td_df, sotu_df, search_term).head()

# %%
# build tf-idf model
def document_frequency(td_df, term: str):
    """
    - Parameters: td_df (Pandas DataFrame) representing a term-document matrix,
      and term (string).
    - Returns: The document frequency value showing the number of documents in
      td_df where term occurs at least once.
    """
    return td_df[td_df[term] > 0].shape[0]

def inverse_document_frequency(td_df, term: str):
    """
    - Parameters: td_df (Pandas DataFrame) representing a term-document matrix,
      and term (string).
    - Returns: The inverse document frequency value for term, calculated as 
      N / log(dft) where N is the number of documents in td_df and dft is the
      document frequency value for term.
    """
    N = td_df.shape[0]
    dft = document_frequency(td_df, term)
    return (N / np.log10(dft))
    
def build_tfidf_df(td_df):
    """
    - Parameters: td_df (Pandas DataFrame) representing a term-document matrix.
    - Returns: Returns a term frequency-inverse document frequency (TF-IDF)
      matrix in the form of a Pandas DataFrame, where each row is a document and
      each column is a token. Values in the dataframe are TF-IDF values for the
      given document / token.
    """
    def calculate_tfidf(col, td_df):
        idf = inverse_document_frequency(td_df, col.name)
        return col * idf
    
    return td_df.apply(calculate_tfidf, td_df=td_df)

# %%
sotu_tfidf_df = build_tfidf_df(sotu_td_df)

# %%
# search based on the tf-idf model
def search_tfidf_df(tfidf_df, text_df, query_string: str):
    """
    - Parameters: tfidf_df (Pandas DataFrame) representing a tf-idf matrix,
      text_df (Pandas DataFrame) with a "text" column and rows that correspond
      to the tfidf_df, and query_string (string).
    - Returns: A new dataframe that only contains rows from text_df where the 
      corresponding tf-idf value was greater than zero for each of the terms 
      in query_string. Additional columns are added to show the tf-idf value
      for each term and the sum of the tf-idf values. 
    """
    terms = query_string.lower().split(" ")
    filters = [tfidf_df[term] > 0 for term in terms]
    filtered_tfidf_df = tfidf_df[np.all(filters, axis=0)][terms]
    filtered_tfidf_df["tfidf_sum"] = filtered_tfidf_df.agg(sum, axis=1)
    full_df = text_df.merge(filtered_tfidf_df, left_index=True, right_index=True)

    return full_df.sort_values("tfidf_sum", ascending=False)

search_tfidf_df(sotu_tfidf_df, sotu_df, search_term).head()


# %%
