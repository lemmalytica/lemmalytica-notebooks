# %%
# Load libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# set plot styles
sns.set_style("darkgrid")

# %%
def load_docs(dir_path):
    """
    - Parameters: dir_path (string) for a directory containing text files.
    - Returns: A list of dictionaries with keys file_name and text.
    """
    docs = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if file_name.endswith(".html") and os.path.isfile(file_path):
            with open(file_path, "r+", encoding="utf-8") as file:
                text = file.read()
                current = {
                    "file_name": file_name,
                    "text": text
                }
                docs.append(current)
    return docs

html_docs = load_docs("data")
html_docs = sorted(html_docs, key = lambda d: d["file_name"])

# %%
def extract_paper_data(paper):
    """
    - Parameters: paper (BeautifulSoup object)
    - Returns: A dictionary of paper attributes.
    """
    title_anchor = paper.find("h3").find("a")
    author_info = paper.find(class_="gs_a")
    authors = author_info.text.split("-")[0].strip().split(", ")
    year = "".join(re.findall(r' \d{4}', author_info.text))[-4:]
    blurb = paper.find(class_="gs_rs")
    usage_data = paper.find(class_="gs_fl")
    citation_anchor = usage_data.find_all("a")[2]
    num_cites = int("".join(re.findall(r'\d*', citation_anchor.text)))
    
    return {
        "title": title_anchor.text if title_anchor else "",
        "authors": authors,
        "blurb": blurb.text if blurb else "",
        "citations": num_cites,
        "year": int(year) if year != "" else "",
        "link": title_anchor["href"] if title_anchor else ""
    }

def paper_df(paper_html):
    """
    - Parameters: paper_html (string of html text from a Google Scholar page)
    - Returns: A Pandas DataFrame with data for each paper in paper_html
    """
    paper_soup = BeautifulSoup(paper_html, "html.parser")
    all_paper_data = []
    results = paper_soup.find(id="gs_res_ccl_mid")
    papers = results.find_all("div", class_="gs_ri")

    for paper in papers:
        paper_data = extract_paper_data(paper)
        all_paper_data.append(paper_data)
    
    return pd.DataFrame(all_paper_data)

def load_papers(html_docs):
    """
    - Parameters: html_docs (a list of dictionaries with file_name and text keys)
    - Returns: A Pandas DataFrame with data from each of the papers in html_docs
    """
    all_dfs = []
    for entry in html_docs:
        df = paper_df(entry["text"])
        all_dfs.append(df)
    
    full_df = pd.concat(all_dfs).sort_values("citations", ascending=False)
    full_df = full_df.reset_index(drop=True)
    return full_df

def clean_papers(papers_df):
    """
    - Parameters: papers_df (Pandas DataFrame)
    - Returns: A dataframe with rows that contain empty cells removed.
    """
    clean_df = papers_df.copy()
    clean_df = clean_df.replace("", np.nan, regex=True)
    clean_df = clean_df.dropna()
    return clean_df

papers = load_papers(html_docs)
papers = clean_papers(papers)

# %%
# add citation_rate column
def get_citation_rate(citations, year_published):
    """
    - Parameters: citations (int), year_published (int)
    - Returns: The number of citations per year, since the year published.
    """
    current_year = pd.datetime.now().year
    years_since_publish = current_year - year_published
    return citations / (years_since_publish + 1)

papers["citation_rate"] = get_citation_rate(papers["citations"], papers["year"])
papers = papers[["title", "authors", "blurb", "citations",
                "citation_rate", "year", "link"]]
papers.head(10)

# %%
# prolific authors
def get_author_counts(papers_df):
    """
    - Parameters: papers_df (Pandas DataFrame)
    - Returns: A dataframe with a count of each author in pandas_df["authors"]
    """
    authors = papers_df["authors"].apply(pd.Series).stack().reset_index(name="author")
    authors = authors["author"]
    author_counts = authors.value_counts()
    author_counts = author_counts.rename_axis("author").reset_index(name="count")
    author_counts = author_counts.sort_values(by=["count", "author"], 
                                              ascending=[False, True])
    return author_counts

author_counts = get_author_counts(papers)
author_counts.head(10)

# %%
def filter_by_author(papers_df, author_name):
    """
    - Parameters: papers_df (Pandas DataFrame), author_name (str)
    - Returns: A dataframe with entries from papers_df by author.
    """
    return papers_df[papers_df["authors"].apply(
        lambda authors: author_name in authors)]

filter_by_author(papers, "C Friedman")

# %%
def get_author_citation_counts(papers_df):
    """
    - Parameters: papers_df (Pandas DataFrame)
    - Returns: A dataframe with a citation count for each author.
    """
    df = papers_df.explode("authors")
    df = df[["authors", "citations", "citation_rate"]]
    author_citations = df.groupby("authors").sum()
    author_citations = author_citations.reset_index()
    return author_citations

author_citation_counts = get_author_citation_counts(papers)
author_citation_counts.head(10)

# %%
# plot top authors
def plot_counts(df, title, subtitle, x_col, x_lab, y_col="count", y_lab="Count"):
    """
    - Parameters df (Pandas DataFrame), title (str), subtitle (str), x_col (str),
      x_lab (str), y_col (str), y_lab (str)
    - Plots a barplot of df using the provided x and y columns.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=x_col,
        y=y_col,
        data=df,
        alpha=0.9,
        ax=ax
    )
    ax.set(
        xlabel=x_lab,
        ylabel=y_lab,
    )
    ax.text(
        x=0.5,
        y=1.15,
        s=title,
        fontsize=16,
        ha="center",
        va="bottom",
        transform=ax.transAxes
    )
    ax.text(
        x=0.5,
        y=1.05,
        s=subtitle,
        fontsize=14,
        ha="center",
        va="bottom",
        transform=ax.transAxes
    )
    plt.xticks(rotation=90)
    plt.show()

title = "Influential NLP Papers on Google Scholar"
subtitle = "Most Prolific Authors"
plot_counts(author_counts.head(10), title, subtitle, "author", "Author")

# %%
# citations by year
def get_yearly_citation_count(papers_df):
    """
    - Parameters: papers_df (Pandas DataFrame)
    - Returns: A dataframe with a count of citations per year
    """
    counts = papers_df.groupby("year").sum()
    counts = counts.reset_index()
    counts["citation_rate"] = get_citation_rate(counts["citations"], counts["year"])
    return counts

yearly_citations = get_yearly_citation_count(papers)

# %%
# plot yearly citations
def plot_citations_by_year(df, title, subtitle, year_col="year"):
    """
    - Parameters df (Pandas DataFrame), title (str), subtitle (str), year_col (str)
    - Plots a scatterplot of the count of citations and citation_rate in df by year.
    """
    fig, (ax0, ax1) = plt.subplots(figsize=(7, 11), nrows=2, ncols=1)

    # total citations
    sns.scatterplot(
        x="year",
        y="citations",
        data=df,
        alpha=0.7,
        color="navy",
        ax=ax0
    )
    ax0.set(
        xlabel="Year",
        ylabel="Total citations",
        title="Total citations, per year"
    )

    # total citation rate
    sns.scatterplot(
        x="year",
        y="citation_rate",
        data=df,
        alpha=0.7,
        color="teal",
        ax=ax1
    )
    ax1.set(
        xlabel="Year",
        ylabel="Citations per year",
        title="Citation rate, by year"
    )

    # titles
    ax0.text(
        x=0.5,
        y=1.20,
        s=title,
        fontsize=16,
        ha="center",
        va="bottom",
        transform=ax0.transAxes
    )
    ax0.text(
        x=0.5,
        y=1.10,
        s=subtitle,
        fontsize=14,
        ha="center",
        va="bottom",
        transform=ax0.transAxes
    )
    plt.show()

subtitle = "Citations and Citation Rate by Year"
plot_citations_by_year(papers, title, subtitle)

# %%
# top papers by citation
def top_papers_by_col(papers_df, sort_col, limit=10):
    """
    - Parameters: papers_df (Pandas DataFrame)
    - Returns: A dataframe with [limit] entries based on the highest values for
      sort_col in papers_df.
    """
    df = papers_df.copy()
    df = df.sort_values(by=sort_col, ascending=False)
    return df.head(limit)

top_papers_by_col(papers, "citations")

# %%
# top papers by citation_rate
top_papers_by_col(papers, "citation_rate")

# %%
# up-and-coming papers
def filter_by_year(df, filter_year, year_col="year"):
    """
    - Parameters: df (Pandas DataFrame), filter_year (int), year_col (str)
    - Returns: A dataframe where year_col in df is filtered by year.
    """
    return df[df[year_col] == filter_year]

top_papers_by_col(filter_by_year(papers, 2020), "citations")

# %%
# plot papers by year
def plot_count_by_year(df, title, subtitle, year_col="year"):
    """
    - Parameters df (Pandas DataFrame), title (str), subtitle (str), year_col (str)
    - Plots a scatterplot of the count of rows in df, grouped by year_col.
    """
    counts = df.groupby(year_col).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        x="year",
        y="count",
        data=counts,
        alpha=0.8,
        ax=ax
    )
    ax.set(
        xlabel="Year",
        ylabel="Count",
    )
    ax.text(
        x=0.5,
        y=1.15,
        s=title,
        fontsize=16,
        ha="center",
        va="bottom",
        transform=ax.transAxes
    )
    ax.text(
        x=0.5,
        y=1.05,
        s=subtitle,
        fontsize=14,
        ha="center",
        va="bottom",
        transform=ax.transAxes
    )
    plt.xticks(rotation=90)
    plt.show()

subtitle = "Papers by Year"
plot_count_by_year(papers, title, subtitle)
