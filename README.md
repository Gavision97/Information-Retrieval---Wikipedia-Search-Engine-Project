# Wikipedia Search Engine Project

![wikipedia_imgjpg](https://github.com/Gavision97/Information-Retrieval---Wikipedia-Search-Engine-Project/assets/150701079/998fd2d9-d319-4ad5-bc30-1c85ed92914a)


## Overview
The Wikipedia Search Engine project is a comprehensive search tool designed specifically for Wikipedia articles. Leveraging various techniques, including the generation of inverted indexes for both titles and text from the Wikipedia corpus, the search engine aims to provide relevant and accurate results for user queries.

## Features
- **Inverted Index Generation:** The project generates inverted indexes for both titles and text from the Wikipedia corpus. These indexes play a crucial role in quickly retrieving relevant documents based on user queries.
- **Page Rank and Page Views:** Utilizing advanced algorithms, the search engine incorporates page rank and page views to enhance the relevance of search results. This ensures that popular and authoritative articles are given priority in the search results.
- **BM25 Integration:** The search engine combines inverted indexes with the BM25 algorithm, a state-of-the-art information retrieval technique. This integration further improves the accuracy and effectiveness of the search engine.
- **Data Processing on GCP:** All index generation and dictionary creation processes are performed on the Google Cloud Platform (GCP) using Dataproc clusters and Spark dataframes. This allows for efficient processing of large volumes of data inherent in Wikipedia articles.

## Usage
1. **Query Input:** Users can input their search queries through the search interface.
2. **Index Retrieval:** The search engine retrieves relevant documents based on the user query using the generated indexes and algorithms.
3. **Result Presentation:** Relevant documents are presented to the user, ranked according to their relevance and importance.

## Technologies Used
- Python
- Apache Spark
- Google Cloud Platform (GCP)
- BM25 Algorithm
- Inverted Indexing
- Page Rank Algorithm
- Page Views Data
