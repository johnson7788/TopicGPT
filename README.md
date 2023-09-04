# TopicGPT
TopicGPT integrates the remarkable capabilities of current LLMs such as GPT-3.5 and GPT-4 into topic modelling. 

While traditional topic models extract topics as simple lists of top-words, such as ["Lion", "Leopard", "Rhino", "Elephant", "Buffalo"], TopicGPT offers rich and dynamic topic representations that can be intuitively understood, extensively investigated and modified in various ways via a simple text commands. 

More specifically, it provides the following core functionalities: 
- Identification of clusters of documents and top-word extraction
- Generation of detailed and informative topic descriptions 
- Extraction of detailed information about topics via Retrieval-Augmented-Generation (RAG)
- Comparison of topics
- Splitting and combining of identified topics
- Addition of new topics based on keywords
- Deletion of topics
  
It is further possible, to directly interact with TopicGPT via prompting and without explicitly calling  functions - an LLM autonomously decides which functionality to use.

## Installation

## Example 

The following example demonstrates how TopicGPT can be used on a real-world dataset. A subset of the Twenty Newsgroups corpus (https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) will be used for this purpose. 

### Load the data

```python
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')) #download the 20 Newsgroups dataset
corpus = data['data'][:1000] # just select the first 1000 documents for this example

corpus = [doc for doc in corpus if doc != ""] #remove empty documents
```
### Initialize the model 

Note that an OpenAi API-Key is needed to compute the embeddings and execute the prompts. See https://platform.openai.com/account/api-keys for more details. 

```python 
from topicgpt.TopicGPT import TopicGPT

tm = TopicGPT(
    openai_api_key = <your-openai-api-key>
)
```

### Fit the model 

The fit method fits the model. This can take, depending on the size of the dataset and wether embeddings have been provided, from a few minutes to several hours. 
```python 
tm.fit(corpus) # the corpus argument has the type list[str] where each string represents one document
```

### Inspect the found topics

Obtain an overview over the indentified topics
```python
print(tm.topic_lis)
```
Output
```
[Topic 0: Scientific Experiments,
 Topic 1: Sports,
 Topic 2: Cryptography and Security,
 Topic 3: Computer Hardware and Software Compatibility,
 Topic 4: medical treatments,
 Topic 5: Motorcycle Racing,
 Topic 6: Automotive,
 Topic 7: Religious Doctrine,
 Topic 8: Space Exploration,
 Topic 9: Technology and Sports,
 Topic 10: War Crimes]
```

#### Topic-based Prompting 

```python
from TopicPrompting.TopicPrompting import TopicPrompting

pmp = TopicPrompting(
    openai_prompting_model = "gpt-4",
    max_context_length_promting = 4000,
    topic_lis = topics,
    openai_key = <your_openai_key>, 
    enhancer=enhancer,
    vocab_embeddings=vocab_embeddings
)
pmp.show_topic_list() #display list of available topics 
```

See the detailed topic description for topic 13

```python
pmp.topic_lis[13].topic_description 
```

This will execute retrieval-augmented generation based on the keyword "Jupiter" for topic 13 and tell you which information on Jupiter topic 13 contains
```python
print(pmp.prompt_knn_search(llm_query = "What information on Jupiter does topic 13 contain)) 
```
You can identify the subtopics of a given topic.
```python
pmp.general_prompt("What subtopics does topic 6 have?")
```

Based on the previous analysis, you can ask TopicGPT to actually split a topic based on the previous analysis. 
```python
pmp.general_prompt("Please actually split topic 6 into its subtopics. Do this inplace.")
```

One can also combine topics. 
```python
pmp.general_prompt("Combine the topics 19 and 20 into one single topic")
```

It is also possible to create completely new, additional topics
```python
pmp.general_prompt("Please create a new topic based on Climate Change")
```
## How TopicGPT works

## References

Please note that the topword extraction methods used for this package are based on similar ideas as found in the Bertopic Model (Grootendorst, Maarten. "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." arXiv preprint arXiv:2203.05794 (2022)) in the case of the tf-idf method and in Top2Vec for the centroid-similarity method (Angelov, Dimo. "Top2vec: Distributed representations of topics." arXiv preprint arXiv:2008.09470 (2020)).


👷‍♀️🚧👷
Note that this repository is still under developement and will be finished by 08.09.2023. 
👷‍♀️🚧👷
