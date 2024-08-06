#!/usr/bin/env python
# coding: utf-8

# Example Usage

import json
import os
import pickle
import httpx
from openai import OpenAI
from topicgpt.TopicGPT import TopicGPT

def load_data():
    """Fetch and clean the 20 Newsgroups dataset."""
    interview_files = [f for f in os.listdir('.') if f.startswith('interview_user202407')]
    corpus =[]
    for file in interview_files:
        with open(file,'r') as f:
            data = json.load(f)
        interview = data["interview"]
        answers = [one["answer"] for one in interview]
        answer_text = "\n".join(answers)
        corpus.append(answer_text)
    report_text = """Meituan Takeaway User Experience Interview Result Report Framework
Objective
To gather detailed feedback on consumers' experiences with Meituan Takeaway.

Background
Interview users of Meituan Takeaway to understand their experiences, preferences, and challenges in order to enhance the service.

Executive Summary
Key insights and overall satisfaction score.

1. Demographics
Age, gender, occupation distribution.
2. Usage Patterns
Frequency of use.
Typical usage scenarios.
Types of products purchased.
3. Experience Highlights
Positive and negative aspects of the service.
4. Satisfaction and Expectations
Satisfaction ratings.
Key advantages and disadvantages.
Expectation fulfillment.
5. Suggestions for Improvement
Common suggestions.
Desired features and new functions.
Potential Insights
Identify key demographics and their specific usage patterns.
Highlight common positive experiences and pain points.
Assess overall satisfaction and areas needing improvement.
""" #报告模版
    corpus.append(report_text)
    return corpus


def initialize_model(api_key, n_topics=6):
    """Initialize the TopicGPT model."""
    http_client = httpx.Client(
        proxy="http://wac8:7890",
    )
    embedding_client = OpenAI(api_key="",base_url="http://localhost:11434/v1", http_client=http_client)
    embedding_model = "mxbai-embed-large"
    # embedding_model = "text-embedding-ada-002"
    return TopicGPT(api_key=api_key, base_url="https://api.groq.com/openai/v1",http_client=http_client, n_topics=n_topics,openai_prompting_model="llama-3.1-70b-versatile",embedding_model=embedding_model,embedding_client=embedding_client)


def fit_model(tm, corpus):
    """Fit the TopicGPT model on the provided corpus."""
    tm.fit(corpus)


def load_model(filepath):
    """Load a pre-trained TopicGPT model from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_topics_overview(tm):
    """Print the list of topics and visualize the clusters."""
    print(tm.topic_lis)
    tm.print_topics()
    tm.visualize_clusters()


def detailed_topic_info(tm, keyword, topic_index):
    """Obtain detailed information about a specific topic."""
    tm.pprompt(f"Which information on the keyword '{keyword}' does topic {topic_index} have?")
    print(tm.topic_lis[topic_index].documents[102])


def split_topic(tm, topic_index, keywords):
    """Split a topic into subtopics based on provided keywords."""
    keywords_str = ', '.join([f"'{keyword}'" for keyword in keywords])
    tm.pprompt(
        f"Please split topic {topic_index} into subtopics based on the keywords {keywords_str}. Do this inplace.")


def combine_topics(tm, topic_indices):
    """Combine multiple topics into one."""
    indices_str = ' and '.join([str(index) for index in topic_indices])
    tm.pprompt(f"Please combine topics {indices_str}. Do this inplace.")


def delete_topic(tm, topic_index):
    """Delete a specific topic."""
    tm.pprompt(f"Please delete topic {topic_index}. Do this inplace.")


def compare_topics(tm, topic_indices):
    """Compare multiple topics."""
    indices_str = ' and '.join([str(index) for index in topic_indices])
    tm.pprompt(f"Please compare topics {indices_str}.")


def add_new_topic(tm, keyword):
    """Add a new topic based on a keyword."""
    tm.pprompt(f"Please add a completely new topic based on the keyword '{keyword}'.")


if __name__ == "__main__":
    api_key = os.environ.get('OPENAI_API_KEY')
    corpus = load_data()
    tm = initialize_model(api_key)

    fit_model(tm, corpus)

    # To load a pre-trained model instead of fitting a new one
    # tm = load_model("../Data/SavedTopicRepresentations/TopicGPT_20ng.pkl")

    get_topics_overview(tm)

    detailed_topic_info(tm, 'moon landing', 13)
    tm.pprompt("What are 5 potential subtopics of topic 6")

    split_topic(tm, 6, ['religious faith', 'atheism', 'ethics and philosophy'])

    combine_topics(tm, [15, 17])

    delete_topic(tm, 10)

    compare_topics(tm, [5, 7])

    add_new_topic(tm, 'Politics and government')
