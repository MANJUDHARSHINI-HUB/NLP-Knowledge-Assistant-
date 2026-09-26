# NLP Knowledge Base

## Natural Language Processing (NLP)
Natural Language Processing (NLP) is a field of Artificial Intelligence that enables computers to process, understand, analyze, and generate human language. NLP is used in chatbots, search, translation, sentiment analysis, summarization, question answering, and text generation.

## Types and Tasks of NLP
NLP is commonly described through tasks and applications rather than one fixed list of types. Major tasks include text classification, sentiment analysis, named entity recognition, machine translation, summarization, question answering, information extraction, text generation, conversational AI, and speech-related language processing.

## NLP Applications
Common applications include chatbots, virtual assistants, search engines, spam detection, sentiment analysis, translation, summarization, grammar checking, information extraction, document analysis, recommendation systems, and question answering.

## NLP Pipeline
A typical NLP pipeline includes collecting text, preprocessing, numerical representation or feature extraction, model processing, and evaluation. Modern transformer systems may need less traditional preprocessing because they operate on tokenized text.

## Challenges in NLP
Challenges include ambiguity, context, sarcasm, slang, spelling variation, multiple meanings, long-range dependencies, multilingual text, low-resource languages, and noisy text.

## NLP vs Traditional Programming
Traditional programming uses explicitly defined rules and logic. NLP deals with unstructured and ambiguous human language and often uses machine learning, statistical methods, or neural networks to learn language patterns.

## Text Preprocessing
Text preprocessing prepares raw text for NLP. Common operations include lowercasing, normalization, tokenization, stop-word handling, stemming, lemmatization, and punctuation or number handling.

## Tokenization
Tokenization splits text into smaller units called tokens. Tokens may be words, subwords, characters, or other pieces of text. Modern transformer models commonly use subword tokenization.

## Sentence Tokenization
Sentence tokenization divides a document into individual sentences.

## Word Tokenization
Word tokenization divides text into word-level units. Modern language models often use subword tokens instead.

## Stop Words
Stop words are common words such as “the”, “is”, and “and” that may be removed in some traditional NLP tasks. They should not always be removed because they can contribute to meaning.

## Stemming
Stemming reduces words to a root-like form using simple rules. The result may not be a valid dictionary word.

## Lemmatization
Lemmatization reduces a word to its dictionary base form, called a lemma, using linguistic information.

## Text Normalization
Text normalization makes text more consistent through operations such as lowercasing, whitespace normalization, spelling normalization, and task-specific transformations.

## Part-of-Speech Tagging
POS tagging assigns grammatical categories such as noun, verb, adjective, adverb, pronoun, or preposition to tokens.

## Named Entity Recognition
NER identifies entities such as people, organizations, locations, dates, and products in text.

## Bag of Words
Bag of Words represents a document using words from a vocabulary and their frequencies. It ignores word order.

## N-grams
An n-gram is a sequence of n consecutive tokens. A unigram has one token, a bigram has two, and a trigram has three.

## Term Frequency
Term Frequency measures how often a term appears in a document.

## Inverse Document Frequency
IDF measures how informative a term is across a collection of documents. Terms occurring in many documents receive lower IDF.

## TF-IDF
TF-IDF combines Term Frequency and Inverse Document Frequency. It gives higher importance to terms frequent in one document but relatively uncommon across the collection.

## Text Vectorization
Text vectorization converts text into numerical representations. Traditional methods include Bag of Words and TF-IDF, while modern systems use dense embeddings.

## Text Classification
Text classification assigns text to predefined categories, such as spam or non-spam, topics, or intents.

## Sentiment Analysis
Sentiment analysis identifies sentiment or emotional polarity in text, commonly positive, negative, or neutral.

## Text Summarization
Text summarization creates a shorter version of a document. Extractive summarization selects existing important parts, while abstractive summarization generates new wording.

## Machine Translation
Machine translation automatically converts text from one language to another. Modern systems commonly use neural networks and transformers.

## Question Answering
Question answering systems provide answers using input text, a knowledge base, retrieved information, or a language model.

## Information Extraction
Information extraction identifies useful structured information from unstructured text, such as entities, dates, relationships, and facts.

## Text Generation
Text generation produces new natural-language text from a prompt or context.

## Spam Detection
Spam detection is a classification task that identifies unwanted messages using language patterns and other features.

## Traditional Machine Learning for NLP
Common algorithms include Naive Bayes, Logistic Regression, Support Vector Machines, and Decision Trees, often combined with Bag of Words or TF-IDF features.

## Naive Bayes
Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem and a conditional-independence assumption.

## Logistic Regression
Logistic Regression is a supervised classification algorithm that can use TF-IDF vectors as input features.

## Support Vector Machine
SVM is a supervised algorithm that finds a decision boundary between classes. Linear SVMs have been widely used for text classification.

## Training, Validation, and Testing
Training data learns model parameters. Validation data helps tune choices during development. Test data evaluates the final model on unseen examples.

## Precision, Recall, and F1 Score
Precision measures how many predicted positive cases are actually positive. Recall measures how many actual positive cases were found. F1 combines precision and recall using their harmonic mean.

## Word Embeddings
Word embeddings are dense numerical representations of words that capture relationships and semantic information.

## Word2Vec
Word2Vec learns word embeddings using context. Its two common approaches are CBOW and Skip-Gram.

## CBOW
CBOW predicts a target word from surrounding context words.

## Skip-Gram
Skip-Gram predicts surrounding context words from a target word.

## GloVe
GloVe learns word vectors using word co-occurrence statistics across a corpus.

## FastText
FastText uses character-level subword information to represent words, which can help with rare words and spelling variations.

## Semantic Similarity
Semantic similarity measures how similar two pieces of text are in meaning. Embeddings and cosine similarity can be used for this.

## Sentence Embeddings
Sentence embeddings represent a sentence or passage as a dense vector and are useful for semantic search, retrieval, clustering, and similarity.

## Cosine Similarity
Cosine similarity compares vectors based on the angle between them. Higher similarity generally indicates more similar vector direction.

## Deep Learning for NLP
Deep learning architectures used in NLP include RNNs, LSTMs, GRUs, sequence-to-sequence models, and transformers.

## RNN
An RNN processes sequences while maintaining information from previous steps through a hidden state. It can struggle with long-range dependencies.

## LSTM
LSTM is a recurrent architecture with gates and memory designed to handle longer dependencies.

## GRU
GRU is a recurrent architecture that uses gates to control information flow and is generally simpler than LSTM.

## Sequence-to-Sequence
Seq2Seq models transform one sequence into another and have been used for translation, summarization, and generation.

## Encoder-Decoder
An encoder converts an input sequence into a representation and a decoder generates an output sequence from that representation.

## Attention
Attention lets a model assign different importance to different parts of an input.

## Self-Attention
Self-attention lets tokens in the same sequence interact with one another so each token can use contextual information from other tokens.

## Multi-Head Attention
Multi-head attention performs multiple attention operations so the model can learn different relationships in the input.

## Transformer
The Transformer is a neural network architecture based primarily on attention rather than recurrence. It is central to modern NLP.

## Positional Encoding
Attention does not inherently represent token order, so transformers use positional information to represent token positions.

## BERT
BERT, or Bidirectional Encoder Representations from Transformers, is an encoder-style transformer model that learns contextual representations using information from both directions.

## GPT
GPT, or Generative Pre-trained Transformer, refers to decoder-based transformer language models designed primarily for text generation and many other language tasks.

## T5
T5, or Text-to-Text Transfer Transformer, frames many NLP tasks as converting input text into output text.

## BERT vs GPT
BERT is primarily an encoder-style model used for language understanding and representation tasks, while GPT-style models are decoder-based generative models designed for producing text.

## Large Language Models
LLMs are large neural language models trained on large amounts of text. They can perform generation, question answering, summarization, classification, rewriting, and other language tasks.

## Pretraining
Pretraining is the initial training stage where a model learns general patterns from a large dataset.

## Fine-Tuning
Fine-tuning continues training a pretrained model on a task-specific or domain-specific dataset.

## Transfer Learning
Transfer learning applies knowledge learned in one setting to another task or domain. Pretrained NLP models are a major example.

## Prompt Engineering
Prompt engineering is designing instructions that guide a language model toward a desired output, including task, context, constraints, format, and examples.

## Zero-Shot Learning
Zero-shot learning performs a task without task-specific examples in the prompt.

## Few-Shot Learning
Few-shot learning provides a small number of examples in the prompt to demonstrate the desired task or format.

## Hallucination
Hallucination in generative AI refers to incorrect, unsupported, or fabricated information presented as an answer. Retrieval and grounding can reduce this risk but cannot guarantee its elimination.

## Embeddings
An embedding converts text into a numerical vector representation. Semantically similar text can have nearby representations in embedding space.

## Vector Database
A vector database stores and searches numerical vectors and can retrieve items using similarity search.

## ChromaDB
ChromaDB is a vector database that can store documents and embeddings and retrieve relevant documents using similarity search. In this project it stores the NLP knowledge base.

## Semantic Search
Semantic search retrieves information based on meaning rather than exact keyword matching. A query and stored documents are represented as embeddings and compared.

## Chunking
Chunking divides a large document into smaller pieces before embedding and retrieval so relevant context can be retrieved more precisely.

## Retrieval-Augmented Generation
RAG combines retrieval with language generation. Relevant knowledge is retrieved first and then provided as context to a language model.

## RAG Workflow
1. User asks a question.
2. The question is converted into an embedding.
3. The vector database searches for similar knowledge.
4. Relevant documents are retrieved.
5. Retrieved context is placed into a prompt.
6. The language model generates the answer using that context.

## Why RAG is Useful
RAG grounds answers in a specific knowledge source instead of relying only on information stored in a model's parameters. It is useful for curated or changing knowledge bases.

## RAG vs Fine-Tuning
RAG supplies external context at query time, while fine-tuning changes model behavior through additional training. RAG is useful when the main requirement is access to an external knowledge source.

## NLP Knowledge Assistant Scope
This application is an NLP-focused knowledge assistant. Users can ask questions about NLP concepts, algorithms, techniques, models, architectures, applications, embeddings, transformers, LLMs, RAG, and related NLP topics. Questions unrelated to NLP are outside the intended scope.

## NLP Knowledge Assistant Architecture
User question -> NLP scope check -> text embedding -> ChromaDB semantic retrieval -> relevant NLP context -> RAG prompt -> language model -> final NLP explanation.

## Sentence Transformers in This Project
Sentence Transformers convert text into dense numerical embeddings. This project uses `all-MiniLM-L6-v2` to embed NLP knowledge sections and user questions.

## Scope Checking in This Project
The scope checker stores example NLP questions. The user's question is embedded and compared with those examples using cosine similarity. If similarity is above the configured threshold, the question is treated as NLP-related; otherwise an out-of-scope response is shown.

## Retriever in This Project
The retriever reads the NLP knowledge file, divides it into sections, creates embeddings, stores them in ChromaDB, and searches for sections most similar to the user's question.

## Ollama in This Project
Ollama provides a local interface for running a language model. When available, the application sends the retrieved NLP context and question to the configured model to generate the answer.

## Streamlit in This Project
Streamlit provides the web interface. It displays the title, instructions, question input, Ask button, answer, and retrieved context without requiring React.

## Python in This Project
Python connects the user interface, scope checker, embedding model, vector database, retrieval process, and language model.

## Why This Project Uses RAG
The assistant needs to answer many NLP questions using a dedicated NLP knowledge source. RAG retrieves relevant knowledge first and provides it as context for answer generation.

## End-to-End Example
For "How does self-attention work?", the scope checker identifies the question as NLP-related. The embedding model converts it to a vector. ChromaDB retrieves relevant transformer and self-attention sections. The retrieved context is placed into a RAG prompt. The language model generates a clear answer based on that context.

## Out-of-Scope Example
For "How do I make biryani?", the scope checker determines that the question is not sufficiently related to NLP. The application returns an NLP-focused message instead of performing normal NLP knowledge retrieval.
