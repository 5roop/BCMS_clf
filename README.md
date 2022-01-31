# PLAN

```latex

\subsection{With feature selection}

Feature selection dataset - web data (as was used until now)

Training dataset - web data (as was used until now)

Hyperparameter selection dataset - train split

Evaluation dataset - dev+test split, log micro- macro-F1 and Confusion Matrices

\subsection{Without feature selection}

Approaches
- 3-6 char ngrams (sklearn, countvectorizer)
- fastText
- maybe Transformers

Training dataset - web data (as is used above, if feasible, otherwise, we will use a subset)

Hyperparameter selection dataset - train split (if needed)

Evaluation dataset - dev+test split,  log micro- macro-F1 and Confusion Matrices

```

# First part: With feature selection

## Methodology
In this part I optimize parameter `N`, meaning that I repeatedly generate `N` best tokens, generate a boolean tokenizer with it and train a Linear SVC on web data. The resulting classifier is evaluated on the train split of Twitter and SETimes dataset.

For SETimes I get the following profile:

![](images/1_linscv_on_setimes_bool_vectorizer.png)


