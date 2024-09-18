# Next-Token Prediction

Create a language model based on a body of text and get high-quality predictions (next word, next phrase, next pixel, etc.).

## Install

`npm i next-token-prediction`

## Usage

#### Simple (from a built-in data bootstrap)

Put this [`/training/`](https://github.com/bennyschmidt/next-token-prediction/tree/master/training) directory in the root of your project.

Now you just need to create your app's `index.js` file and run it. Your model will start training on the .txt files located in [`/training/documents/`](https://github.com/bennyschmidt/next-token-prediction/tree/master/training/documents). After training is complete it will run these 4 queries:

```javascript
const { Language: LM } = require('next-token-prediction');

const MyLanguageModel = async () => {
  const agent = await LM({
    bootstrap: true
  });

  // Predict the next word

  agent.getTokenPrediction('what');

  // Predict the next 5 words

  agent.getTokenSequencePrediction('what is', 5);

  // Complete the phrase

  agent.complete('hopefully');

  // Get a top k sample of completion predictions

  agent.getCompletions('The sun');
};

MyLanguageModel();
```

-----

#### Advanced (provide `trainingData` or create it from .txt files)

Put this [`/training/`](https://github.com/bennyschmidt/next-token-prediction/tree/master/training) directory in the root of your project.

Because training data was committed to this repo, you can optionally skip training, and just use the bootstrapped training data, like this:

```javascript
const { dirname } = require('path');
const __root = dirname(require.main.filename);

const { Language: LM } = require('next-token-prediction');
const OpenSourceBooksDataset = require(`${__root}/training/datasets/OpenSourceBooks`);

const MyLanguageModel = async () => {
  const agent = await LM({
    dataset: OpenSourceBooksDataset
  });

  // Complete the phrase

  agent.complete('hopefully');
};

MyLanguageModel();
```

Or, train on your own provided text files:

```javascript
const { dirname } = require('path');
const __root = dirname(require.main.filename);

const { Language: LM } = require('next-token-prediction');

const MyLanguageModel = () => {
  // The following .txt files should exist in a `/training/documents/`
  // directory in the root of your project

  const agent = await LM({
    files: [
      'marie-antoinette',
      'pride-and-prejudice',
      'to-kill-a-mockingbird',
      'basic-algebra',
      'a-history-of-war',
      'introduction-to-c-programming'
    ]
  });

  // Complete the phrase

  agent.complete('hopefully');
};

MyLanguageModel();
```

#### Training embeddings

> [!NOTE]
> By default, next-token prediction does not use vector search. To enable it, set `VARIANCE=1` (any value higher than `0`) in an `.env`. This will change the prediction from returning the next likeliest token (n-gram search) to returning the most similar token (vector search) e.g. "The quick brown fox *jumped*..." (n-gram prediction) vs "The quick brown fox *juked*..." (vector similarity). Note that vector search is considerably slower and more resource intensive.

When running the n-gram training using the built-in training method, vector embeddings (144-dimensional) are also created for each token pair to capture context and semantics (e.g. the token `Jordan` has different values in the fragment `Michael Jordan` than it does in the fragment `Syria, Jordan`). The goal of vector search is to optionally enable paraphrasing, slang and profanity filtering, and more. 

Run tests

`npm test`

## Examples

[Readline Completion](./examples/readline-completion/index.js)

[UI Autocomplete](https://github.com/bennyschmidt/next-token-prediction/tree/master/examples/ui-autocomplete)

## Videos

https://github.com/bennyschmidt/next-token-prediction/assets/45407493/68c070bd-ee03-4b7e-8ba3-3885f77fd9f9

https://github.com/bennyschmidt/next-token-prediction/assets/45407493/cd4a1102-5a82-4a6f-abb8-e96805fa65fd

### Browser example: Fast autocomplete

With more training data you can get more suggestions, eventually hitting a tipping point where it can complete anything.

https://github.com/bennyschmidt/next-token-prediction/assets/45407493/942bdabf-4bf5-4d7a-b0db-2331d8c3dd18

## Inspiration

3Blue1Brown video on YouTube:

![YouTube](https://img.youtube.com/vi/wjZofJX0v4M/0.jpg)

Watch: [YouTube](https://www.youtube.com/watch?v=wjZofJX0v4M)

## Goals

1. Provide a high-quality text prediction library for:
  - autocomplete
  - autocorrect
  - spell checking
  - search/lookup
  - summarizing
  - paraphrasing

2. Create pixel and audio transformers for other prediction formats

3. Demystify LLMs & simplify methodologies

4. Make a high-quality, free/open chat-focused LLM in JavaScript, and an equally sophisticated image-focused diffusion model. Working on this [here](https://github.com/bennyschmidt/llimo).
