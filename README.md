# Next-Token Prediction

Train a language model on a body of text and get high-quality predictions (next word, next phrase, next pixel, etc.). With enough training data and a good chat interface, this can be used instead of well-known decoder-only models like GPT, Mistral, etc.

## Install

`npm i next-token-prediction`

## Usage

#### Simple (from a built-in data bootstrap)

```javascript
const MyLanguageModel = async () => {
  const { Language: LM } = require('next-token-prediction');

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

#### Advanced (provide `trainingData` or create it from .txt files)

From the dataset included in this repo:

```javascript
const MyLanguageModel = async () => {
  const { dirname } = require('path');
  const __root = dirname(require.main.filename);

  const { Language: LM } = require('next-token-prediction');
  const OpenSourceBooksDataset = require(`${__root}/training/datasets/OpenSourceBooks`);

  const agent = await LM({
    dataset: OpenSourceBooksDataset
  });

  // Complete the phrase

  agent.complete('hopefully');
};

MyLanguageModel();
```

From your own .txt files.

⚠️ This can take a while, and uses a lot of CPU/GPU:

```javascript
const MyLanguageModel = () => {
  const { dirname } = require('path');
  const __root = dirname(require.main.filename);

  const { Language: LM } = require('next-token-prediction');

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

## Video

TBD
