# Next-Token Prediction

Create a language model based on a body of text and get high-quality predictions (next word, next phrase, next pixel, etc.). With enough training data and a good chat interface, this can be used instead of well-known decoder-only models like GPT, Mistral, etc.

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

Because training data was committed to this repo, you can optionally skip training, and just use the bootstrapped dataset and embeddings, like this:

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

## Examples

[Readline Completion](./examples/readline-completion/index.js)

## Video

TBD
