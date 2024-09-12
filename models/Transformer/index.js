const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;

const { merge } = require('lodash');
const dotenv = require('dotenv');

const {
  combineDocuments
} = require('../../utils');

dotenv.config();

const {
  RANKING_BATCH_SIZE = 50,
  MAX_RESPONSE_LENGTH = 240
} = process.env;

// Tokenizer utils. Designed for words and phrases.

const MATCH_PUNCTUATION = new RegExp(/[.,\/#!$%?“”\^&\*;:{}=\_`~()]/g);
const MATCH_LOWER_UPPER = new RegExp(/([a-z])([A-Z])/g);
const MATCH_NEW_LINES = new RegExp(/\n/g);

const FORMAT_PLAIN_TEXT = [
  /\.\s+|\n|\r|\0/mg,
  /\s-+\s/mg,
  /[©|]\s?/mg,
  /[!(–?$”“…]/mg,
  /\s{2,}|^\s/mg
];

const MATCH_TERMINATORS = new RegExp(/([.?!])\s*(?=[A-Z])/g);
const MATCH_NON_ALPHANUMERIC = new RegExp(/[^a-zA-Z0-9]/);
const MISSING_NGRAM_ERROR = 'Failed to look up n-gram.';
const NOTIF_TRAINING = 'Training...';
const NOTIF_END_OF_STATEMENT = 'End of sequence.';
const NOTIF_UNKNOWN_TOKEN = 'Skipping unrecognized token.';
const NOTIF_END_OF_DATA = 'End of training data.';
const NOTIF_CREATING_CONTEXT = 'Creating context...';
const PARAMETER_CHUNK_SIZE = 50000;
const DIMENSIONS = 9;
const UNIT = 0.09;

// Generator function to chunk arrays
// Use with `PARAMETER_CHUNK_SIZE` for models
// with many parameters to avoid memory errors

function* chunkArray (array, chunkSize) {
  for (let i = 0; i < array.length; i += chunkSize) {
    yield array.slice(i, i + chunkSize);
  }
}

// In-memory representation of context data

const Context = {
  trie: {},
  embeddings: {},
  tokens: [],
  sequences: []
};

module.exports = () => {
  let trainingText = '';

  /**
   * getTokenPrediction
   * Predict the next token or token sequence
   * (agnostic)
   */

  const getTokenPrediction = token => {
    // ngram search

    const rankedTokens = Object.keys(
      ngramSearch(
        token.replace(
          token.charAt(0),
          token.charAt(0).toUpperCase()
        )
      )
    );

    // rank by weight

    rankedTokens.sort((a, b) => a[1] - b[1]);

    const highestRankedToken = rankedTokens[0];

    // return highest ranked token and a top k sample
    // along with an error message (if any)

    if (highestRankedToken) {
      return {
        token: highestRankedToken,
        rankedTokenList: rankedTokens.slice(-RANKING_BATCH_SIZE)
      };
    }

    if (Context.embeddings.hasOwnProperty(token)) {
      // embedding search

      const result = Object
        .keys(embeddings[token])
        .sort((a, b) => embeddings[token][a][0] - embeddings[token][b][0])
        .pop()
        .replace(/\\n/g, ' ')
        .trim();

      if (result) {
        return {
          token: result,
          rankedTokenList: [result]
        };
      }
    }

    const message = MISSING_NGRAM_ERROR;

    return {
      error: {
        message
      },
      token: null,
      rankedTokenList: []
    };
  };

  /**
   * getTokenSequencePrediction
   * Predict the next sequence of tokens.
   * Designed for words and phrases.
   */

  const getTokenSequencePrediction = (token, sequenceLength = 2) => {
    const sequence = [];

    let result = token;

    // get top k sample from getTokenPrediction

    const {
      rankedTokenList: keyPredictions
    } = getTokenPrediction(token);

    // iterate over each token prediction, deriving a
    // new sequence prediction for each token

    for (let i = 0; i < sequenceLength; i++) {
      const {
        token: prediction
      } = getTokenPrediction(result);

      if (prediction) {
        const sanitizedPrediction = prediction
          .replace(/\\n/g, ' ')
          .trim();

        result += ` ${sanitizedPrediction}`;

        sequence.push(sanitizedPrediction);
      }
    }

    // remove duplicates and extra whitespace

    result = [...new Set(sequence)]
      .join(' ')
      .trim();

    // return highest ranked completion and highest
    // ranked next token, along with a top k sample

    return {
      completion: result,
      sequenceLength,
      token: sequence[0],
      rankedTokenList: keyPredictions
    };
  };

  /**
   * getCompletions
   * Complete an input and provide a ranked list
   * of alternatives. Designed for words and phrases.
   */

  const getCompletions = input => {

    // get top k sample from getTokenSequencePrediction

    const {
      completion,
      token,
      rankedTokenList
    } = getTokenSequencePrediction(input, MAX_RESPONSE_LENGTH);

    const completions = [
      completion
    ];

    // build a top k sample of completion predictions

    for (const predictedToken of rankedTokenList) {
      const {
        completion: prediction
      } = getTokenSequencePrediction(`${input} ${predictedToken}`, MAX_RESPONSE_LENGTH);

      completions.push(`${predictedToken} ${prediction}`);
    }

    // return highest ranked completion and highest
    // ranked next token, along with a top k sample of
    // completions

    return {
      completion,
      token,
      rankedTokenList,
      completions
    };
  };

  /**
   * createContext
   * Designed for words and phrases.
   */

  const createContext = embeddings => {
    // Store current context in memory as a trie

    console.log(NOTIF_CREATING_CONTEXT);

    // split sequences

    Context.sequences = trainingText.replace(/\n/g, ' ')
      .replace(MATCH_TERMINATORS, '$1|')
      .split('|')
      .map(toPlainText);

    // create n-grams of all sequences and
    // vectorize each word of each sequence

    const ngrams = Context.sequences.map(sequence => {
      let cursor;

      const words = sequence.split(' ');

      const ngram = words.reduce((a, b) => {
        if (typeof(cursor) === 'object') {
          cursor = cursor[b] = {};
        } else {
          cursor = a[b] = {};
        }

        return a;
      }, {});

      return ngram;
    });

    // deep merge all n-gram sequences

    const embeddingCollection = (
      chunkArray(ngrams, PARAMETER_CHUNK_SIZE)
    );

    // keep references in memory

    for (const embedding of embeddingCollection) {
      Context.trie = merge(Context.trie, ...embedding);
    }

    Context.embeddings = embeddings;

    console.log('Done.');
  };

  /**
   * ngramSearch
   * Look up n-gram by token sequence
   */

  const ngramSearch = sequence => (
    sequence
      .split(/ /)
      .reduce((a, b) => a?.[b], Context.trie) || {}
  );

  /**
   * train
   * Rank tokens then create embeddings.
   * Designed for words and phrases.
   */

  const train = async dataset => {
    const { name, files } = dataset;

    const startTime = Date.now();

    console.log(NOTIF_TRAINING);

    trainingText = await combineDocuments(files);

    // tokenize

    Context.trainingTokens = trainingText.split(' ');

    // Weights

    const tokens = [...Context.trainingTokens];
    const embeddings = {};

    for (const token of tokens) {
      const index = tokens.indexOf(token);

      // End statement on punctuation

      if (token.match(MATCH_PUNCTUATION)) {
        console.log(NOTIF_END_OF_STATEMENT);

        continue;
      }

      // Skip unparsable tokens

      if (MATCH_NON_ALPHANUMERIC.test(token)) {
        console.log(NOTIF_UNKNOWN_TOKEN);

        continue;
      }

      let nextToken = tokens[index + 1];

      // Ensure next token exists

      if (!nextToken) {
        console.log(NOTIF_END_OF_DATA);

        continue;
      }

      if (!embeddings[token]?.[nextToken]) {
        embeddings[token] = {
          ...embeddings[token],

          // initialize vector with random values

          [nextToken]: Array.from(
            { length: DIMENSIONS },
            () => Math.random() / 100
          )
        };
      }

      const frequency = embeddings[token][nextToken][0] + UNIT;

      embeddings[token][nextToken][0] = frequency;

      console.log(`Token "${nextToken}" ranked: ${frequency} (when following ${token}).`);
    }

    // save weights to file

    const embeddingsPath = `${__root}/training/embeddings/${name}.json`;

    await fs.writeFile(
      embeddingsPath,
      JSON.stringify(embeddings)
    );

    console.log(`Wrote to file: ${embeddingsPath}.`);

    console.log(
      `Training completed in ${(Date.now() - startTime) / 1000} seconds.`
    );

    // create in-memory context

    createContext(embeddings);
  };

  /**
   * ingest
   * Provide new document text
   */

  const ingest = text => {
    trainingText = text;
    Context.trainingTokens = trainingText.split(' ');
  };

  /**
   * toPlainText
   * Transform text to a plain format. Designed for words
   * and phrases. Capitalizes the first token of sequences,
   * removing certain special characters, new lines, etc.
   */

  const toPlainText = text => text
    .replace(
      text.charAt(0),
      text.charAt(0).toUpperCase()
    )
    .replace(MATCH_LOWER_UPPER, '$1 $2')
    .replace(MATCH_NEW_LINES, ' ')
    .replace(FORMAT_PLAIN_TEXT[0], ' ')
    .replace(FORMAT_PLAIN_TEXT[1], ' ')
    .replace(FORMAT_PLAIN_TEXT[2], ' ')
    .replace(FORMAT_PLAIN_TEXT[3], ' ')
    .replace(FORMAT_PLAIN_TEXT[4], ' ');

  // Transformer API

  return {
    ingest,
    train,
    createContext,
    getTokenPrediction,
    getTokenSequencePrediction,
    getCompletions
  };
};
