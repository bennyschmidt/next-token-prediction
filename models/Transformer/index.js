const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;

const { merge } = require('lodash');
const dotenv = require('dotenv');

const WordToVec8 = require('./wordToVec8');

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
  bigrams: {}
};

module.exports = () => {
  let trainingText = '';
  let trainingTokens = [];
  let trainingTokenSequences = [];
  let wordToVec8 = {};

  /**
   * getSingleTokenPrediction
   * Predict the next token (agnostic)
   *
   * Uses n-gram and wordToVec8.
   */

  const getSingleTokenPrediction = token => {
    // wordToVec8 transform

    const wordToVec8Prediction = (
      wordToVec8.getMostFrequentNextToken(token)
    );

    // n-gram lookup

    const tokens = Context.bigrams[token];

    if (tokens) {
      const rankedTokens = [];

      for (const rankedToken of tokens) {
        rankedTokens.push([rankedToken, tokens[rankedToken]]);
      }

      // rank tokens by weight

      rankedTokens.sort((a, b) => a[1] - b[1]);

      const highest = rankedTokens[0];
      const highestRankedToken = highest[0];

      // return the highest ranked token & top k sample
      // and a wordToVec8 prediction for comparison

      if (highestRankedToken) {
        return {
          token: highestRankedToken,
          rankedTokenList: rankedTokens
            .map(([_token]) => _token)
            .slice(-RANKING_BATCH_SIZE),
          wordToVec8Prediction
        };
      }
    }

    const message = MISSING_NGRAM_ERROR;

    return {
      error: {
        message
      },
      token: null,
      rankedTokenList: [],
      wordToVec8Prediction
    };
  };

  /**
   * getTokenPrediction
   * Predict the next token or token sequence
   * (agnostic)
   *
   * Uses n-gram and wordToVec8.
   */

  const getTokenPrediction = token => {
    // lookup token in model

    const rankedTokens = Object.keys(
      lookup(
        token.replace(
          token.charAt(0),
          token.charAt(0).toUpperCase()
        )
      )
    );

    // rank tokens by weight

    rankedTokens.sort((a, b) => a[1] - b[1]);

    const highest = rankedTokens[0];

    const {
      token: singleTokenPrediction,
      rankedTokenList: singleTokenPredictionList,
      wordToVec8Prediction
    } = getSingleTokenPrediction(token);


    // return highest ranked token and a top k sample
    // along with an error message (if any)

    if (highest) {
      return {
        token: highest,
        rankedTokenList: rankedTokens.slice(-RANKING_BATCH_SIZE),
        wordToVec8Prediction
      };
    } else {
      if (singleTokenPrediction) {
        return {
          token: singleTokenPrediction.replace(/\\n/g, ' ').trim(),
          rankedTokenList: singleTokenPredictionList,
          wordToVec8Prediction
        };
      }

      const message = MISSING_NGRAM_ERROR;

      return {
        error: {
          message
        },
        token: null,
        rankedTokenList: [],
        wordToVec8Prediction: null
      };
    }
  };

  /**
   * getTokenSequencePrediction
   * Predict the next sequence of tokens.
   * Designed for words and phrases.
   *
   * Uses n-gram.
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
   *
   * Uses n-gram.
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
   * createEmbedding
   * Bigram-to-Vec8, designed for words
   * and phrases.
   */

  const createEmbedding = bigrams => {
    console.log(NOTIF_CREATING_CONTEXT);

    // Pre-process vector embeddings with initial
    // bigram assumptions

    wordToVec8 = WordToVec8(bigrams);

    // split text into token sequences
    // and do some text formatting

    trainingTokenSequences = trainingText.replace(/\n/g, ' ')
      .replace(MATCH_TERMINATORS, '$1|')
      .split('|')
      .map(toPlainText);

    // create n-grams for known sequences

    const ngrams = trainingTokenSequences.map(sequence => {
      let cursor;

      const words = sequence.split(' ');

      const ngram = words
        .reduce((a, b) => {
          if (typeof(cursor) === 'object') {
            cursor = cursor[b] = {};
          } else {
            cursor = a[b] = {};
          }

          return a;
        }, {});

        return ngram;
    });

    // deep merge all token sequences into
    // a single hierarchy (for speed and convenience)

    const bigramCollection = (
      chunkArray(ngrams, PARAMETER_CHUNK_SIZE)
    );

    for (const bigram of bigramCollection) {
      Context.trie = merge(Context.trie, ...bigram);
    }

    console.log('Done.');
  };

  /**
   * lookup
   * Look up n-gram by token sequence
   */

  const lookup = sequence => (
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
    // load data

    const { name, files } = dataset;

    const startTime = Date.now();

    console.log(NOTIF_TRAINING);

    trainingText = await combineDocuments(files);

    // tokenize

    trainingTokens = trainingText.split(' ');

    const tokens = [...trainingTokens];
    const bigrams = {};

    tokens.forEach((token, index) => {
      // End statement on punctuation

      if (token.match(MATCH_PUNCTUATION)) {
        console.log(NOTIF_END_OF_STATEMENT);

        return;
      }

      // Skip unparsable tokens

      if (MATCH_NON_ALPHANUMERIC.test(token)) {
        console.log(NOTIF_UNKNOWN_TOKEN);

        return;
      }

      let nextToken = tokens[index + 1];

      // Ensure next token exists

      if (!nextToken) {
        console.log(NOTIF_END_OF_DATA);

        return;
      }

      if (!bigrams[token]?.[nextToken]) {
        bigrams[token] = {
          ...bigrams[token],

          [nextToken]: 0
        };
      }

      bigrams[token][nextToken]++;

      console.log(`Token "${nextToken}" ranked: ${bigrams[token][nextToken]} (when following ${token}).`);
    });

    // save weights to file

    const bigramsPath = `${__root}/training/bigrams/${name}.json`;

    await fs.writeFile(
      bigramsPath, JSON.stringify(Context.bigrams)
    );

    console.log(`Wrote to file: ${bigramsPath}.`);

    console.log(
      `Training completed in ${(Date.now() - startTime) / 1000} seconds. The bigrams for this data is located at "${bigramsPath}".`
    );

    // keep a reference in memory for some NLP
    // tasks

    Context.bigrams = bigrams;

    // create vector embeddings from weights

    createEmbedding(bigrams);
  };

  /**
   * ingest
   * Provide new document text
   */

  const ingest = text => {
    trainingText = text;
    trainingTokens = trainingText.split(' ');
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
    // Utilities

    ingest,
    train,
    createEmbedding,

    // N-gram prediction

    getSingleTokenPrediction,
    getTokenPrediction,
    getTokenSequencePrediction,
    getCompletions,

    // wordToVec8 prediction

    ...wordToVec8
  };
};
