const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;

const { merge } = require('lodash');
const dotenv = require('dotenv');

const {
  combineDocuments,
  isLowerCase
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
const MISSING_EMBEDDING_ERROR = 'Failed to look up embedding.';
const NOTIF_TRAINING = 'Training...';
const NOTIF_END_OF_STATEMENT = 'End of sequence.';
const NOTIF_UNKNOWN_TOKEN = 'Skipping unrecognized token.';
const NOTIF_END_OF_DATA = 'End of training data.';
const NOTIF_CREATING_EMBEDDING = 'Creating model embedding...';
const PARAMETER_CHUNK_SIZE = 50000;

// Generator function to chunk arrays
// Use with `PARAMETER_CHUNK_SIZE` for models
// with many parameters to avoid memory errors

function* chunkArray (array, chunkSize) {
  for (let i = 0; i < array.length; i += chunkSize) {
    yield array.slice(i, i + chunkSize);
  }
}

module.exports = () => {
  let trainingText = '';
  let trainingTokens = [];
  let trainingTokenSequences = [];
  let embedding = {};
  let model = {};

  /**
   * getSingleTokenPrediction
   * Predict the next token (agnostic)
   */

  const getSingleTokenPrediction = token => {

    // lookup token in embeddings

    const tokens = embedding[token];

    if (tokens) {
      const rankedTokens = [];

      for (const rankedToken in tokens) {
        rankedTokens.push([rankedToken, tokens[rankedToken]]);
      }

      // rank tokens by weight

      rankedTokens.sort((a, b) => a[1] - b[1]);

      const highest = rankedTokens[0];
      const highestRankedToken = highest[0];

      // return the highest ranked token and a top k sample

      if (highestRankedToken) {
        return {
          token: highestRankedToken,
          rankedTokenList: rankedTokens
            .map(([_token]) => _token)
            .slice(-RANKING_BATCH_SIZE)
        };
      }
    }

    const message = MISSING_EMBEDDING_ERROR;

    return {
      error: {
        message
      },
      token: null,
      rankedTokenList: []
    };
  };

  /**
   * getTokenPrediction
   * Predict the next token or token sequence
   * (agnostic)
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

    // return highest ranked token and a top k sample
    // along with an error message (if any)

    if (highest) {
      return {
        token: highest,
        rankedTokenList: rankedTokens.slice(-RANKING_BATCH_SIZE)
      };
    } else {
      const {
        token: singleTokenPrediction,
        rankedTokenList: singleTokenPredictionList
      } = getSingleTokenPrediction(token);

      if (singleTokenPrediction) {
        return {
          token: singleTokenPrediction.replace(/\\n/g, '').trim(),
          rankedTokenList: singleTokenPredictionList
        };
      }

      const message = MISSING_EMBEDDING_ERROR;

      return {
        error: {
          message
        },
        token: null,
        rankedTokenList: []
      };
    }
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
          .replace(/\\n/g, '')
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
   * createEmbedding
   * Create a model embedding. Designed for words and
   * phrases.
   */

  const createEmbedding = updatedEmbedding => {
    console.log(NOTIF_CREATING_EMBEDDING);

    embedding = updatedEmbedding;

    // split text into token sequences
    // and do some text formatting

    trainingTokenSequences = trainingText.replace(/\n/g, '')
      .replace(MATCH_TERMINATORS, '$1|')
      .split('|')
      .map(toPlainText);

    // create embeddings for known sequences

    const sequenceEmbeddings = trainingTokenSequences.map(sequence => {
      let cursor;

      const words = sequence.split(' ');

      const sequenceEmbedding = words
        .map((word, wordIndex) => {
          const nextWord = words[wordIndex + 1];

          if (!nextWord) return word;

          return (
            !isLowerCase(word.charAt(0)) &&
            !isLowerCase(nextWord.charAt(0))
          ) ? `${word} ${nextWord}`: word
        })
        .reduce((a, b) => {
          if (typeof(cursor) === 'object') {
            cursor = cursor[b] = {};
          } else {
            cursor = a[b] = {};
          }

          return a;
        }, {});

        return sequenceEmbedding;
    });

    // deep merge all token sequences into
    // a single hierarchy (for speed and convenience)

    const embeddingsCollection = [
      ...chunkArray(sequenceEmbeddings, PARAMETER_CHUNK_SIZE)
    ];

    for (const embeddings of embeddingsCollection) {
      const modelFragment = merge(...embeddings);

      for (const key in Object.keys(modelFragment)) {
        model[key] = modelFragment[key];
      }
    }

    console.log('Done.');
  };

  /**
   * lookup
   * Look up embedding sequence
   */

  const lookup = sequence => (
    sequence
      .split(/ /)
      .reduce((a, b) => a?.[b], model) || {}
  );

  /**
   * train
   * Rank tokens then create embeddings and save
   * files as needed. Designed for words and phrases.
   */

  const train = async dataset => {
    const { name, files } = dataset;

    const startTime = Date.now();

    console.log(NOTIF_TRAINING);

    trainingText = await combineDocuments(files);

    trainingTokens = trainingText.split(' ');

    const tokens = [...trainingTokens];

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

      if (!embedding[token]?.[nextToken]) {
        embedding[token] = {
          ...embedding[token],

          [nextToken]: 0
        };
      }

      embedding[token][nextToken]++;

      console.log(`Token "${nextToken}" ranked: ${embedding[token][nextToken]} (when following ${token}).`);
    });

    const embeddingPath = `${__root}/training/embeddings/${name}.json`;

    await fs.writeFile(
      embeddingPath, JSON.stringify(embedding)
    );

    console.log(`Wrote to file: ${embeddingPath}.`);

    console.log(
      `Training completed in ${(Date.now() - startTime) / 1000} seconds. The embedding for this data is located at "${embeddingPath}".`
    );

    createEmbedding(embedding);
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
    .replace(MATCH_NEW_LINES, '')
    .replace(FORMAT_PLAIN_TEXT[0], ' ')
    .replace(FORMAT_PLAIN_TEXT[1], ' ')
    .replace(FORMAT_PLAIN_TEXT[2], ' ')
    .replace(FORMAT_PLAIN_TEXT[3], ' ')
    .replace(FORMAT_PLAIN_TEXT[4], ' ');

  // Transformer API

  return {
    getSingleTokenPrediction,
    getTokenPrediction,
    getTokenSequencePrediction,
    getCompletions,
    createEmbedding,
    train,
    ingest
  };
};
