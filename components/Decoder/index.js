const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;

const { merge } = require('lodash');

const { combineDocuments } = require('../../utils');

const MATCH_PUNCTUATION = new RegExp(/[.,\/#!$%?“”\^&\*;:{}=\_`~()]/g);
const MISSING_EMBEDDING_ERROR = 'Failed to look up embedding.';
const MAX_RESPONSE_LENGTH = 240;
const RANKING_BATCH_SIZE = 10;

module.exports = () => {
  let trainingText = '';
  let trainingTokens = [];
  let trainingTokenSequences = [];
  let embedding = {};
  let model;

  /**
   * getSingleTokenPrediction
   * Predict the next token (word, pixel, etc.)
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
   * Predict the next sequence of tokens (phrase,
   * gradient, etc.)
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
   * of alternatives
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
   * phrases. For pixels and gradients (etc.), would need
   * to extend/replace this in a more image-focused model
   */

  const createEmbedding = updatedEmbedding => {
    console.log('Creating model embedding...');

    embedding = updatedEmbedding;

    // split text into token sequences

    trainingTokenSequences = trainingText
      .replace(/([.?!])\s*(?=[A-Z])/g, '$1|')
      .split('|')
      .map(sequence => sequence
        .replace(
          sequence.charAt(0),
          sequence.charAt(0).toUpperCase()
        ));

    // create embeddings for known sequences

    const sequenceEmbeddings = trainingTokenSequences.map(sequence => {
      let cursor;

      // define the end of a sequence by newlines and spaces

      const sequenceEmbedding = sequence
        .replace(/\n/g, '')
        .split(' ')
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

    model = merge(...sequenceEmbeddings);

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
   * Sanitize and rank tokens then create embeddings
   * and save files a needed. Designed for words and
   * phrases. For pixels and gradients (etc.), would
   * need to extend/replace this in a more image-
   * focused model
   */

  const train = async dataset => {
    const { name, files } = dataset;

    const startTime = Date.now();

    console.log('Training...');

    trainingText = await combineDocuments(files);

    trainingTokens = trainingText.split(' ');

    const tokens = [...trainingTokens];

    tokens.map((token, index) => {
      // End statement on punctuation

      if (token.match(MATCH_PUNCTUATION)) {
        console.log('End of statement.');

        return;
      }

      // Skip unparsable tokens

      if (/[^a-zA-Z0-9]/.test(token)) {
        console.log('Skipping unrecognized token.');

        return;
      }

      let nextToken = tokens[index + 1];

      // Ensure next token exists

      if (!nextToken) {
        console.log('End of training data.');

        return;
      }

      const nextTokenStart = nextToken[0];

      // End statement on capital letter

      if (/^\p{Lu}/u.test(nextTokenStart)) {
        console.log('End of statement.');

        return;
      }

      // Sanitize nextToken

      nextToken = nextToken.replace(MATCH_PUNCTUATION, '');

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

  // Decoder API

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
