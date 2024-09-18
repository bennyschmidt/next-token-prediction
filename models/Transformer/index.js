const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;

const { merge } = require('lodash');
const dotenv = require('dotenv');

const Vector = require('../../components/Vector');

const {
  alphabet,
  combineDocuments,
  getPartsOfSpeech,
  partsOfSpeech,
  suffixes,
  tokenize
} = require('../../utils');

dotenv.config();

const {
  PARAMETER_CHUNK_SIZE = 50000,
  RANKING_BATCH_SIZE = 50,
  MAX_RESPONSE_LENGTH = 240,
  VARIANCE = 0
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
const DONE = 'Done.';

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
   * ngramSearch
   * Look up n-gram by token sequence.
   */

  const ngramSearch = input => (
    input
      .split(/ /)
      .reduce((a, b) => a?.[b], Context.trie) || {}
  );

  /**
   * embeddingSearch
   * Look up embedding by token.
   */

  const embeddingSearch = (prevToken, token) => {
    const [second, first] = tokenize(`${prevToken} ${token}`).reverse();

    if (!first || !Context.embeddings[first]) {
      return Vector.fromNull();
    }

    return Context.embeddings[first][second];
  };

  /**
   * dotProduct
   * Dot product of two vectors.
   */

  const dotProduct = (
    vectorA = Vector.fromNull(),
    vectorB = Vector.fromNull()
  ) => (
    vectorA
      .map((_, i) => vectorA[i] * vectorB[i])
      .reduce((m, n) => m + n)
  );

  /**
   * getSimilarToken
   * Get a similar token.
   */

  const getSimilarToken = (prevToken, token) => {
    const tokenEmbedding = embeddingSearch(prevToken, token);

    const entries = Object.keys(Context.embeddings);
    const result = [];

    for (const entry of entries) {
      const tokens = Object.keys(Context.embeddings[entry]);

      for (const token of tokens) {
        const embedding = Context.embeddings[entry][token];

        if (embedding) {
          result.push({
            token,
            embedding,
            similarity: dotProduct(
              tokenEmbedding,
              embedding
            )
          });
        }
      }
    }

    const rankedTokenList = result
      .sort((a, b) => b.similarity - a.similarity)
      .slice(-RANKING_BATCH_SIZE)
      .filter(Boolean)
      .map(rankedResult => rankedResult.token);

    return {
      token: rankedTokenList[rankedTokenList.length - 1],
      rankedTokenList
    };
  };

  /**
   * getTokenPrediction
   * Predict the next token or token sequence
   * (agnostic).
   */

  const getTokenPrediction = token => {
    if (!token) {
      return {
        token: '',
        rankedTokenList: []
      };
    }

    // ngram search

    const rankedTokens = Object.keys(
      ngramSearch(
        token.replace(
          token.charAt(0),
          token.charAt(0).toUpperCase()
        )
      )
    );

    const highestRankedToken = rankedTokens[rankedTokens.length - 1];

    if (highestRankedToken) {
      if (VARIANCE > 0) {
        const {
          token: similarToken,
          rankedTokenList
        } = getSimilarToken(token, highestRankedToken);

        if (similarToken) {
          return {
            token: similarToken,
            rankedTokenList
          };
        }
      }

      return {
        token: highestRankedToken,
        rankedTokenList: rankedTokens.slice(-RANKING_BATCH_SIZE)
      };
    }

    const message = MISSING_NGRAM_ERROR;

    return {
      error: {
        message
      },
      token: '',
      rankedTokenList: []
    };
  };

  /**
   * getTokenSequencePrediction
   * Predict the next sequence of tokens.
   * Designed for words and phrases.
   */

  const getTokenSequencePrediction = (input, sequenceLength = 2) => {
    const sequence = [];

    let result = input;

    // get top k sample from getTokenPrediction

    const {
      rankedTokenList: keyPredictions
    } = getTokenPrediction(input);

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
   * Create model components in memory.
   */

  const createContext = embeddings => {
    Context.embeddings = embeddings;

    // Store current context in memory as a trie

    console.log(NOTIF_CREATING_CONTEXT);

    // split sequences

    Context.sequences = trainingText.replace(/\n/g, ' ')
      .replace(MATCH_TERMINATORS, '$1|')
      .split('|')
      .map(toPlainText);

    // create n-grams of all sequences

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

    const ngramMap = (
      chunkArray(ngrams, PARAMETER_CHUNK_SIZE)
    );

    // keep reference in memory

    for (const ngram of ngramMap) {
      Context.trie = merge(Context.trie, ...ngram);
    }

    console.log(DONE);
  };

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

    // 1. Tokenize

    Context.trainingTokens = tokenize(trainingText);

    const tokens = [...Context.trainingTokens];
    const embeddings = {};

    let maxNextWordFrequency = 0;
    let nextWordFrequencyIndexStart = 0;

    // 2. Analyze

    for (let index = 0; index < tokens.length; index++) {
      const token = tokens[index];

      if (!token) continue;

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

      const nextToken = tokens[index + 1];

      // Ensure next token exists

      if (!nextToken) {
        console.log(NOTIF_END_OF_DATA);

        continue;
      }

      if (!embeddings[token]) {
        embeddings[token] = {};
      }

      if (!embeddings[token][nextToken]) {
        embeddings[token][nextToken] = Vector.fromNull();
      }

      /**
       * Training metrics
       * I. Composition
       *
       * Distribution of 66 alpha-numeric (and other)
       * symbols.
       */

      const letters = alphabet.split('');

      for (const letter of letters) {
        if (nextToken.includes(letter)) {
          const letterIndex = letters.indexOf(letter);

          embeddings[token][nextToken][letterIndex] = parseFloat(
            nextToken.split('').filter(char => char === letter).length /
            nextToken.length
          );
        }
      }

      /**
       * Training metrics
       * II. Parts-of-speech
       *
       * Distribution of 36 parts-of-speech types.
       */

      const posIndexStart = alphabet.length;

      const [tag] = getPartsOfSpeech(nextToken.toLowerCase());

      if (tag?.pos) {
        const tagIndex = Object.keys(partsOfSpeech.indexOf(tag.pos));

        embeddings[token][nextToken][posIndexStart + tagIndex] = 1;
      }

      /**
       * Training metrics
       * III. Prevalence
       *
       * Token prevalence (in the dataset).
       */

      const prevalenceIndexStart = (
        posIndexStart +
        partsOfSpeech.length
      );

      // Prevalence

      embeddings[token][nextToken][prevalenceIndexStart] = parseFloat(tokens
        .filter(_token => _token === nextToken)
        .length / tokens.length);

      /**
       * Training metrics
       * IV. Word suffixes
       *
       * Distribution of 37 common rhyme suffixes.
       */

      const suffixesIndexStart = prevalenceIndexStart + 1;

      for (const suffix of suffixes) {
        const suffixIndex = suffixes.indexOf(suffix);

        embeddings[token][nextToken][suffixesIndexStart + suffixIndex] = (
          new RegExp(suffix).test(nextToken.slice(-suffix.length))
            ? 1
            : 0
        );
      }

      /**
       * Training metrics
       * V. Next-word frequency
       *
       * Token occurrence count (as next token).
       */

      nextWordFrequencyIndexStart = suffixesIndexStart + suffixes.length;

      const nextWordFrequency = (
        ++embeddings[token][nextToken][nextWordFrequencyIndexStart]
      );

      console.log(`Token "${nextToken}" ranked: ${nextWordFrequency} (when following "${token}").`);

      // TODO: Expand vocabulary

      const isStopWord = /i|me|my|myself|we|our|ours|ourselves|you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves|what|which|who|whom|this|that|these|those|am|is|are|was|were|be|been|being|have|has|had|having|do|does|did|doing|a|an|the|and|but|if|or|because|as|until|while|of|at|by|for|with|about|against|between|into|through|during|before|after|above|below|to|from|up|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|s|t|can|will|just|don|should|now/
        .toString()
        .match(new RegExp(nextToken));

      if (!isStopWord && (nextWordFrequency > maxNextWordFrequency)) {
        console.log(`Set new highest embedding value (of token "${nextToken}").`);

        maxNextWordFrequency = nextWordFrequency;
      }

      /**
       * Training metrics
       * VI. Vulgar
       *
       * Slang, slurs, profanity, etc.
       */

      const vulgarIndexStart = nextWordFrequencyIndexStart + 1;

      // TODO: Vulgarity

      embeddings[token][nextToken][vulgarIndexStart] = 0;

      /**
       * Training metrics
       * VII. Style
       *
       * Extend embeddings with stylistic
       * features.
       */

      // Pirate
      // TODO: Expand vocabulary

      const isPirate = /ahoy|arrr|matey|blimey|scallywag/.test(
        new RegExp(`${token}|${nextToken}`)
      ) || (token === 'me' && tag.pos.match('NN'));

      embeddings[token][nextToken][vulgarIndexStart + 1] = isPirate
        ? 1
        : 0;

      // Victorian
      // TODO: Expand vocabulary

      const isVictorian = /abeyance|ado|blunderbuss|carriage|chambre|corset|dandy|dote|doth|esquire|futile|grand|hath|hence|lively|nonesuch|thee|thou|thy|vestibule|wonderful/.test(
        new RegExp(`${token}|${nextToken}`)
      );

      embeddings[token][nextToken][vulgarIndexStart + 2] = isVictorian
        ? 1
        : 0;

      console.log(`Updated word embedding for "${nextToken}".`);
    }

    // 3. Normalize

    for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
      const computedToken = tokens[tokenIndex];
      const nextComputedToken = tokens[tokenIndex + 1];
      const value = embeddings[computedToken]?.[nextComputedToken]?.[nextWordFrequencyIndexStart];

      if (value) {
        const normalizedValue = Math.min(
          1,
          parseFloat(value / maxNextWordFrequency)
        );

        console.log(`Adjusted word embedding for "${nextComputedToken}".`);

        embeddings[computedToken][nextComputedToken][nextWordFrequencyIndexStart] = parseFloat(
          normalizedValue
        );
      }
    }

    // save embedding to file

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
   * Provide new document text.
   */

  const ingest = text => {
    trainingText = text;
    Context.trainingTokens = trainingText.split(' ');
  };

  /**
   * toPlainText
   * Transform text to a plain format.Capitalizes
   * the first token of sequences, removing certain
   * special characters, new lines, etc.
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
