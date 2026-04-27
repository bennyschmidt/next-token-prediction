const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;

const { merge } = require('lodash');
const dotenv = require('dotenv');

const { combineImages } = require('../../utils');

dotenv.config();

const {
  PARAMETER_CHUNK_SIZE = 50000,
  RANKING_BATCH_SIZE = 50,
  MAX_RESPONSE_LENGTH = 1024
} = process.env;

const DONE = 'Done.';
const NOTIF_TRAINING = 'Training Gradient...';
const NOTIF_CREATING_CONTEXT = 'Creating pixel context...';

/**
 * toCommonHex
 * Mathematically snaps a color to a 24-bit hex value.
 */

const toCommonHex = (hex) => {
  if (!hex) return '#000000';

  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);

  const steps = 10;
  const factor = 255 / (steps - 1);

  const snap = (val) => {
    const quantized = Math.round(val / factor) * factor;

    return Math.min(255, Math.round(quantized)).toString(16).padStart(2, '0');
  };

  return `#${snap(r)}${snap(g)}${snap(b)}`;
};

const Context = {
  trie: {},
  sequences: []
};

module.exports = () => {
  let trainingData = [];

  const ngramSearch = input => (
    input.split(' ').reduce((a, b) => a?.[b], Context.trie) || {}
  );

  const getPixelPrediction = pixel => {
    if (!pixel) return { token: '', rankedTokenList: [] };
    const rankedTokens = Object.keys(ngramSearch(pixel));
    const highestRanked = rankedTokens[rankedTokens.length - 1];

    return {
      token: highestRanked || '',
      rankedTokenList: rankedTokens.slice(-RANKING_BATCH_SIZE)
    };
  };

  const getPixelSequencePrediction = (input, sequenceLength = 10) => {
    const sequence = [];

    let result = input;

    const { rankedTokenList: keyPredictions } = getPixelPrediction(input);

    for (let i = 0; i < sequenceLength; i++) {
      const { token: prediction } = getPixelPrediction(result.split(' ').pop());

      if (prediction) {
        sequence.push(prediction);
        result += ` ${prediction}`;
      }
    }

    return {
      completion: sequence.join(' '),
      token: sequence[0],
      rankedTokenList: keyPredictions
    };
  };

  const getCompletions = input => {
    const { completion, token, rankedTokenList } = getPixelSequencePrediction(input, MAX_RESPONSE_LENGTH);
    const completions = [completion];

    for (const predictedToken of rankedTokenList) {
      const { completion: prediction } = getPixelSequencePrediction(`${input} ${predictedToken}`, 10);

      completions.push(`${predictedToken} ${prediction}`);
    }

    return { completion, token, rankedTokenList, completions };
  };

  const createContext = () => {
    console.log(NOTIF_CREATING_CONTEXT);

    const ngrams = [trainingData.reduce((acc, hex, i) => {
      if (trainingData[i + 1]) {
        acc[hex] = acc[hex] || {};
        acc[hex][trainingData[i + 1]] = {};
      }

      return acc;
    }, {})];

    Context.trie = merge(Context.trie, ...ngrams);

    console.log(DONE);
  };

  /**
   * train
   * Ranks pixel transitions using a frequency trie.
   */

  const train = async dataset => {
    const { files } = dataset;
    const startTime = Date.now();
    const trie = {};

    console.log(NOTIF_TRAINING);

    const rawPixels = await combineImages(files);

    trainingData = rawPixels.map(toCommonHex);

    for (let i = 0; i < trainingData.length - 1; i++) {
      const current = trainingData[i];
      const next = trainingData[i + 1];

      if (!trie[current]) {
        trie[current] = {};
      }

      trie[current][next] = (trie[current][next] || 0) + 1;
    }

    for (const [pixel, transitions] of Object.entries(trie)) {
      const sortedKeys = Object.keys(transitions).sort((a, b) => {
        return transitions[a] - transitions[b];
      });

      Context.trie[pixel] = {};

      for (const next of sortedKeys) {
        Context.trie[pixel][next] = {};
      }
    }

    console.log(
      `Gradient training completed in ${(Date.now() - startTime) / 1000} seconds.`
    );
  };

  const ingest = pixels => {
    trainingData = pixels.map(toCommonHex);
  };

  return {
    ingest,
    train,
    createContext,
    getPixelPrediction,
    getPixelSequencePrediction,
    getCompletions,
    toCommonHex
  };
};
