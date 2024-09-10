const posTags = require('./pos');
const { alphabet, vowels } = require('./letters');

const tokenizeSequence = sequence => sequence
  .trim()
  .replace(/[\p{P}$+<=>^`|~]/gu, '')
  .split(' ');

module.exports = bigrams => {

  /**
   * N-gram
   *
   * getTokenPrevalence
   * getMaxTokenPrevalence
   * getMostFrequentNextToken
   * getMostFrequentNextTokenValue
   * getMaxTokenFrequency
   * getMaxFrequency
   */

  const getTokenPrevalence = token => bigrams.hasOwnProperty(token)
    ? Object.keys(bigrams[token]).length
    : 0;

  const getMaxTokenPrevalence = () => Object.keys(bigrams)
    .map(key => getTokenPrevalence(key))
    .sort((a, b) => a > b ? 1 : -1)
    .pop();

  const getMostFrequentNextToken = token => bigrams.hasOwnProperty(token) && Object
    .keys(bigrams[token])
    .sort((a, b) => bigrams[token][a] - bigrams[token][b])
    .pop();

  const getMostFrequentNextTokenValue = token => bigrams.hasOwnProperty(token)
    ? bigrams[token][getMostFrequentNextToken(token)]
    : -1;

  const getMaxTokenFrequency = token => bigrams.hasOwnProperty(token)
    ? Object
      .values(bigrams[token])
      .sort((a, b) => a > b ? 1 : -1)
      .pop()
    : 0;

  const getMaxFrequency = () => Object
    .keys(bigrams)
    .map(getMaxTokenFrequency)
    .sort((a, b) => a > b ? 1 : -1)
    .pop();

  /**
   * Parts-of-speech
   *
   * getFirstVowel
   * getLinearPOS
   */

  const getVowels = string => {
    let vowels = [];

    const isVowel = character => (
      vowels.indexOf(character.toLowerCase()) !== -1
    );

    for (const character of string) {
      if (isVowel(character)) {
        vowels.push(character);
      }
    }

    return vowels;
  };

  const getFirstVowel = string => {
    const isVowel = character => (
      vowels.indexOf(character.toLowerCase()) !== -1
    );

    for (const character of string) {
      if (isVowel(character)) {
        return character;
      }
    }
  };

  const getLinearPOS = tag => posTags.indexOf(tag);

  /**
   * Embeddings
   *
   * toVecs
   * getSum
   * getDotProduct
   * getSimilarityIndexByToken
   */

  const toVecs = (sequence = '') => {
    const embeddings = [];

    // tokenize

    const tokens = tokenizeSequence(sequence);

    // compute & normalize

    for (const token of tokens) {
      if (!bigrams[token]) continue;

      // pos tag

      const posTag = 'RB';
      const posLinear = getLinearPOS(posTag);
      const lastLetter = alphabet.indexOf(token.charAt(token.length - 1));
      const lastVowel = alphabet.indexOf(getFirstVowel(token.split('').reverse().join('')));
      const vowelCount = getVowels(token).length << 0;
      const firstLetter = alphabet.indexOf(token.charAt(0));
      const firstVowel = alphabet.indexOf(getFirstVowel(token));
      const prevalence = getTokenPrevalence(token);
      const frequency = getMostFrequentNextTokenValue(token);
      const { length } = token;

      // Embeddings

      const prenormalized = [
        posLinear,
        lastLetter,
        lastVowel,
        vowelCount,
        firstLetter,
        firstVowel,
        prevalence,
        frequency,
        length
      ];

      const maxPosLinear = posTags.length - 1;
      const maxLastLetter = 25;
      const maxLastVowel = 25;
      const maxVowelCount = 10;
      const maxFirstLetter = 25;
      const maxFirstVowel = 25;
      const maxPrevalence = getMaxTokenPrevalence();
      const maxFrequency = getMaxFrequency();
      const maxLength = 20;

      const maximums = [
        maxPosLinear,
        maxLastLetter,
        maxLastVowel,
        maxVowelCount,
        maxFirstLetter,
        maxFirstVowel,
        maxFrequency,
        maxPrevalence,
        maxLength
      ];

      const embedding = (
        prenormalized.map((value, index) => (
          Math.max(0, Math.min(1, value / maximums[index])))
        )
      );

      embeddings.push(embedding);
    }

    return embeddings;
  };

  const getSum = vector => (
    vector.reduce((a, b) => a + b)
  );

  const getDotProduct = (a, b) => (
    a.map((_, index) => a[index] * b[index]).reduce((m, n) => m + n)
  );

  const getSimilarityIndexByToken = (token, sequence) => {
    const embeddings = toVecs(sequence);
    const tokenIndex = tokenizeSequence(sequence).indexOf(token);

    const tokenEmbedding = embeddings[tokenIndex];
    const products = {};

    embeddings.forEach((embedding, index) => {
      if (index === tokenIndex) return;

      const dotProduct = getDotProduct(tokenEmbedding, embedding);
      const tokenComparison = tokenizeSequence(sequence)[index];

      products[tokenComparison] = dotProduct;
    });

    const similarityIndex = Object.keys(products).sort((a, b) => (
      products[a] > products[b] ? 1 : -1
    ));

    return similarityIndex;
  };

  return {
    // N-gram prediction

    getTokenPrevalence,
    getMaxTokenPrevalence,
    getMostFrequentNextToken,
    getMostFrequentNextTokenValue,
    getMaxTokenFrequency,
    getMaxFrequency,

    // Parts-of-speech analysis

    getVowels,
    getFirstVowel,
    getLinearPOS,

    // Vector embeddings

    toVecs,
    getSum,
    getDotProduct,
    getSimilarityIndexByToken
  }
};
