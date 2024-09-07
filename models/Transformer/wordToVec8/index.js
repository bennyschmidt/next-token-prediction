const posSpecificityList = require('./pos');
const { alphabet, vowels } = require('./letters');

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
   * getPOSSpecificity
   */

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

  const getPOSSpecificity = tag => posSpecificityList.indexOf(tag);

  /**
   * Embeddings
   *
   * toVecs
   * getSum
   * getDotProduct
   * getSimilarityIndexByToken
   */

  const toVecs = (sequence = '') => {
    const embeddings = {};

    // tokenize

    const tokens = sequence
      .toLowerCase()
      .trim()
      .replace(/[\p{P}$+<=>^`|~]/gu, '')
      .split(' ');

    for (const token of tokens) {
      if (!bigrams[token]) continue;

      // pos tag

      const posTag = 'RB';

      // frequency

      const frequency = getMostFrequentNextTokenValue(token);

      // prevalence

      const prevalence = getTokenPrevalence(token);

      // specificity

      const specificity = getPOSSpecificity(posTag);

      // length

      const { length } = token;

      // first letter

      const firstLetter = alphabet.indexOf(token.charAt(0));

      // last letter

      const lastLetter = alphabet.indexOf(token.charAt(token.length - 1));

      // first vowel

      const firstVowel = alphabet.indexOf(getFirstVowel(token));

      // last vowel

      const lastVowel = alphabet.indexOf(getFirstVowel(token.split('').reverse().join('')));

      // Embeddings

      const prenormalized = [
        frequency,
        prevalence,
        specificity,
        length,
        firstLetter,
        lastLetter,
        firstVowel,
        lastVowel
      ];

      const maxFrequency = getMaxFrequency();
      const maxPrevalence = getMaxTokenPrevalence();
      const maxSpecificity = posSpecificityList.length - 1;
      const maxLength = 20;
      const maxFirstLetter = 25;
      const maxLastLetter = 25;
      const maxFirstVowel = 25;
      const maxLastVowel = 25;

      const maximums = [
        maxFrequency,
        maxPrevalence,
        maxSpecificity,
        maxLength,
        maxFirstLetter,
        maxLastLetter,
        maxFirstVowel,
        maxLastVowel
      ];

      const embedding = (
        prenormalized.map((value, index) => (
          Math.max(0, Math.min(1, value / maximums[index])))
        )
      );

      embeddings[token] = embedding;
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

    const products = {};
    const tokenEmbedding = embeddings[token];

    for (const tokenComparison of Object.keys(embeddings)) {
      const embedding = embeddings[tokenComparison];
      const dotProduct = getDotProduct(tokenEmbedding, embedding);

      products[tokenComparison] = dotProduct;
    }

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

    getFirstVowel,
    getPOSSpecificity,

    // Vector embeddings

    toVecs,
    getSum,
    getDotProduct,
    getSimilarityIndexByToken
  }
};
