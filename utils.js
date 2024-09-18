const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;

const Tagger = require('wink-pos-tagger');

// See: https://winkjs.org/wink-pos-tagger/

const FORMAT_ERROR = 'Invalid file format.';

const tagger = Tagger();

const partsOfSpeech = [
  'CC',
  'CD',
  'DT',
  'EX',
  'FW',
  'IN',
  'JJ',
  'JJR',
  'JJS',
  'LS',
  'MD',
  'NN',
  'NNS',
  'NNP',
  'NNPS',
  'PDT',
  'POS',
  'PRP',
  'PRP$',
  'RB',
  'RBR',
  'RBS',
  'RP',
  'SYM',
  'TO',
  'UH',
  'VB',
  'VBD',
  'VBG',
  'VBN',
  'VBP',
  'VBZ',
  'WDT',
  'WP',
  'WP$',
  'WRB'
];

const suffixes = [
  'ack',
  'ail',
  'ain',
  'ake',
  'ale',
  'ame',
  'an',
  'ank',
  'ap',
  'are',
  'ash',
  'at',
  'ate',
  'aw',
  'ay',
  'eat',
  'ell',
  'est',
  'ice',
  'ick',
  'ide',
  'ight',
  'ill',
  'in',
  'ine',
  'ing',
  'ink',
  'ip',
  'it',
  'ock',
  'oke',
  'op',
  'ore',
  'ot',
  'ug',
  'ump',
  'unk'
];

module.exports = {
  alphabet: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#$%&',
  vowels: 'aeiou',
  y: 'y',
  x: 'x',
  w: 'w',
  k: 'k',
  j: 'j',

  partsOfSpeech,
  suffixes,

  combineDocuments: async documents => {
    let text = '';

    for (const document of documents) {
      const sourceFile = await fs.readFile(
        `${__root}/training/documents/${document}.txt`
      );

      if (!sourceFile?.toString) {
        throw new Error(FORMAT_ERROR);
      }

      const source = sourceFile.toString().trim();

      text += `\n${source}`;
    }

    return text;
  },

  fetchEmbeddings: async name => {
    const file = await fs.readFile(
      `${__root}/training/embeddings/${name}.json`
    );

    const embeddings = JSON.parse(file.toString());

    return embeddings;
  },

  getPartsOfSpeech: text => (
    tagger.tagSentence(text)
  ),

  isLowerCase: letter => (
    letter === letter.toLowerCase() &&
    letter !== letter.toUpperCase()
  ),

  tokenize: input => (
    input
      .trim()
      .replace(/[\p{P}$+<=>^`(\\\n)|~]/gu, ' ')
      .split(' ')
  )
};
