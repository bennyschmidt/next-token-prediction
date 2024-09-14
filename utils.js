const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;

const FORMAT_ERROR = 'Invalid file format.';

module.exports = {
  alphabet: 'abcdefghijklmnopqrstuvwxyz',
  vowels: 'aeiou',
  y: 'y',
  x: 'x',
  w: 'w',
  k: 'k',
  j: 'j',

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

  isLowerCase: letter => (
    letter === letter.toLowerCase() &&
    letter !== letter.toUpperCase()
  )
};
