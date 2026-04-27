const { dirname } = require('path');
const __root = dirname(require.main.filename);
const fs = require('fs').promises;
const zlib = require('zlib');

const FORMAT_ERROR = 'Invalid file format.';

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

   combineImages: async images => {
     let allPixels = [];

     for (const imageName of images) {
       const buffer = await fs.readFile(`${__root}/training/images/${imageName}.png`);

       // 1. Validating image

       if (buffer.readUInt32BE(0) !== 0x89504E47) continue;

       let pos = 8;
       let width, height;
       let idatChunks = [];

       // 2. Walking the chunks

       while (pos < buffer.length) {
         const length = buffer.readUInt32BE(pos);
         const type = buffer.toString('ascii', pos + 4, pos + 8);

         if (type === 'IHDR') {
           width = buffer.readUInt32BE(pos + 8);
           height = buffer.readUInt32BE(pos + 12);
         } else if (type === 'IDAT') {
           idatChunks.push(buffer.slice(pos + 8, pos + 8 + length));
         } else if (type === 'IEND') break;

         pos += length + 12;
       }

       // 3. Decompressing the Pixel Stream

       const rawData = zlib.inflateSync(Buffer.concat(idatChunks));

       // 4. Scanline parsing (skipping the filter byte)

       const bytesPerPixel = 4;
       const stride = (width * bytesPerPixel) + 1;

       for (let y = 0; y < height; y++) {
         for (let x = 0; x < width; x++) {
           const idx = (y * stride) + 1 + (x * bytesPerPixel);

           const r = rawData[idx].toString(16).padStart(2, '0');
           const g = rawData[idx + 1].toString(16).padStart(2, '0');
           const b = rawData[idx + 2].toString(16).padStart(2, '0');

           allPixels.push(`#${r}${g}${b}`);
         }
       }
     }

     return allPixels;
   },

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
