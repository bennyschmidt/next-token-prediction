const { dirname } = require('path');
const __root = dirname(require.main.filename);

const Decoder = require('../../components/Decoder');

const {
  combineDocuments,
  fetchEmbeddingByName
} = require('../../utils');

const DEFAULT_DATASET = require(`${__root}/training/datasets/OpenSourceBooks`);

const NEW_DATASET_NAME = 'New Dataset';

/**
 * Create a language model from a
 * dataset or specify a list of files
 */

module.exports = async ({
  dataset,
  files,
  bootstrap = false
} = {}) => {
  let datasetName = NEW_DATASET_NAME;
  let trainingData = null;
  let decoder;

  /**
   * init
   * Train, create embeddings, and initialize the
   * model API for use
   */

  const init = async () => {
    /**
     * If bootstrap is true, start the LM
     * with default training data
     */

    if (bootstrap) {
      files = DEFAULT_DATASET.files;
      datasetName = DEFAULT_DATASET.name;
    }

    /**
     * If a dataset is provided, create an embedding
     * for the language model to use
     */

    if (dataset?.name) {

      // concatenate and store all reference text

      const text = await combineDocuments(dataset.files);

      // load the corresponding embedding file

      const embedding = await fetchEmbeddingByName(dataset.name);

      // build training data object

      trainingData = {
        text,
        embedding
      };
    }

    /**
     * Instantiate a text decoder either from
     * existing training data or by providing files
     */

    if (trainingData) {

      // Skips initial extraction and training
      // just instantiates the model with embeddings

      decoder = fromTrainingData(trainingData);
    } else {

      // Performs an intensive training operation
      // using provided files, creating an elaborate
      // model structure and corresponding values
      // (embeddings)

      decoder = await fromFiles(files);
    }
  };

  /**
   * fromTrainingData
   * Create a new decoder model with new text
   * and embedding
   */

  const fromTrainingData = ({
    text,
    embedding: updatedEmbedding
  }) => {
    const textDecoder = Decoder();

    textDecoder.ingest(text);
    textDecoder.createEmbedding(updatedEmbedding);

    return textDecoder;
  };

  /**
   * fromFiles
   * Create a new decoder model from .txt files
   * (if none, default to the bootstrap)
   */

  const fromFiles = async files => {
    const textDecoder = Decoder();

    if (files) {
      await textDecoder.train({
        name: datasetName,
        files
      });
    } else {
      await textDecoder.train({
        name: DEFAULT_DATASET.name,
        files: DEFAULT_DATASET.files
      });
    }

    return textDecoder;
  };

  /**
   * complete
   * Pass query to the decoder and return
   * the highest-ranked completion
   */

  const complete = query => decoder.getCompletions(query).completion;

  // Language API (extends Decoder)

  await init();

  return {
    ...decoder,

    fromTrainingData,
    fromFiles,
    complete
  };
};
