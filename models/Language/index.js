const { dirname } = require('path');
const __root = dirname(require.main.filename);

const Transformer = require('../Transformer');

const {
  combineDocuments,
  fetchNgramByName
} = require('../../utils');

const DEFAULT_DATASET = require(`${__root}/training/datasets/OpenSourceBooks`);

const NEW_DATASET_NAME = 'New Dataset';

/**
 * Create a language model from a
 * dataset or specify a list of files
 */

module.exports = async ({
  name,
  dataset,
  files,
  bootstrap = false
} = {}) => {
  let datasetName = name || NEW_DATASET_NAME;
  let trainingData = null;
  let transformer;

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
      datasetName = dataset.name;

      // concatenate and store all reference text

      const text = await combineDocuments(dataset.files);

      // load the corresponding bigrams

      const bigrams = await fetchNgramByName(dataset.name);

      // build training data object

      trainingData = {
        text,
        bigrams
      };
    }

    /**
     * Instantiate a text transformer either from
     * existing training data or by providing files
     */

    if (trainingData) {

      // Skips initial extraction and training
      // just instantiates the model with bigrams

      transformer = fromTrainingData(trainingData);
    } else {

      // Performs an intensive training operation
      // using provided files, creating an elaborate
      // model structure and corresponding values

      transformer = await fromFiles(files);
    }
  };

  /**
   * fromTrainingData
   * Create a new transformer model with new text
   * and embeddings
   */

  const fromTrainingData = ({
    text,
    bigrams
  }) => {
    const textTransformer = Transformer();

    textTransformer.ingest(text);
    textTransformer.createEmbedding(bigrams);

    return textTransformer;
  };

  /**
   * fromFiles
   * Create a new transformer model from .txt files
   * (if none, default to the bootstrap)
   */

  const fromFiles = async files => {
    const textTransformer = Transformer();

    if (files) {
      await textTransformer.train({
        name: datasetName,
        files
      });
    } else {

      // Default to bootstrap

      await textTransformer.train({
        name: DEFAULT_DATASET.name,
        files: DEFAULT_DATASET.files
      });
    }

    return textTransformer;
  };

  /**
   * complete
   * Pass query to the transformer and return
   * the highest-ranked completion
   */

  const complete = query => (
    transformer.getCompletions(query).completion
  );

  // Language API (extends Transformer)

  await init();

  return {
    ...transformer,

    fromTrainingData,
    fromFiles,
    complete
  };
};
