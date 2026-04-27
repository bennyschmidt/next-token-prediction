const { dirname } = require('path');
const __root = dirname(require.main.filename);

const Phrase = require('../Phrase');

const { combineDocuments } = require('../../utils');

const DEFAULT_DATASET = require(`${__root}/training/datasets/Default`);

const NEW_DATASET_NAME = 'New Dataset';

/**
 * Create a language model from a
 * dataset or specify a list of files.
 */

module.exports = async ({
  name,
  dataset,
  files,
  bootstrap = false
} = {}) => {
  let datasetName = name || NEW_DATASET_NAME;
  let trainingData = null;
  let phraseModel;

  /**
   * init
   * Train, create context and initialize the
   * model API for use.
   */

  const init = async () => {
    /**
     * If bootstrap is true, start the LM
     * with default training data.
     */

    if (bootstrap) {
      files = DEFAULT_DATASET.files;
      datasetName = DEFAULT_DATASET.name;
    }

    /**
     * If a dataset is provided, create a model
     * from training data.
     */

    if (dataset?.name) {
      datasetName = dataset.name;

      // concatenate and store all reference text

      const text = await combineDocuments(dataset.files);

      // build training data object

      trainingData = {
        text
      };
    }

    /**
     * Instantiate a phrase model either from
     * existing training data or by providing
     * files.
     */

    if (trainingData) {

      // Skips initial extraction and training
      // just instantiates the model.

      phraseModel = fromTrainingData(trainingData);
    } else {

      // Performs an intensive training operation
      // using provided files, creating a model
      // structure and corresponding values.

      phraseModel = await fromFiles(files);
    }
  };

  /**
   * fromTrainingData
   * Create a new phrase model.
   */

  const fromTrainingData = ({
    text
  }) => {
    const textPhrase = Phrase();

    textPhrase.ingest(text);
    textPhrase.createContext();

    return textPhrase;
  };

  /**
   * fromFiles
   * Create a new phrase model from .txt files
   * (if none, default to the bootstrap).
   */

  const fromFiles = async files => {
    const textPhrase = Phrase();

    if (files) {
      await textPhrase.train({
        name: datasetName,
        files
      });
    } else {

      // Default to bootstrap

      await textPhrase.train({
        name: DEFAULT_DATASET.name,
        files: DEFAULT_DATASET.files
      });
    }

    return textPhrase;
  };

  /**
   * complete
   * Pass query to the phrase model and return
   * the highest-ranked completion.
   */

  const complete = query => (
    phraseModel.getCompletions(query).completion
  );

  // Language API (extends Phrase)

  await init();

  return {
    ...phraseModel,

    fromTrainingData,
    fromFiles,
    complete
  };
};
