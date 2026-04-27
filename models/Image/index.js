const { dirname } = require('path');
const __root = dirname(require.main.filename);

const Gradient = require('../Gradient');
const { combineImages } = require('../../utils');

const NEW_MODEL_NAME = "New Image Model";

/**
 * Image Model
 * Manages training on image data and generating
 * next-pixel sequences.
 */
 
module.exports = async ({
  name,
  pixels,
  dataset,
  files,
  bootstrap = false
} = {}) => {
  let modelName = name || NEW_MODEL_NAME;
  let gradientModel;

  /**
   * init
   * Handles the initial setup, determining if the model
   * loads from raw pixel arrays or scans the file system.
   */

  const init = async () => {
    if (dataset?.files) {
      modelName = dataset.name;
      files = dataset.files;
    }

    if (pixels) {
      gradientModel = fromTrainingData({ pixels });
    } else {
      gradientModel = await fromImages(files);
    }
  };

  /**
   * fromTrainingData
   * Directly instantiates a Gradient model from an existing
   * array of hex pixels.
   */

  const fromTrainingData = ({ pixels }) => {
    const instance = Gradient();

    instance.ingest(pixels);
    instance.createContext();

    return instance;
  };

  /**
   * fromImages
   * Aggregates PNG files from the training folder and
   * performs the frequency tally training.
   */

  const fromImages = async (imageFiles) => {
    const instance = Gradient();

    if (imageFiles && imageFiles.length > 0) {
      await instance.train({
        name: modelName,
        files: imageFiles
      });
    }

    return instance;
  };

  /**
   * generate
   * Simple API to get the highest-ranked pixel sequence
   * based on a starting color.
   */

  const generate = seedPixel => (
    gradientModel.getCompletions(seedPixel).completion
  );

  /**
   * render
   * Converts the flat string sequence back into a
   * manipulatable array.
   */

  const render = (pixelString) => {
    return pixelString.split(' ');
  };

  await init();

  return {
    ...gradientModel,

    fromTrainingData,
    fromImages,
    generate,
    render
  };
};
