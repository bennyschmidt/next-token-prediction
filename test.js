const { dirname } = require('path');
const __root = dirname(require.main.filename);

const {
  Language: LanguageModel,
  Image: ImageModel
} = require('./models');

const DefaultDataset = require(`${__root}/training/datasets/Default`);
const DefaultImageDataset = require(`${__root}/training/datasets/images/Default`);

/**
 * Language model tests
 */

const withDataset = async (dataset, query) => {
  const agent = await LanguageModel({
    dataset
  });

  console.log(`query: ${query}`);

  // Log next token prediction

  console.log(
    'getTokenPrediction >>',
    `query: ${query}`,
    agent.getTokenPrediction(query)
  );
};

const withBootstrap = async query => {
  // Bootstrap with a default dataset

  const agent = await LanguageModel({
    bootstrap: true
  });

  // Log completions

  console.log(
    'getCompletions >>',
    `query: ${query}`,
    agent.getCompletions(query)
  );
};

const withFiles = async (files, query) => {
  const agent = await LanguageModel({
    files
  });

  // Log completion

  console.log(
    'complete >>',
    `query: ${query}`,
    agent.complete(query)
  );
};

const runTests = async () => {
  // Unit: Run different queries in isolation
  //       with different datasets

  await withDataset(DefaultDataset, 'luckily');

  await withDataset(DefaultDataset, 'with all');

  await withDataset(DefaultDataset, 'grass');

  await withDataset(DefaultDataset, 'unless');

  // e2e: Run training on user provided files

  await withFiles(['the-phantom-of-the-opera'], 'eloquence');

  // e2e: Run training from bootstrap then query

  await withBootstrap('people');
};

/**
 * Image model tests
 */

const withImageDataset = async (dataset, seedPixel) => {
  const agent = await ImageModel({
    dataset
  });

  console.log(`pixel: ${seedPixel}`)

  // Log next pixel prediction

  console.log(
    'getPixelPrediction >>',
    `pixel: ${seedPixel}`,
    agent.getPixelPrediction(seedPixel)
  );

  // Log sequence generation

  console.log(
    'getGradientCompletion >>',
    `pixel: ${seedPixel}`,
    agent.render(agent.generate(seedPixel))
  );
};

const runImageTests = async () => {
  // Unit: Predict a gradient from common starter colors

  await withImageDataset(DefaultImageDataset, '#000000');

  await withImageDataset(DefaultImageDataset, '#ffffff');

  await withImageDataset(DefaultImageDataset, '#c6e3ff');

  console.log('Done.');
};

(async () => {
  await runTests();

  await runImageTests();
})();
