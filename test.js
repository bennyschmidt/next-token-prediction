const { dirname } = require('path');
const __root = dirname(require.main.filename);

const { Language: LanguageModel } = require('./models');
const DefaultDataset = require(`${__root}/training/datasets/Default`);

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

runTests();
