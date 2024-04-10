const { dirname } = require('path');
const __root = dirname(require.main.filename);

const { Language: LanguageModel } = require('./models');
const OpenSourceBooksDataset = require(`${__root}/training/datasets/OpenSourceBooks`);

const withDataset = async query => {
  const agent = await LanguageModel({
    dataset: OpenSourceBooksDataset
  });

  // Or bootstrap with a default dataset
  // const agent = await LanguageModel({
  //   bootstrap: true
  // });

  console.log(`query: ${query}`);

  // Log next token prediction

  console.log(
    'getSingleTokenPrediction >>',
    `query: ${query}`,
    agent.getSingleTokenPrediction(query)
  );

  // Log next token sequence prediction (5 tokens)

  console.log(
    'getTokenSequencePrediction >>',
    `query: ${query}`,
    agent.getTokenSequencePrediction(query, 5)
  );

  // Log completions

  console.log(
    'getCompletions >>',
    `query: ${query}`,
    agent.getCompletions(query)
  );

  // Log chat completion

  console.log(
    'complete >>',
    `query: ${query}`,
    agent.complete(query)
  );
};

const withTraining = async query => {
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

const withFiles = async query => {
  const agent = await LanguageModel({
    files: ['brave-new-world']
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

  await withDataset('what');

  await withDataset('What is');

  await withDataset('what is the');

  await withDataset('hopefully');

  await withDataset('where is');

  // e2e: Run full training then query

  await withTraining('The sun');

  // e2e: Run full training on user provided files

  await withFiles('Society');
};

runTests();
