const readline = require('node:readline');
const { stdin: input, stdout: output } = require('node:process');
const { dirname } = require('path');
const __root = dirname(require.main.filename);

const { Language: LM } = require('../../models');
const OpenSourceBooksDataset = require(`${__root}/training/datasets/OpenSourceBooks`);

const rl = readline.createInterface({ input, output });

const MyLanguageModel = async (verbose = false) => {
  const agent = await LM({
    dataset: OpenSourceBooksDataset
  });

  const print = input => {
    if (
      input.toLowerCase() === 'bye' ||
      input.toLowerCase() === 'exit'
    ) {
      console.log('Bye!');
      rl.close();
      process.exit();
    }

    const {
      completion,
      completions,
      rankedTokenList
    } = agent.getCompletions(input);

    console.log(`${completion}`, '\n');

    if (verbose) {
      console.log('Top K (next words):', rankedTokenList);
      console.log('Top K (next phrases):', completions);
    }

    prompt();
  };

  const prompt = () => {
    rl.question('Type something... (press ENTER to suggest) ', print);
  };

  prompt();
};

MyLanguageModel();
