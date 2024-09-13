const { dirname } = require('path');
const __root = dirname(require.main.filename);

const { Language: LM } = require('next-token-prediction');
const OpenSourceBooksDataset = require(`${__root}/training/datasets/OpenSourceBooks`);

const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

let agent = {};

/**
 * onComplete
 * Endpoint to handle word prediction
 */

const onComplete = async (req, res) => {
  const { input } = req.body;

  const {
    completion,
    completions,
    rankedTokenList
  } = agent.getCompletions(input);

  res.json({
    completion,
    completions,
    rankedTokenList
  });
};

/**
 * onIndex
 * Endpoint to serve the page
 */

const onIndex = (_, res) => res.sendFile(__dirname + '/index.html');

/**
 * onStart
 * Handle server start
 */

const onStart = async () => {
  // Boot up LM

  agent = await LM({
    dataset: OpenSourceBooksDataset
  });

  // Configure router and listen for requests

  app.use(bodyParser.json());
  app.get('/', onIndex);
  app.post('/complete', onComplete);
  app.listen(port, () => console.log(`Listening on port ${port}...`));
};

// Server start

onStart();
