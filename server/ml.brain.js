const brain = require('brain.js')
const fs = require('fs')
const path = require('path')

const trainingSet = JSON.parse(fs.readFileSync(path.join(__dirname, '/data/training.json')))

const mlp = new brain.NeuralNetwork({
  activation: 'relu', // activation function
  hiddenLayers: [16, 16],
  learningRate: 0.01 // global learning rate, useful when training using streams
})

console.log('training network')

mlp.train(trainingSet, {
  learningRate: 0.01,
  iterations: trainingSet.length,
  errorThresh: 0.005,
  log: true,
  logPeriod: 1
})

fs.writeFileSync(path.join(__dirname, '/data/model.json'), mlp.toJSON())
