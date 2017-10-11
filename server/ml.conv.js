const convnet = require('convnetjs')
const fs = require('fs')
const path = require('path')

const trainingSet = JSON.parse(fs.readFileSync(path.join(__dirname, '/data/training.json')))

const trainingData = trainingSet.map(function (data) {
  const vol = new convnet.Vol(28, 28, 1)
  data.input.forEach((d, i) => (vol.w[i] = d))
  return {
    input: vol,
    output: data.output
  }
})

const layers = []
layers.push({type: 'input', out_sx: 28, out_sy: 28, out_depth: 1})
layers.push({type: 'conv', sx: 6, filters: 16, stride: 1, pad: 2, activation: 'relu'})
layers.push({type: 'pool', sx: 2, stride: 2})
layers.push({type: 'conv', sx: 6, filters: 20, stride: 1, pad: 2, activation: 'relu'})
layers.push({type: 'pool', sx: 2, stride: 2})
layers.push({type: 'conv', sx: 6, filters: 20, stride: 1, pad: 2, activation: 'relu'})
layers.push({type: 'pool', sx: 2, stride: 2})
layers.push({type: 'regression', num_neurons: 10})

const CNN = new convnet.Net()
CNN.makeLayers(layers)
const trainer = new convnet.Trainer(CNN, {method: 'adadelta', l2_decay: 0.001})

trainingData.forEach((data, i) => {
  if (i % 50 === 0) {
    const output = data.output.findIndex((v) => v === 1)
    console.log('stats ', i, trainer.train(data.input, data.output), 'prediction:', CNN.getPrediction(), 'answer:', output)
  } else {
    trainer.train(data.input, data.output)
  }
})

fs.writeFileSync(path.join(__dirname, '/data/model.conv.json'), JSON.stringify(CNN.toJSON()))
