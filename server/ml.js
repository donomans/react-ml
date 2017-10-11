const synaptic = require('synaptic')
// const convnet = require('convnetjs')
const brain = require('brain.js')
const fs = require('fs')
const path = require('path')

// const labels = fs.readFileSync(path.join(__dirname, '/data/train-labels-idx1-ubyte'))
// const images = fs.readFileSync(path.join(__dirname, '/data/train-images-idx3-ubyte'))
const trainingSet = JSON.parse(fs.readFileSync(path.join(__dirname, '/data/training.json')))
// const layers = [];
// layers.push({ type: 'input', out_sy: 28, out_sx: 28, out_depth: 1 })
// layers.push({ type: 'input', out_sy: 28, out_sx: 28, out_depth: 1 })
// layers.push({ type: 'input', out_sy: 28, out_sx: 28, out_depth: 1 })
// layers.push({ type: 'softmax', num_classes: 10 })
// const CNN = convnet.Net();

const inputLayer = new synaptic.Layer(784)
const hiddenFirst = new synaptic.Layer(16)
const hiddenSecond = new synaptic.Layer(16)
const outputLayer = new synaptic.Layer(10)

inputLayer.project(hiddenFirst)
hiddenFirst.project(hiddenSecond)
hiddenSecond.project(outputLayer)

const mlp = new synaptic.Network({
  input: inputLayer,
  hidden: [hiddenFirst, hiddenSecond],
  output: outputLayer
})

// console.log('building training set')
// const trainingSet = []
// for (let image = 0; image <= 59999; image++) {
//   const pixelData = []
//   for (let y = 0; y <= 27; y++) {
//     for (let x = 0; x <= 27; x++) {
//       pixelData.push(images[(image * 28 * 28) + (x + (y * 28)) + 16])
//     }
//   }
//   const labelValue = parseInt(labels[image + 8], 10)
//   const outputValue = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
//   outputValue[labelValue] = 1
//   const trainingData = {
//     input: pixelData,
//     output: outputValue
//   }
//   trainingSet.push(trainingData)
// }
// console.log('writing training data to disk')
// fs.writeFileSync(path.join(__dirname, '/data/training.json'), JSON.stringify(trainingSet))
// const output = cnn.activate(pixelData)
// cnn.propagate()

console.log('training network')
const trainer = new synaptic.Trainer(mlp)
// trainingSet.forEach(function (data, index) {
//   const activation = mlp.activate(data.input)
//   if (index % 200 === 0) {
//     data.output.find(function (val, i) {
//       if (val === 1) {
//         console.log('error %', 1 - activation[i], 'for iteration', index);
//       }
//     })
//   }
//   mlp.propagate(0.01, data.output)
// })

let loops = trainingSet.length / 500
for (let index = 0; index <= loops; index++) {
  const t = trainingSet.slice(loops * index, loops)
  console.log('training a set of', loops, 'iterations')
  trainer.train(t, {
    rate: 0.01,
    iterations: t.length,
    error: 0.005,
    log: 20,
    cost: synaptic.Trainer.cost.CROSS_ENTROPY,
    // schedule: {
    //   every: 20,
    //   do: function (data) {
    //     console.log('error', data.error, 'iterations', data.iterations, 'rate', data.rate)
    //   }
    // }
  })
}

fs.writeFileSync(path.join(__dirname, '/data/model.json'), JSON.stringify(mlp.toJSON()))
