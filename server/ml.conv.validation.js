const convnet = require('convnetjs')
const fs = require('fs')
const path = require('path')
const model = JSON.parse(fs.readFileSync(path.join(__dirname, '/models/model.conv.json')))

const labels = fs.readFileSync(path.join(__dirname, '/data/t10k-labels-idx1-ubyte'))
const images = fs.readFileSync(path.join(__dirname, '/data/t10k-images-idx3-ubyte'))

const CNN = new convnet.Net()
CNN.fromJSON(model)

console.log('building validation set')
const testingSet = []
for (let image = 0; image <= 9999; image++) {
  const pixelData = []
  for (let y = 0; y <= 27; y++) {
    for (let x = 0; x <= 27; x++) {
      pixelData.push(images[(image * 28 * 28) + (x + (y * 28)) + 16])
    }
  }
  const labelValue = parseInt(labels[image + 8], 10)
  const outputValue = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
  outputValue[labelValue] = 1
  const vol = new convnet.Vol(28, 28, 1)
  pixelData.forEach((d, i) => (vol.w[i] = d))
  const trainingData = {
    input: vol,
    output: outputValue
  }
  testingSet.push(trainingData)
}
