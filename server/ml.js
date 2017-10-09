const synaptic = require('synaptic');
const fs = require('fs');

const labels = fs.readFileSync(__dirname + '/data/train-labels-idx1-ubyte');
const images = fs.readFileSync(__dirname + '/data/train-images-idx3-ubyte');
