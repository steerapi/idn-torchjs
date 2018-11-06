var torchjs = require('@idn/torchjs');
var path = require('path');

var script_module = new torchjs.ScriptModule(path.join(__dirname, 'lenet.pt'));
var tensor = torchjs.ones([1, 1, 28, 28], false);

const { performance } = require('perf_hooks');

script_module.cuda();
let start, end;
start = performance.now();
let otensor = script_module.forward(tensor);
end = performance.now();
console.log(`      gpu: ${end - start} ms`);
