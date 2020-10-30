var torchjs = require('../');
var path = require('path');

var script_module = new torchjs.ScriptModule(path.join(__dirname, '../tests/detr.pt'));
var tensor = torchjs.ones([1, 3, 100, 100], false);

const { performance } = require('perf_hooks');

// script_module.cuda();
let start, end;
start = performance.now();
let otensor = script_module.forward(tensor);
console.log(otensor)
end = performance.now();
console.log(`      gpu: ${end - start} ms`);
