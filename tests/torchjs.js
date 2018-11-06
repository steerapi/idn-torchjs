var torchjs = require('bindings')('torchjs');
var path = require('path');
var assert = require('assert');

describe('torchjs', function() {
  it('should create ones', function() {
    var tensor = torchjs.ones([10], false);
    console.log(tensor);
    assert.deepEqual(tensor.data, new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
  });
  it('should create zeros', function() {
    var tensor = torchjs.zeros([10], false);
    console.log(tensor);
    assert.deepEqual(tensor.data, new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
  });
  it('should run the model on cpu', function() {
    var script_module = new torchjs.ScriptModule(path.join(__dirname, 'lenet.pt'));
    var tensor = torchjs.ones([1, 1, 28, 28], false);

    const { performance } = require('perf_hooks');

    let start, end;
    start = performance.now();
    let otensor = script_module.forward(tensor);
    end = performance.now();
    console.log(`      cpu: ${end - start} ms`);
  });
  it('should run the model on gpu', function() {
    var script_module = new torchjs.ScriptModule(path.join(__dirname, 'lenet.pt'));
    var tensor = torchjs.ones([1, 1, 28, 28], false);

    const { performance } = require('perf_hooks');

    script_module.cuda();
    let start, end;
    start = performance.now();
    let otensor = script_module.forward(tensor);
    end = performance.now();
    console.log(`      gpu: ${end - start} ms`);
  });
});
