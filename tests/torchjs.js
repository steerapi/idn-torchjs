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
    this.timeout(10000);
    var script_module = new torchjs.ScriptModule(path.join(__dirname, 'lenet.pt'));
    var tensor = torchjs.ones([1, 1, 28, 28], false);

    const { performance } = require('perf_hooks');

    let start, end;
    start = performance.now();
    let otensor = script_module.forward(tensor);
    end = performance.now();
    console.log(`      cpu: ${end - start} ms`);
  });
  it('should check if cuda is available', function() {
    var script_module = new torchjs.ScriptModule(path.join(__dirname, 'lenet.pt'));
    const isAvailable = script_module.is_cuda_available();
  });
  it('should run the model on gpu or skip if cuda is not available', function() {
    this.timeout(10000);
    var script_module = new torchjs.ScriptModule(path.join(__dirname, 'lenet.pt'));
    if(!script_module.is_cuda_available()){
      return;
    }
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
