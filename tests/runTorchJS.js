var torchjs = require('bindings')('torchjs');

// var assert = require('assert');
// describe('torchjs', function() {
//   it('should calculate pi synchronously', function() {});
// });

// var script_module = new torchjs.ScriptModule("model.pt");
// var data = torchjs.ones([1, 3, 224, 224], false);
var script_module = new torchjs.ScriptModule('lenet.pt');
var data = torchjs.ones([1, 1, 28, 28], false);

const { performance } = require('perf_hooks');

let start, end;
start = performance.now();
for (let i = 0; i < 1000; i++) {
  let output = script_module.forward(data);
}
end = performance.now();
console.log(`cpu: ${end - start} ms`);

script_module.cuda();
start = performance.now();
for (let i = 0; i < 1000; i++) {
  let output = script_module.forward(data);
}
end = performance.now();
console.log(`gpu: ${end - start} ms`);

// var data1 = torchjs.ones([1, 10], false);
// var data0 = torchjs.zeros([1, 10], false);
// console.log(data0);
// console.log(data1);
// data1.tensor = data0.tensor;
// console.log(data1);

// console.log(torchjs);
// var calculations = 10000000;

// function printResult(type, pi, ms) {
//     console.log(type, "method:");
//     console.log(
//         "\tπ ≈ " + pi + " (" + Math.abs(pi - Math.PI) + " away from actual)"
//     );
//     console.log("\tTook " + ms + "ms");
//     console.log();
// }

// function runSync() {
//     var start = Date.now();
//     // Estimate() will execute in the current thread,
//     // the next line won't return until it is finished
//     var result = addon.calculateSync(calculations);
//     printResult("Sync", result, Date.now() - start);
// }

// function runAsync() {
//     // how many batches should we split the work in to?
//     var batches = 16;
//     var ended = 0;
//     var total = 0;
//     var start = Date.now();

//     function done(err, result) {
//         if (err) {
//             return;
//         }
//         total += result;

//         // have all the batches finished executing?
//         if (++ended == batches) {
//             printResult("Async", total / batches, Date.now() - start);
//         }
//     }

//     // for each batch of work, request an async Estimate() for
//     // a portion of the total number of calculations
//     for (var i = 0; i < batches; i++) {
//         addon.calculateAsync(calculations / batches, done);
//     }
// }

// runSync();
// runAsync();
